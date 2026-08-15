//! Vision tower (SigLIP/CLIP-style encoder + projector).
//! Assimilates design from LLaVA and Moondream.
//! Native Rust, ndarray-based.

use crate::config::{VisionConfig, VisionTowerKind};
use ndarray::{Array1, Array2, Array3, s};

// ---------------------------------------------------------------------------
// Image loading: disk → normalized [C, H, W] f32 tensor
// ---------------------------------------------------------------------------

/// Load an image from disk, resize to `target_size`×`target_size`, convert to RGB,
/// and return a normalized f32 tensor of shape `[3, target_size, target_size]`.
/// Pixel values are normalized to the range used by SigLIP/CLIP: `(pixel / 255.0 - 0.5) * 2.0`
/// (i.e. roughly [-1, 1]).
pub fn load_image(path: &str, target_size: usize) -> Result<Array3<f32>, String> {
    let img = image::open(path).map_err(|e| format!("image::open({path}): {e}"))?;
    let rgb = img.into_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let (w, h) = resized.dimensions();
    let mut pixels = Array3::zeros((3, h as usize, w as usize));
    for y in 0..h {
        for x in 0..w {
            let px = resized.get_pixel(x, y);
            // Normalize to [-1, 1]
            pixels[[0, y as usize, x as usize]] = (px[0] as f32 / 255.0 - 0.5) * 2.0;
            pixels[[1, y as usize, x as usize]] = (px[1] as f32 / 255.0 - 0.5) * 2.0;
            pixels[[2, y as usize, x as usize]] = (px[2] as f32 / 255.0 - 0.5) * 2.0;
        }
    }
    Ok(pixels)
}

// ---------------------------------------------------------------------------
// GGUF vision tower weight loader
// ---------------------------------------------------------------------------

/// Try to load `VisionTowerWeights` from a GGUF file.
/// Supports LLaVA and Moondream naming conventions with several possible prefixes:
/// `v.`, `vision.`, `vision_tower.`, `visual.`, or bare (no prefix).
/// Also supports the LlamaV (LLaVA) GGUF convention where tensors use
/// `vision.encoder.layers.0.attn_norm.weight` etc.
pub fn load_vision_gguf(path: &str) -> Result<(VisionTowerWeights, VisionConfig), String> {
    use crate::gguf::GgufBuffer;

    let buf = GgufBuffer::open(path)?;

    // ---- Determine naming prefix ----
    let prefixes = ["v.", "vision.", "vision_tower.", "visual.", ""];
    let prefix = prefixes
        .iter()
        .find(|p| {
            // Check if the prefix exists by looking for known tensors
            let candidates = [
                format!("{}cls_token", p),
                format!("{}patch_embed.weight", p),
                format!("{}patch_embed", p),
                format!("{}pos_embed", p),
            ];
            candidates.iter().any(|c| buf.tensor(c).is_some())
        })
        .ok_or_else(|| {
            format!(
                "no known vision tower prefix found in {path}. \
                 Tried: v., vision., vision_tower., visual., bare. \
                 Expected tensors like cls_token, patch_embed, pos_embed, projector"
            )
        })?;

    eprintln!("[vision] detected prefix: \"{prefix}\"");

    // Helper: try multiple candidate names for a 1D weight
    let load_arr1 = |candidates: &[String]| -> Result<Array1<f32>, String> {
        for name in candidates {
            if buf.tensor(name).is_some() {
                return buf.dequant_arr1(name);
            }
        }
        Err(format!(
            "none of {:?} found",
            candidates
        ))
    };

    // Helper: try multiple candidate names for a 2D weight
    let load_arr2 = |candidates: &[String]| -> Result<Array2<f32>, String> {
        for name in candidates {
            if buf.tensor(name).is_some() {
                return buf.dequant_arr2(name);
            }
        }
        Err(format!(
            "none of {:?} found",
            candidates
        ))
    };

    // ---- Load non-layer weights ----
    let cls_token = load_arr1(&[
        format!("{}cls_token", prefix),
        format!("{}cls_token.weight", prefix),
    ])?;
    let patch_embed_data = load_arr2(&[
        format!("{}patch_embed", prefix),
        format!("{}patch_embed.weight", prefix),
        format!("{}embedding.patch_embedding.weight", prefix),
    ])?;
    let pos_embed = load_arr2(&[
        format!("{}pos_embed", prefix),
        format!("{}pos_embed.weight", prefix),
        format!("{}position_ids", prefix),
    ])?;
    let pre_norm = load_arr1(&[
        format!("{}pre_norm", prefix),
        format!("{}pre_norm.weight", prefix),
        format!("{}ln_pre.weight", prefix),
        format!("{}pre_layernorm.weight", prefix),
    ])?;
    let post_norm = load_arr1(&[
        format!("{}post_norm", prefix),
        format!("{}post_norm.weight", prefix),
        format!("{}ln_post.weight", prefix),
        format!("{}transformer.ln_f.bias", prefix),
        format!("{}transformer.ln_f.weight", prefix),
    ])?;
    let projector = load_arr2(&[
        format!("{}projector", prefix),
        format!("{}projector.weight", prefix),
        format!("{}projector.0.weight", prefix),
        format!("{}multi_modal_projector.linear.weight", prefix),
        format!("{}mm.projector.0.weight", prefix),
    ])?;

    // ---- Count layers ----
    let vd = cls_token.len();
    let mut n_layers = 0usize;
    let layer_patterns = [
        format!("{}encoder.layers.0.attn_norm", prefix),
        format!("{}encoder.layers.0.attn_norm.weight", prefix),
        format!("{}layers.0.attn_norm", prefix),
        format!("{}layers.0.attn_norm.weight", prefix),
        format!("{}transformer.resblocks.0.attn_norm", prefix),
        format!("{}transformer.resblocks.0.ln_1.weight", prefix),
    ];
    // Check if layer pattern exists
    if layer_patterns.iter().any(|p| buf.tensor(p).is_some()) {
        // Count layers by probing
        for i in 0..99 {
            let probes = [
                format!("{}encoder.layers.{}.attn_norm.weight", prefix, i),
                format!("{}encoder.layers.{}.attn_norm", prefix, i),
                format!("{}layers.{}.attn_norm.weight", prefix, i),
                format!("{}layers.{}.attn_norm", prefix, i),
                format!("{}transformer.resblocks.{}.ln_1.weight", prefix, i),
            ];
            if probes.iter().any(|p| buf.tensor(p).is_some()) {
                n_layers = i + 1;
            } else {
                break;
            }
        }
    }
    if n_layers == 0 {
        // Fallback: assume same as n_vision_layers from metadata, or default 12
        n_layers = 12;
    }
    eprintln!("[vision] detected {n_layers} encoder layers, dim={vd}");

    // ---- Build layer names ----
    fn layer_name(prefix: &str, i: usize, weight: &str, suffix: &str) -> Vec<String> {
        let variants = [
            format!("{}encoder.layers.{}.{}{}", prefix, i, weight, suffix),
            format!("{}encoder.layers.{}.{}", prefix, i, weight),
            format!("{}layers.{}.{}{}", prefix, i, weight, suffix),
            format!("{}layers.{}.{}", prefix, i, weight),
            format!("{}transformer.resblocks.{}.{}{}", prefix, i, weight, suffix),
        ];
        variants.to_vec()
    }

    // ---- Load each layer ----
    let mut layers = Vec::with_capacity(n_layers);
    let inter = projector.shape()[0]; // use proj_dim as estimate for intermediate

    for i in 0..n_layers {
        let attn_norm = load_arr1(&layer_name(prefix, i, "attn_norm", ".weight"))?;
        let wq = load_arr2(&layer_name(prefix, i, "wq", ".weight"))?;
        let wk = load_arr2(&layer_name(prefix, i, "wk", ".weight"))?;
        let wv = load_arr2(&layer_name(prefix, i, "wv", ".weight"))?;
        let wo = load_arr2(&layer_name(prefix, i, "wo", ".weight"))?;
        let ffn_norm = load_arr1(&layer_name(prefix, i, "ffn_norm", ".weight"))?;
        let w1 = load_arr2(&layer_name(prefix, i, "w1", ".weight"))?;
        let w2 = load_arr2(&layer_name(prefix, i, "w2", ".weight"))?;
        let w3 = load_arr2(&layer_name(prefix, i, "w3", ".weight"))?;

        layers.push(VisionLayer {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w1,
            w2,
            w3,
        });
    }

    let cfg = VisionConfig {
        enabled: true,
        tower: VisionTowerKind::SigLIP,
        image_size: ((pos_embed.shape()[0] - 1) as f64).sqrt() as usize * 16, // estimate from n_patches
        patch_size: 16, // default
        vision_dim: vd,
        n_vision_layers: n_layers,
        n_vision_heads: 12, // common default
        vision_intermediate: inter,
        proj_dim: projector.shape()[0],
    };

    Ok((
        VisionTowerWeights {
            cls_token,
            patch_embed: patch_embed_data,
            pos_embed,
            pre_norm,
            layers,
            post_norm,
            projector,
        },
        cfg,
    ))
}

// ---------------------------------------------------------------------------
// Single ViT transformer layer (pre-norm, MHA + SwiGLU FFN)
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct VisionLayer {
    /// Attention pre-norm weight  [vision_dim]
    pub attn_norm: Array1<f32>,
    /// Q projection          [vision_dim, vision_dim]
    pub wq: Array2<f32>,
    /// K projection          [vision_dim, vision_dim]
    pub wk: Array2<f32>,
    /// V projection          [vision_dim, vision_dim]
    pub wv: Array2<f32>,
    /// Output projection     [vision_dim, vision_dim]
    pub wo: Array2<f32>,
    /// FFN pre-norm weight   [vision_dim]
    pub ffn_norm: Array1<f32>,
    /// Gate projection       [vision_intermediate, vision_dim]
    pub w1: Array2<f32>,
    /// Down projection       [vision_dim, vision_intermediate]
    pub w2: Array2<f32>,
    /// Up projection         [vision_intermediate, vision_dim]
    pub w3: Array2<f32>,
}

// ---------------------------------------------------------------------------
// Vision tower weights (SigLIP-style ViT + projector)
// ---------------------------------------------------------------------------
#[derive(Clone)]
#[allow(dead_code)]
pub struct VisionTowerWeights {
    /// CLS token embedding   [vision_dim]
    pub cls_token: Array1<f32>,
    /// Patch embedding       [vision_dim, patch_size * patch_size * 3]  (RGB)
    pub patch_embed: Array2<f32>,
    /// Position embeddings   [1 + n_patches, vision_dim]
    pub pos_embed: Array2<f32>,
    /// Pre-encoder norm      [vision_dim]
    pub pre_norm: Array1<f32>,
    /// Encoder transformer layers
    pub layers: Vec<VisionLayer>,
    /// Post-encoder norm     [vision_dim]
    pub post_norm: Array1<f32>,
    /// Projector (vision -> text dim)  [proj_dim, vision_dim]
    pub projector: Array2<f32>,
}

#[allow(dead_code)]
impl VisionTowerWeights {
    /// Encode image pixels into projected patch token embeddings.
    ///
    /// `pixels`: [C, H, W] — normalized image tensor (C=1 for grayscale, C=3 for RGB).
    ///
    /// Returns `[n_patches, proj_dim]` — one vector per image patch, projected
    /// into the text-model's latent space.
    pub fn encode(&self, pixels: &Array3<f32>, cfg: &VisionConfig) -> Array2<f32> {
        let channels = pixels.shape()[0];
        let height = pixels.shape()[1];
        let width = pixels.shape()[2];
        let ps = cfg.patch_size;
        let n_patches_h = height / ps;
        let n_patches_w = width / ps;
        let n_patches = n_patches_h * n_patches_w;
        let vd = cfg.vision_dim;

        // ---- 1. Patch embed ----
        let mut x = Array2::zeros((n_patches, vd));
        for ph in 0..n_patches_h {
            for pw in 0..n_patches_w {
                let idx = ph * n_patches_w + pw;
                let mut flat = Array1::zeros(ps * ps * channels);
                let mut f = 0;
                for c in 0..channels {
                    for i in 0..ps {
                        for j in 0..ps {
                            flat[f] = pixels[[c, ph * ps + i, pw * ps + j]];
                            f += 1;
                        }
                    }
                }
                let proj = self.patch_embed.dot(&flat);
                x.row_mut(idx).assign(&proj);
            }
        }

        // ---- 2. Add position embeddings (skip CLS row) ----
        for i in 0..n_patches {
            let mut row = x.row_mut(i);
            for j in 0..vd {
                row[j] += self.pos_embed[[i + 1, j]];
            }
        }

        // ---- 3. Prepend CLS token ----
        let n_seq = 1 + n_patches;
        let mut seq = Array2::zeros((n_seq, vd));
        seq.row_mut(0).assign(&self.cls_token);
        seq.slice_mut(s![1.., ..]).assign(&x);
        // Add CLS position embedding
        for j in 0..vd {
            seq[[0, j]] += self.pos_embed[[0, j]];
        }

        // ---- 4. Pre-norm ----
        for i in 0..n_seq {
            let n = crate::engine::rmsnorm_row(&seq.row(i).to_owned(), &self.pre_norm, crate::engine::EPS);
            seq.row_mut(i).assign(&n);
        }

        // ---- 5. Encoder transformer layers ----
        for layer in &self.layers {
            Self::forward_vit_layer(&mut seq, layer, cfg);
        }

        // ---- 6. Post-norm ----
        for i in 0..n_seq {
            let n = crate::engine::rmsnorm_row(&seq.row(i).to_owned(), &self.post_norm, crate::engine::EPS);
            seq.row_mut(i).assign(&n);
        }

        // ---- 7. Project patch tokens (skip CLS) to text dim ----
        let mut out = Array2::zeros((n_patches, cfg.proj_dim));
        for i in 0..n_patches {
            let patch = seq.row(i + 1);
            let proj = self.projector.dot(&patch);
            out.row_mut(i).assign(&proj);
        }

        out
    }

    /// Forward one ViT transformer layer (pre-norm MHA + SwiGLU FFN).
    fn forward_vit_layer(seq: &mut Array2<f32>, layer: &VisionLayer, cfg: &VisionConfig) {
        let t = seq.nrows();
        let vd = cfg.vision_dim;
        let nh = cfg.n_vision_heads;
        let hd = vd / nh;
        let _mid = cfg.vision_intermediate;

        // ---- Attention block ----
        // Pre-norm
        let mut x_norm = Array2::zeros((t, vd));
        for i in 0..t {
            let n = crate::engine::rmsnorm_row(&seq.row(i).to_owned(), &layer.attn_norm, crate::engine::EPS);
            x_norm.row_mut(i).assign(&n);
        }

        let q = crate::engine::linear(&x_norm, &layer.wq);
        let k = crate::engine::linear(&x_norm, &layer.wk);
        let v = crate::engine::linear(&x_norm, &layer.wv);

        let mut attn_out = Array2::zeros((t, vd));
        for h in 0..nh {
            let base = h * hd;
            let q_h = q.slice(s![.., base..base + hd]);
            let k_h = k.slice(s![.., base..base + hd]);
            let v_h = v.slice(s![.., base..base + hd]);

            // Scores [t, t]
            let scores = q_h.dot(&k_h.t()) / (hd as f32).sqrt();

            // Softmax
            let mut probs = Array2::zeros((t, t));
            for i in 0..t {
                let max_v = scores.row(i).iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.row(i).iter().map(|s| (s - max_v).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for j in 0..t {
                    probs[[i, j]] = exps[j] / sum;
                }
            }

            let out_h = probs.dot(&v_h); // [t, hd]
            for i in 0..t {
                for j in 0..hd {
                    attn_out[[i, base + j]] = out_h[[i, j]];
                }
            }
        }

        // Output projection + residual
        let attn_proj = crate::engine::linear(&attn_out, &layer.wo);
        *seq = &*seq + &attn_proj;

        // ---- FFN block (SwiGLU) ----
        let mut ffn_norm = Array2::zeros((t, vd));
        for i in 0..t {
            let n = crate::engine::rmsnorm_row(&seq.row(i).to_owned(), &layer.ffn_norm, crate::engine::EPS);
            ffn_norm.row_mut(i).assign(&n);
        }

        let gate = crate::engine::silu(&crate::engine::linear(&ffn_norm, &layer.w1));
        let up = crate::engine::linear(&ffn_norm, &layer.w3);
        let hidden = &gate * &up;
        let ffn_out = crate::engine::linear(&hidden, &layer.w2);
        *seq = &*seq + &ffn_out;
    }

    /// Create a random VisionTowerWeights for testing.
    pub fn random(cfg: &VisionConfig) -> Self {
        let vd = cfg.vision_dim;
        let mid = cfg.vision_intermediate;
        let ps = cfg.patch_size;
        let n_patches = (cfg.image_size / ps).pow(2);
        let n_seq = 1 + n_patches;
        let _hd = vd / cfg.n_vision_heads;

        let mut s = 42u64;
        let mk2 = |rows: usize, cols: usize, s: &mut u64| {
            let mut a = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[[i, j]] = ((*s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
            a
        };
        let mk1 = |n: usize, s: &mut u64| {
            let mut a = Array1::zeros(n);
            for i in 0..n {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[i] = 1.0 + ((*s >> 33) as f32 / (u32::MAX as f32)) * 0.1;
            }
            a
        };

        let mut layers = Vec::with_capacity(cfg.n_vision_layers);
        for _ in 0..cfg.n_vision_layers {
            layers.push(VisionLayer {
                attn_norm: mk1(vd, &mut s),
                wq: mk2(vd, vd, &mut s),
                wk: mk2(vd, vd, &mut s),
                wv: mk2(vd, vd, &mut s),
                wo: mk2(vd, vd, &mut s),
                ffn_norm: mk1(vd, &mut s),
                w1: mk2(mid, vd, &mut s),
                w2: mk2(vd, mid, &mut s),
                w3: mk2(mid, vd, &mut s),
            });
        }

        let pos_embed = mk2(n_seq, vd, &mut s);

        VisionTowerWeights {
            cls_token: mk1(vd, &mut s),
            patch_embed: mk2(vd, ps * ps * 3, &mut s),
            pos_embed,
            pre_norm: mk1(vd, &mut s),
            layers,
            post_norm: mk1(vd, &mut s),
            projector: mk2(cfg.proj_dim, vd, &mut s),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VisionTowerKind;

    fn small_vision_cfg() -> VisionConfig {
        VisionConfig {
            enabled: true,
            tower: VisionTowerKind::SigLIP,
            image_size: 16,
            patch_size: 4,
            vision_dim: 8,
            n_vision_layers: 1,
            n_vision_heads: 2,
            vision_intermediate: 16,
            proj_dim: 16,
        }
    }

    fn make_weights(cfg: &VisionConfig) -> VisionTowerWeights {
        let vd = cfg.vision_dim;
        let mid = cfg.vision_intermediate;
        let ps = cfg.patch_size;
        let n_patches = (cfg.image_size / ps).pow(2);
        let n_seq = 1 + n_patches;
        let n_heads = cfg.n_vision_heads;
        let _hd = vd / n_heads;

        let mut s = 42u64;
        let mk2 = |rows: usize, cols: usize, s: &mut u64| {
            let mut a = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[[i, j]] = ((*s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
            a
        };
        let mk1 = |n: usize, s: &mut u64| {
            let mut a = Array1::zeros(n);
            for i in 0..n {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[i] = 1.0 + ((*s >> 33) as f32 / (u32::MAX as f32)) * 0.1;
            }
            a
        };

        // Create layer
        let layer = VisionLayer {
            attn_norm: mk1(vd, &mut s),
            wq: mk2(vd, vd, &mut s),
            wk: mk2(vd, vd, &mut s),
            wv: mk2(vd, vd, &mut s),
            wo: mk2(vd, vd, &mut s),
            ffn_norm: mk1(vd, &mut s),
            w1: mk2(mid, vd, &mut s),
            w2: mk2(vd, mid, &mut s),
            w3: mk2(mid, vd, &mut s),
        };

        // Make pos_embed: [1 + n_patches, vd]
        let pos_embed = mk2(n_seq, vd, &mut s);

        VisionTowerWeights {
            cls_token: mk1(vd, &mut s),
            patch_embed: mk2(vd, ps * ps * 3, &mut s),
            pos_embed,
            pre_norm: mk1(vd, &mut s),
            layers: vec![layer],
            post_norm: mk1(vd, &mut s),
            projector: mk2(cfg.proj_dim, vd, &mut s),
        }
    }

    #[test]
    fn vision_tower_forward_produces_valid_output() {
        let cfg = small_vision_cfg();
        let weights = make_weights(&cfg);

        let image_size = cfg.image_size;
        let n_patches = (image_size / cfg.patch_size).pow(2);
        let _ps = cfg.patch_size;

        // Create synthetic RGB image [3, H, W]
        let mut pixels = Array3::zeros((3, image_size, image_size));
        let mut s = 9999u64;
        for c in 0..3 {
            for i in 0..image_size {
                for j in 0..image_size {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    pixels[[c, i, j]] = ((s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
        }

        let out = weights.encode(&pixels, &cfg);

        assert_eq!(out.shape(), &[n_patches, cfg.proj_dim],
            "vision tower output shape: expected [{n_patches}, {}], got [{}, {}]",
            cfg.proj_dim, out.shape()[0], out.shape()[1]);
        for &v in out.iter() {
            assert!(v.is_finite(), "vision tower output contains non-finite value: {v}");
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "vision tower output is all zeros (max_abs={max_abs})");
    }

    #[test]
    fn vision_tower_rgb_input_different_seed() {
        let cfg = small_vision_cfg();
        let weights = make_weights(&cfg);

        let image_size = cfg.image_size;
        let n_patches = (image_size / cfg.patch_size).pow(2);

        // RGB image with different pixel values
        let mut pixels = Array3::zeros((3, image_size, image_size));
        let mut s = 7777u64;
        for c in 0..3 {
            for i in 0..image_size {
                for j in 0..image_size {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    pixels[[c, i, j]] = ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
                }
            }
        }

        let out = weights.encode(&pixels, &cfg);

        assert_eq!(out.shape(), &[n_patches, cfg.proj_dim]);
        for &v in out.iter() {
            assert!(v.is_finite());
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "rgb vision output (alt seed) is all zeros");
    }

    #[test]
    fn vision_random_constructor_works() {
        let cfg = small_vision_cfg();
        let weights = VisionTowerWeights::random(&cfg);

        // Verify structure
        assert_eq!(weights.layers.len(), cfg.n_vision_layers);
        assert_eq!(weights.cls_token.len(), cfg.vision_dim);
        assert_eq!(weights.projector.shape(), &[cfg.proj_dim, cfg.vision_dim]);

        // Can run forward pass
        let image_size = cfg.image_size;
        let n_patches = (image_size / cfg.patch_size).pow(2);
        let mut pixels = Array3::zeros((3, image_size, image_size));
        for c in 0..3 {
            for i in 0..image_size {
                for j in 0..image_size {
                    pixels[[c, i, j]] = (i as f32 / image_size as f32) - 0.5;
                }
            }
        }
        let out = weights.encode(&pixels, &cfg);
        assert_eq!(out.shape(), &[n_patches, cfg.proj_dim]);
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn vision_load_image_valid_pixels() {
        // Create a tiny synthetic image in memory and write to temp file
        let mut img = image::RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Rgb([x as u8 * 64, y as u8 * 64, 128]));
            }
        }
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("kai_test_vision.png");
        img.save(&path).expect("save temp image");

        // Load and check shape/normalization
        let pixels = load_image(&path.to_str().unwrap(), 4).expect("load_image");
        assert_eq!(pixels.shape(), &[3, 4, 4]);
        for &v in pixels.iter() {
            assert!(v.is_finite(), "pixel value not finite: {v}");
            assert!(v >= -1.1 && v <= 1.1, "pixel out of range: {v}");
        }
        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
