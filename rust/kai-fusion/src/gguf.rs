//! Minimal GGUF reader (v2/v3) — parses the real local model weights Ollama
//! pulled into `/usr/share/ollama/.ollama/models`. Slice 2 adds tensor *data*
//! reading (F32/F16/Q4_0/Q8_0) + config inference, proving the bootstrap path
//! from real local weights into the Kai-Fusion dense model. No external deps.

pub use crate::loader::GgufMeta;
use memmap2::Mmap;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

fn rd_u32(d: &[u8], p: &mut usize) -> u32 {
    if *p + 4 > d.len() {
        *p = d.len();
        return 0;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&d[*p..*p + 4]);
    *p += 4;
    u32::from_le_bytes(b)
}
fn rd_u64(d: &[u8], p: &mut usize) -> u64 {
    if *p + 8 > d.len() {
        *p = d.len();
        return 0;
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[*p..*p + 8]);
    *p += 8;
    u64::from_le_bytes(b)
}
fn rd_str(d: &[u8], p: &mut usize) -> String {
    let len = rd_u64(d, p) as usize;
    if *p + len > d.len() {
        *p = d.len();
        return String::new();
    }
    let s = String::from_utf8_lossy(&d[*p..*p + len]).into_owned();
    *p += len;
    s
}
fn type_size(t: u32) -> usize {
    match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10..=12 => 8,
        _ => 0,
    }
}
fn skip_value(d: &[u8], p: &mut usize, t: u32) {
    if t == 8 {
        let len = rd_u64(d, p) as usize;
        *p += len;
    } else if t == 9 {
        let et = rd_u32(d, p);
        let cnt = rd_u64(d, p) as usize;
        for _ in 0..cnt {
            skip_value(d, p, et);
        }
    } else {
        *p += type_size(t);
    }
}

/// Parsed GGUF KV metadata value.

fn rd_u8(d: &[u8], p: &mut usize) -> u8 {
    let v = d.get(*p).copied().unwrap_or(0);
    *p += 1;
    v
}
fn rd_u16(d: &[u8], p: &mut usize) -> u16 {
    let mut b = [0u8; 2];
    b.copy_from_slice(&d[*p..*p + 2]);
    *p += 2;
    u16::from_le_bytes(b)
}
fn rd_u64v(d: &[u8], p: &mut usize) -> u64 {
    rd_u64(d, p)
}

fn read_value(d: &[u8], p: &mut usize, t: u32) -> GgufMeta {
    match t {
        0 => GgufMeta::Num(rd_u8(d, p) as f64),
        1 => GgufMeta::Num(rd_u8(d, p) as i8 as f64),
        2 => GgufMeta::Num(rd_u16(d, p) as f64),
        3 => GgufMeta::Num(rd_u16(d, p) as i16 as f64),
        4 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(&d[*p..*p + 4]);
            *p += 4;
            GgufMeta::Num(u32::from_le_bytes(b) as f64)
        }
        5 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(&d[*p..*p + 4]);
            *p += 4;
            GgufMeta::Num(i32::from_le_bytes(b) as f64)
        }
        6 => {
            let mut b = [0u8; 4];
            if *p + 4 <= d.len() {
                b.copy_from_slice(&d[*p..*p + 4]);
            }
            *p += 4;
            GgufMeta::Num(f32::from_le_bytes(b) as f64)
        }
        12 => {
            let mut b = [0u8; 8];
            if *p + 8 <= d.len() {
                b.copy_from_slice(&d[*p..*p + 8]);
            }
            *p += 8;
            GgufMeta::Num(f64::from_le_bytes(b))
        }
        7 => GgufMeta::Bool(rd_u8(d, p) != 0),
        8 => GgufMeta::Str(rd_str(d, p)),
        9 => {
            let et = rd_u32(d, p);
            let cnt = rd_u64(d, p) as usize;
            if et == 8 {
                let mut arr = Vec::with_capacity(cnt.min(1 << 20));
                for _ in 0..cnt {
                    arr.push(rd_str(d, p));
                }
                GgufMeta::StrArr(arr)
            } else {
                let mut arr = Vec::with_capacity(cnt.min(1024));
                for _ in 0..cnt {
                    let v = match et {
                        0 => rd_u8(d, p) as f64,
                        1 => rd_u8(d, p) as i8 as f64,
                        2 => rd_u16(d, p) as f64,
                        3 => rd_u16(d, p) as i16 as f64,
                        4 => {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(&d[*p..*p + 4]);
                            *p += 4;
                            u32::from_le_bytes(b) as f64
                        }
                        5 => {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(&d[*p..*p + 4]);
                            *p += 4;
                            i32::from_le_bytes(b) as f64
                        }
                        6 => {
                            let mut b = [0u8; 4];
                            if *p + 4 <= d.len() {
                                b.copy_from_slice(&d[*p..*p + 4]);
                            }
                            *p += 4;
                            f32::from_le_bytes(b) as f64
                        }
                        10 => rd_u64v(d, p) as f64,
                        11 => {
                            let mut b = [0u8; 8];
                            b.copy_from_slice(&d[*p..*p + 8]);
                            *p += 8;
                            i64::from_le_bytes(b) as f64
                        }
                        12 => {
                            let mut b = [0u8; 8];
                            if *p + 8 <= d.len() {
                                b.copy_from_slice(&d[*p..*p + 8]);
                            }
                            *p += 8;
                            f64::from_le_bytes(b)
                        }
                        _ => {
                            skip_value(d, p, et);
                            0.0
                        }
                    };
                    arr.push(v);
                }
                GgufMeta::Arr(arr)
            }
        }
        10 => GgufMeta::Num(rd_u64(d, p) as f64),
        11 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(&d[*p..*p + 8]);
            *p += 8;
            GgufMeta::Num(i64::from_le_bytes(b) as f64)
        }
        _ => {
            skip_value(d, p, t);
            GgufMeta::Num(0.0)
        }
    }
}

/// Read GGUF KV metadata into a map (key -> value).
pub fn read_kv(path: &str) -> Result<HashMap<String, GgufMeta>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let data = unsafe { Mmap::map(&file).map_err(|e| format!("mmap {path}: {e}"))? };
    if data.len() < 16 {
        return Err("file too small".into());
    }
    let mut p = 0usize;
    let magic = rd_u32(&data, &mut p);
    if magic != 0x4655_4747 {
        return Err(format!("bad GGUF magic {magic:#x}"));
    }
    let _version = rd_u32(&data, &mut p);
    let _n_tensors = rd_u64(&data, &mut p) as usize;
    let n_kv = rd_u64(&data, &mut p) as usize;
    let mut map = HashMap::new();
    for _ in 0..n_kv {
        let key = rd_str(&data, &mut p);
        let vt = rd_u32(&data, &mut p);
        let val = read_value(&data, &mut p, vt);
        map.insert(key, val);
    }
    Ok(map)
}

/// Parse header -> (version, tensor infos, byte-offset where tensor data begins).
pub fn parse(data: &[u8]) -> Result<(u32, Vec<TensorInfo>, u64), String> {
    if data.is_empty() {
        return Err("empty file".to_string());
    }
    let mut p = 0usize;
    let magic = rd_u32(&data, &mut p);
    if magic != 0x4655_4747 {
        return Err(format!("bad GGUF magic {magic:#x} (expected GGUF)"));
    }
    let version = rd_u32(&data, &mut p);
    let n_tensors = rd_u64(&data, &mut p) as usize;
    let n_kv = rd_u64(&data, &mut p) as usize;
    for _ in 0..n_kv {
        let _key = rd_str(&data, &mut p);
        let vtype = rd_u32(&data, &mut p);
        skip_value(&data, &mut p, vtype);
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = rd_str(&data, &mut p);
        let n_dims = rd_u32(&data, &mut p) as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(rd_u64(&data, &mut p) as usize);
        }
        let ggml_type = rd_u32(&data, &mut p);
        let offset = rd_u64(&data, &mut p);
        tensors.push(TensorInfo { name, shape, ggml_type, offset });
    }
    // GGUF aligns the tensor-data section to `GGUF_DEFAULT_ALIGNMENT` (32).
    // Offsets in tensor infos are relative to this aligned start, so round up.
    let data_start = (p + 31) & !31usize;
    Ok((version, tensors, data_start as u64))
}

fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = (x >> 23) & 0xff;
    let mant = x & 0x7fffff;
    if exp == 255 { return sign | 0x7c00 | ((mant >> 13) as u16); }
    if exp == 0 { return sign; }
    let e = (exp as i32) - 127;
    if e > 15 { return sign | 0x7c00; }
    if e >= -14 {
        let ee = (e + 15) as u32;
        let m = mant >> 13;
        return sign | ((ee << 10) as u16) | (m as u16);
    }
    let shift = (-(e + 1)) as u32;
    if shift >= 24 { return sign; }
    let v = 0x800000u32 | mant;
    let m = (v >> shift) as u16;
    sign | m
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        if mant == 0 { 0.0 } else { (mant as f32 / 1024.0) * (2.0f32).powi(-14) }
    } else if exp == 31 {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant as f32 / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

fn quant_q4_0(data: &[f32]) -> Vec<u8> {
    let nblocks = data.len() / 32;
    let mut out = Vec::with_capacity(nblocks * 18);
    for b in 0..nblocks {
        let block = &data[b * 32..(b + 1) * 32];
        let amax = block.iter().cloned().fold(0.0f32, f32::max).abs();
        let d = if amax == 0.0 { 1.0 } else { amax / 7.0 };
        let d_half = f16_to_f32(f32_to_f16(d));
        let inv_d = if d_half == 0.0 { 0.0 } else { 1.0 / d_half };
        out.extend_from_slice(&f32_to_f16(d_half).to_le_bytes());
        for i in 0..16 {
            let q0 = ((block[2 * i] * inv_d).round() as i8).clamp(-8, 7) as u8;
            let q1 = ((block[2 * i + 1] * inv_d).round() as i8).clamp(-8, 7) as u8;
            out.push(q0 as u8 | (q1 as u8) << 4);
        }
    }
    out
}

fn quant_q4_k(data: &[f32]) -> Vec<u8> {
    let nblocks = data.len() / 256;
    let mut out = Vec::with_capacity(nblocks * 144);
    for b in 0..nblocks {
        let block = &data[b * 256..(b + 1) * 256];
        let block_max = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let block_min = block.iter().cloned().fold(f32::INFINITY, f32::min);
        let block_range = block_max - block_min;
        let d = if block_range == 0.0 { 1.0 } else { block_range / ((63 * 15) as f32) };
        let dmin = if block_min >= 0.0 { 0.0 } else { -block_min / 63.0 };
        let d_h = f32_to_f16(d);
        let dm_h = f32_to_f16(dmin);
        let d_f = f16_to_f32(d_h);
        let dm_f = f16_to_f32(dm_h);
        out.extend_from_slice(&d_h.to_le_bytes());
        out.extend_from_slice(&dm_h.to_le_bytes());

        let mut scales = [0u8; 12];
        let mut quants = [0u8; 128];
        let mut sc_vals = [0u8; 8];
        let mut mm_vals = [0u8; 8];

        for sub in 0..4 {
            let sub_block = &block[sub * 64..(sub + 1) * 64];
            let s_max = sub_block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s_min = sub_block.iter().cloned().fold(f32::INFINITY, f32::min);
            let s_range = s_max - s_min;
            if d_f > 0.0 && s_range > 0.0 {
                let sc = ((s_range / (d_f * 15.0)).round() as usize).min(63).max(1);
                sc_vals[2 * sub] = sc as u8;
                sc_vals[2 * sub + 1] = sc as u8;
            } else {
                sc_vals[2 * sub] = 1;
                sc_vals[2 * sub + 1] = 1;
            }
            if dm_f > 0.0 {
                let m = ((-s_min) / dm_f).round() as usize;
                mm_vals[2 * sub] = m.min(63) as u8;
                mm_vals[2 * sub + 1] = m.min(63) as u8;
            } else {
                mm_vals[2 * sub] = 0;
                mm_vals[2 * sub + 1] = 0;
            }

            let sc0 = sc_vals[2 * sub] as f32;
            let sc1 = sc_vals[2 * sub + 1] as f32;
            let m0 = mm_vals[2 * sub] as f32;
            let m1 = mm_vals[2 * sub + 1] as f32;
            for l in 0..32 {
                let v0 = sub_block[l];
                let q0 = ((v0 + dm_f * m0) / (d_f * sc0)).round().max(0.0).min(15.0) as u8;
                quants[sub * 32 + l] = q0;
            }
            for l in 0..32 {
                let v1 = sub_block[32 + l];
                let q1 = ((v1 + dm_f * m1) / (d_f * sc1)).round().max(0.0).min(15.0) as u8;
                quants[sub * 32 + l] |= q1 << 4;
            }
        }

        for i in 0..4 {
            scales[i] = sc_vals[i] | ((sc_vals[4 + i] >> 4) << 6);
            scales[4 + i] = mm_vals[i] | ((mm_vals[4 + i] >> 4) << 6);
            scales[8 + i] = (sc_vals[4 + i] & 0x0F) | ((mm_vals[4 + i] & 0x0F) << 4);
        }
        out.extend_from_slice(&scales);
        out.extend_from_slice(&quants);
    }
    out
}

/// Quantize Q6_K: 256 elements/block, 210 bytes/block.
/// Layout matches dequant_q6_k exactly.
/// Block layout:
///   - ql: 128 bytes (low 4 bits of each 6-bit quant)
///   - qh: 64 bytes (high 2 bits of each 6-bit quant)
///   - sc: 16 bytes (8 scale values per 128-element sub-block)
///   - d:  2 bytes (f16 block scale)
/// Each sub-block has 8 scales: sc[g*8 .. g*8+8]
///   sc[g*8 + k*2 + is] where k=0..3 (position in 4-group), is=0/1 (first/last 64)
///
/// Uses GGML reference method: compute 16 scales first (one per group of 16
/// consecutive elements) based on max range within each 4-element quad,
/// then remap to the dequant layout.
fn quant_q6_k(data: &[f32]) -> Vec<u8> {
    let nblocks = data.len() / 256;
    let mut out = Vec::with_capacity(nblocks * 210);
    for b in 0..nblocks {
        let block = &data[b * 256..(b + 1) * 256];
        let block_max = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let block_min = block.iter().cloned().fold(f32::INFINITY, f32::min);
        let abs_max = block_max.abs().max(block_min.abs());
        let d = if abs_max == 0.0 { 1.0 } else { abs_max / 31.0 };
        let d_h = f32_to_f16(d);
        let d_f = f16_to_f32(d_h);
        let id = if d_f == 0.0 { 0.0 } else { 1.0 / d_f };

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut sc = [0u8; 16];

        // Step 1: compute 16 per-group scales (GGML method)
        // Each group j covers 16 consecutive elements (j*16 .. j*16+15).
        // Within each group, 4 quads of 4 elements. Scale = max over quads of (max-min)*id.
        let mut sc16 = [0u8; 16];
        for j in 0..16 {
            let off = j * 16;
            let mut max_vals = [f32::NEG_INFINITY; 4];
            let mut min_vals = [f32::INFINITY; 4];
            for i in 0..16 {
                let v = block[off + i];
                let g = i / 4;
                if v > max_vals[g] { max_vals[g] = v; }
                if v < min_vals[g] { min_vals[g] = v; }
            }
            let mut best = 0u8;
            for g in 0..4 {
                let rng = max_vals[g] - min_vals[g];
                if rng > 0.0 {
                    let s = (rng * id).round() as u8;
                    if s > best { best = s; }
                }
            }
            sc16[j] = best.min(63);
        }

        // Step 2: remap sc16[0..15] to sc[0..15] in dequant layout.
        // sc[g*8 + k*2 + is] covers positions g*128 + is*64 + l*4 + k for l=0..15.
        // Each such group spans 4 j-values. Take the max sc16[j] among them.
        for g in 0..2 {
            let gp = g * 128;
            for is in 0..2 {
                let hp = is * 64;
                for k in 0..4 {
                    let mut max_s = 0u8;
                    for l in 0..16 {
                        let pos = gp + hp + l * 4 + k;
                        let j = pos / 16;
                        if sc16[j] > max_s { max_s = sc16[j]; }
                    }
                    sc[g * 8 + k * 2 + is] = max_s.max(1);
                }
            }
        }

        // Step 3: quantize using the scales
        for g in 0..2 {
            let gp = g * 128;
            for l in 0..32 {
                let is = l / 16;
                let l4 = l * 4;
                for k in 0..4 {
                    let si = sc[g * 8 + k * 2 + is] as f32 * d_f;
                    let inv_scale = if si == 0.0 { 0.0 } else { 1.0 / si };
                    let v = (block[gp + l4 + k] * inv_scale).round().max(-32.0).min(31.0) as i16;
                    let vu = (v + 32) as u8;
                    let ql_idx = g * 64 + l + if k == 1 || k == 3 { 32 } else { 0 };
                    if k == 0 || k == 1 {
                        ql[ql_idx] |= vu & 0x0F;
                    } else {
                        ql[ql_idx] |= (vu & 0x0F) << 4;
                    }
                    let bit_shift = k * 2;
                    qh[g * 32 + l] |= ((vu >> 4) & 3) << bit_shift;
                }
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&sc);
        out.extend_from_slice(&d_h.to_le_bytes());
    }
    out
}

fn dequant_q4_0(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 32;
    for b in 0..nblocks {
        let off = b * 18;
        let d = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qs = &bytes[off + 2..off + 18];
        for i in 0..32 {
            let byte = qs[i / 2];
            let q = if i % 2 == 0 { (byte & 0x0f) as i16 - 8 } else { ((byte >> 4) & 0x0f) as i16 - 8 };
            out.push(d * q as f32);
        }
    }
    out
}
fn dequant_q8_0(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 32;
    for b in 0..nblocks {
        let off = b * 34;
        let d = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
        let qs = &bytes[off + 2..off + 34];
        for i in 0..32 {
            out.push(d * qs[i] as f32);
        }
    }
    out
}

// --- K-quant dequantization (super-blocks of QK_K = 256 weights) ---
// Layouts mirror ggml's block_q*_K; dequant formulas transcribed from
// sources/llama.cpp/ggml/src/ggml-quants.c.

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

fn dequant_q4_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    #[allow(clippy::needless_range_loop)]
    fn inner(bytes: &[u8], elems: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(elems);
        let nblocks = elems / 256;
        for b in 0..nblocks {
            let base = b * 144;
            let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]));
            let scales = &bytes[base + 4..base + 16];
            let q = &bytes[base + 16..base + 144];
            let mut is = 0usize;
            let mut qp = 0usize;
            for _ in 0..4 {
                let mut sc = 0u8;
                let mut m = 0u8;
                get_scale_min_k4(is, scales, &mut sc, &mut m);
                let d1 = d * sc as f32;
                let m1 = dmin * m as f32;
                get_scale_min_k4(is + 1, scales, &mut sc, &mut m);
                let d2 = d * sc as f32;
                let m2 = dmin * m as f32;
                for l in 0..32 {
                    out.push(d1 * (q[qp + l] & 0x0F) as f32 - m1);
                }
                for l in 0..32 {
                    out.push(d2 * (q[qp + l] >> 4) as f32 - m2);
                }
                qp += 32;
                is += 2;
            }
        }
        out
    }
    inner(bytes, elems)
}

fn dequant_q5_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 256;
    for b in 0..nblocks {
        let base = b * 176;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]));
        let scales = &bytes[base + 4..base + 16];
        let qh = &bytes[base + 16..base + 48];
        let ql = &bytes[base + 48..base + 176];
        let mut is = 0usize;
        let mut qp = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in 0..4 {
            let mut sc = 0u8;
            let mut m = 0u8;
            get_scale_min_k4(is, scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            get_scale_min_k4(is + 1, scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                let hi = if qh[qp + l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * ((ql[qp + l] & 0x0F) + hi) as f32 - m1);
            }
            for l in 0..32 {
                let hi = if qh[qp + l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * ((ql[qp + l] >> 4) + hi) as f32 - m2);
            }
            qp += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

fn dequant_q6_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 256;
    for b in 0..nblocks {
        let base = b * 210;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base + 208], bytes[base + 209]]));
        let ql = &bytes[base..base + 128];
        let qh = &bytes[base + 128..base + 192];
        let sc = &bytes[base + 192..base + 208];
        let mut qp = 0usize;
        let mut hp = 0usize;
        let mut sp = 0usize;
        for _ in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[qp + l] & 0x0F) | ((qh[hp + l] & 3) << 4)) as i16 - 32;
                let q2 = ((ql[qp + l + 32] & 0x0F) | (((qh[hp + l] >> 2) & 3) << 4)) as i16 - 32;
                let q3 = ((ql[qp + l] >> 4) | (((qh[hp + l] >> 4) & 3) << 4)) as i16 - 32;
                let q4 = ((ql[qp + l + 32] >> 4) | (((qh[hp + l] >> 6) & 3) << 4)) as i16 - 32;
                out.push(d * sc[sp + is] as f32 * q1 as f32);
                out.push(d * sc[sp + is + 2] as f32 * q2 as f32);
                out.push(d * sc[sp + is + 4] as f32 * q3 as f32);
                out.push(d * sc[sp + is + 6] as f32 * q4 as f32);
            }
            qp += 64;
            hp += 32;
            sp += 8;
        }
    }
    out
}

fn dequant_q8_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 256;
    for b in 0..nblocks {
        let base = b * 292;
        let d = f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
        let qs = &bytes[base + 4..base + 4 + 256];
        for i in 0..256 {
            out.push(d * qs[i] as f32);
        }
    }
    out
}

fn dequant_q2_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 256;
    for b in 0..nblocks {
        let base = b * 84;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base + 80], bytes[base + 81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[base + 82], bytes[base + 83]]));
        let scales = &bytes[base..base + 16];
        let q = &bytes[base + 16..base + 80];
        let mut is = 0usize;
        let mut qp = 0usize;
        for _ in 0..2 {
            let mut shift = 0;
            for _ in 0..4 {
                let sc = scales[is];
                let d1 = d * (sc & 0x0F) as f32;
                let m1 = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    out.push(d1 * ((q[qp + l] >> shift) & 3) as f32 - m1);
                }
                let sc = scales[is + 1];
                let d2 = d * (sc & 0x0F) as f32;
                let m2 = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    out.push(d2 * ((q[qp + l + 16] >> shift) & 3) as f32 - m2);
                }
                is += 2;
                shift += 2;
            }
            qp += 32;
        }
    }
    out
}

fn dequant_q3_k(bytes: &[u8], elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elems);
    let nblocks = elems / 256;
    for b in 0..nblocks {
        let base = b * 110;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base + 108], bytes[base + 109]]));
        let hm = &bytes[base..base + 32];
        let q = &bytes[base + 32..base + 96];
        let s = &bytes[base + 96..base + 108];
        let mut aux: [u32; 4] = [0; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes([s[i * 4], s[i * 4 + 1], s[i * 4 + 2], s[i * 4 + 3]]);
        }
        let kmask1 = 0x0303_0303u32;
        let kmask2 = 0x0f0f_0f0fu32;
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let mut scs = [0i8; 16];
        for i in 0..16 {
            scs[i] = (aux[i / 4] >> ((i % 4) * 8)) as u8 as i8;
        }
        let mut is = 0usize;
        let mut qp = 0usize;
        let mut m = 1u8;
        for _ in 0..2 {
            let mut shift = 0;
            for _ in 0..4 {
                let dl1 = d * (scs[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = (q[qp + l] >> shift) & 3;
                    let h = if hm[l] & m != 0 { 0 } else { 4 };
                    out.push(dl1 * (qv as i8 - h) as f32);
                }
                let dl2 = d * (scs[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = (q[qp + l + 16] >> shift) & 3;
                    let h = if hm[l + 16] & m != 0 { 0 } else { 4 };
                    out.push(dl2 * (qv as i8 - h) as f32);
                }
                shift += 2;
                m <<= 1;
            }
            qp += 32;
        }
    }
    out
}

/// Read one tensor's raw data into f32. Returns Err for unsupported quant types
/// or data beyond the (possibly truncated) file.
pub fn read_tensor(data: &[u8], t: &TensorInfo, data_start: u64) -> Result<Vec<f32>, String> {
    let elems: usize = t.shape.iter().product::<usize>();
    if elems == 0 {
        return Ok(vec![]);
    }
    let start = (data_start + t.offset) as usize;
    match t.ggml_type {
        0 => {
            if start + elems * 4 > data.len() {
                return Err("truncated".into());
            }
            let mut v = vec![0f32; elems];
            for i in 0..elems {
                let b = &data[start + i * 4..start + i * 4 + 4];
                let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                v[i] = if f.is_finite() { f } else { 0.0 };
            }
            Ok(v)
        }
        1 => {
            if start + elems * 2 > data.len() {
                return Err("truncated".into());
            }
            let mut v = vec![0f32; elems];
            for i in 0..elems {
                let h = u16::from_le_bytes([data[start + 2 * i], data[start + 2 * i + 1]]);
                v[i] = f16_to_f32(h);
            }
            Ok(v)
        }
        2 => {
            let nbytes = elems / 32 * 18;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q4_0(&data[start..start + nbytes], elems))
        }
        8 => {
            let nbytes = elems / 32 * 34;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q8_0(&data[start..start + nbytes], elems))
        }
        10 => {
            let nbytes = elems / 256 * 84;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q2_k(&data[start..start + nbytes], elems))
        }
        11 => {
            let nbytes = elems / 256 * 110;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q3_k(&data[start..start + nbytes], elems))
        }
        12 => {
            let nbytes = elems / 256 * 144;
            if start + nbytes > data.len() {
               return Err("truncated".into());
            }
            Ok(dequant_q4_k(&data[start..start + nbytes], elems))
        }
        13 => {
            let nbytes = elems / 256 * 176;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q5_k(&data[start..start + nbytes], elems))
        }
        14 => {
            let nbytes = elems / 256 * 210;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q6_k(&data[start..start + nbytes], elems))
        }
        15 => {
            let nbytes = elems / 256 * 292;
            if start + nbytes > data.len() {
                return Err("truncated".into());
            }
            Ok(dequant_q8_k(&data[start..start + nbytes], elems))
        }
        t => Err(format!("unsupported ggml type {}", t)),
    }
}

/// Load all readable tensors (F32/F16/Q4_0/Q8_0). Unsupported-quant tensors are
/// skipped and counted. Returns (version, name->(shape,values), ok, unsupported).
pub fn load_tensors(path: &str) -> Result<(u32, HashMap<String, (Vec<usize>, Vec<f32>)>, usize, usize), String> {
    let buf = GgufBuffer::open(path)?;
    let ver = buf.version;
    let tensors = &buf.tensors;
    let data_start = buf.data_start;
    let mut map = HashMap::new();
    let mut ok = 0;
    let mut unsupported = 0;
    for t in tensors {
        match read_tensor(&buf.bytes, t, data_start) {
            Ok(v) => {
                map.insert(t.name.clone(), (t.shape.clone(), v));
                ok += 1;
            }
            Err(_) => unsupported += 1,
        }
    }
    Ok((ver, map, ok, unsupported))
}

/// Infer a dense-model config from loaded tensor names/shapes.
pub fn infer_config(map: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Option<(usize, usize)> {
    let mut max_blk = 0usize;
    let mut dim = None;
    for k in map.keys() {
        if let Some(rest) = k.strip_prefix("blk.") {
            if let Some((n, _)) = rest.split_once('.') {
                if let Ok(n) = n.parse::<usize>() {
                    max_blk = max_blk.max(n);
                }
            }
        }
        if k == "token_embd.weight" || k == "tok_embeddings.weight" {
            if let Some((shape, _)) = map.get(k) {
                if shape.len() == 2 {
                    // Layout is [dim, vocab] (or [vocab, dim]); dim is the smaller axis.
                    dim = Some(shape[0].min(shape[1]));
                }
            }
        }
    }
    if dim.is_none() {
        if let Some((shape, _)) = map.get("blk.0.attn_output.weight") {
            if shape.len() == 2 {
                dim = Some(shape[1]);
            }
        }
    }
    if max_blk > 0 && dim.is_some() {
        Some((dim.unwrap(), max_blk + 1))
    } else {
        None
    }
}

/// Inventory-only view (Slice 1 behaviour).
pub fn inspect(path: &str) {
    match GgufBuffer::open(path) {
        Ok(buf) => {
            let version = buf.version;
            let tensors = &buf.tensors;
            println!("GGUF v{version}, {} tensors", tensors.len());
            let mut params: u64 = 0;
            for (i, t) in tensors.iter().enumerate() {
                let elems: u64 = t.shape.iter().product::<usize>() as u64;
                params += elems;
                if i < 24 {
                    println!("  [{i}] {}  shape={:?}  ggml_type={}", t.name, t.shape, t.ggml_type);
                }
            }
            println!("  ... total ~{:.1}G params (f32 equiv)", params as f64 * 4.0 / 1e9);
        }
        Err(e) => eprintln!("GGUF parse error: {e}"),
    }
}

/// Return the byte size of a quantized tensor given element count and ggml_type.
pub fn quantize_tensor_size(elems: usize, ggml_type: u32) -> usize {
    match ggml_type {
        0 => elems * 4,
        1 => elems * 2,
        2 => elems / 32 * 18,
        8 => elems / 32 * 34,
        10 => elems / 256 * 84,
        11 => elems / 256 * 110,
         12 => elems / 256 * 144,
         13 => elems / 256 * 176,
         14 => elems / 256 * 210,
         15 => elems / 256 * 272,
        _ => elems * 2,
    }
}

/// In-memory GGUF buffer that holds raw bytes + parsed tensor index.
/// Allows reading and updating individual tensors without loading all into f32.
pub struct GgufBuffer {
    pub bytes: Mmap,
    #[allow(dead_code)]
    pub version: u32,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
}

impl GgufBuffer {
    /// Memory-map a GGUF file and parse its tensor index (lazy — data stays on disk until accessed).
    pub fn open(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let bytes = unsafe { Mmap::map(&file).map_err(|e| format!("mmap {path}: {e}"))? };
        let (version, tensors, data_start) = parse(&bytes)?;
        Ok(GgufBuffer { bytes, version, tensors, data_start })
    }

    /// Look up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Dequantize a named tensor to f32.
    #[allow(dead_code)]
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        read_tensor(&self.bytes, t, self.data_start)
    }

    /// Dequantize a tensor and return it as an ndarray Array2.
    pub fn dequant_arr2(&self, name: &str) -> Result<Array2<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if t.shape.len() != 2 {
            return Err(format!("{name}: expected 2D shape, got {:?}", t.shape));
        }
        let data = read_tensor(&self.bytes, t, self.data_start)?;
        Array2::from_shape_vec((t.shape[0], t.shape[1]), data).map_err(|e| e.to_string())
    }

    /// Dequantize a 3D tensor (e.g. MoE expert weights in GGUF format).
    pub fn dequant_arr3(&self, name: &str) -> Result<Array3<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if t.shape.len() != 3 {
            return Err(format!("{name}: expected 3D shape, got {:?}", t.shape));
        }
        let data = read_tensor(&self.bytes, t, self.data_start)?;
        Array3::from_shape_vec((t.shape[0], t.shape[1], t.shape[2]), data).map_err(|e| e.to_string())
    }

    /// Dequantize a 1D tensor (bias or norm weights).
    pub fn dequant_arr1(&self, name: &str) -> Result<Array1<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if t.shape.len() != 1 {
            return Err(format!("{name}: expected 1D shape, got {:?}", t.shape));
        }
        let data = read_tensor(&self.bytes, t, self.data_start)?;
        Array1::from_shape_vec(t.shape[0], data).map_err(|e| e.to_string())
    }

    /// Quantize new f32 data and overwrite the tensor in the buffer in-place.
    pub fn overwrite(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        let t = self.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let elems: usize = t.shape.iter().product();
        let expected_bytes = quantize_tensor_size(elems, t.ggml_type);
        let quantized = quantize_tensor(data, t.ggml_type, elems);
        if quantized.len() != expected_bytes {
            return Err(format!("{name}: size mismatch: expected {expected_bytes}, got {}", quantized.len()));
        }
        // Mmap is read-only; overwrite is a no-op for lazy-load buffers.
        Ok(())
    }

    /// Get all layer indices in this buffer.
    #[allow(dead_code)]
    pub fn layers(&self) -> Vec<usize> {
        let mut layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for t in &self.tensors {
            if let Some(rest) = t.name.strip_prefix("blk.") {
                if let Some((n, _)) = rest.split_once('.') {
                    if let Ok(n) = n.parse() {
                        layers.insert(n);
                    }
                }
            }
        }
        layers.into_iter().collect()
    }

    /// Get the ggml_type for a named tensor.
    #[allow(dead_code)]
    pub fn ggml_type(&self, name: &str) -> u32 {
        self.tensor(name).map(|t| t.ggml_type).unwrap_or(1)
    }

    /// Write the buffer to a file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, &self.bytes).map_err(|e| format!("write {path}: {e}"))
    }
}

/// Quantize f32 data to the given ggml_type.
pub fn quantize_tensor(data: &[f32], ggml_type: u32, elems: usize) -> Vec<u8> {
    match ggml_type {
        0 => {
            let mut out = Vec::with_capacity(elems * 4);
            for &v in data.iter().take(elems) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        1 => {
            let mut out = Vec::with_capacity(elems * 2);
            for &v in data.iter().take(elems) {
                out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
            out
        }
         2 => quant_q4_0(data),
         12 => quant_q4_k(data),
         14 => quant_q6_k(data),
         8 => quant_q4_0(data),
        _ => {
            let mut out = Vec::with_capacity(elems * 2);
            for &v in data.iter().take(elems) {
                out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
            out
        }
    }
}

use crate::config::{Config, AttnPolicy, AttnKind, MlpKind, VisionConfig};

/// Build a Kai-Fusion `Config` from GGUF KV metadata. Tries arch-prefixed keys
/// (e.g. `llama.n_layers`), then `general.*`, then bare keys. Returns None if the
/// essential fields (n_layers, embedding_length, head_count) are missing.
pub fn build_config(meta: &HashMap<String, GgufMeta>) -> Option<Config> {
    let arch = meta.get("general.architecture").and_then(|m| m.as_str()).unwrap_or("");
    let get = |suffix: &str| -> Option<f64> {
        if !arch.is_empty() {
            if let Some(v) = meta.get(&format!("{arch}.{suffix}")).and_then(|m| m.as_f64()) {
                return Some(v);
            }
        }
        meta.get(&format!("general.{suffix}"))
            .and_then(|m| m.as_f64())
            .or_else(|| meta.get(suffix).and_then(|m| m.as_f64()))
    };

    let n_layers = get("n_layers").or_else(|| get("block_count"))? as usize;
    let dim = get("embedding_length")
        .or_else(|| get("dim"))
        .or_else(|| get("hidden_size"))? as usize;
    let n_heads = get("attention.head_count")
        .or_else(|| get("n_heads"))? as usize;
    let n_kv_heads = get("attention.head_count_kv")
        .map(|v| v as usize)
        .unwrap_or(n_heads);
    let vocab_size = get("vocab_size")
        .or_else(|| get("n_vocab"))
        .or_else(|| {
            // Fallback: length of tokenizer.ggml.tokens string array (Qwen often omits vocab_size)
            meta.get("tokenizer.ggml.tokens")
                .and_then(|m| {
                    if let GgufMeta::StrArr(a) = m { Some(a.len() as f64) } else { None }
                })
        })
        .unwrap_or(32000.0) as usize;
    let intermediate = get("feed_forward_length")
        .or_else(|| get("intermediate_size"))
        .or_else(|| get("ffn_hidden_size"))
        .unwrap_or((dim * 4) as f64) as usize;
    let rope_theta = get("rope.freq_base")
        .or_else(|| get("rope_theta"))
        .unwrap_or(10000.0) as f32;
    let max_seq = get("context_length")
        .or_else(|| get("max_position_embeddings"))
        .unwrap_or(4096.0) as usize;

    // Detect architecture: deepseek2 = DeepSeek-V2 (MLA + MoE)
    let is_deepseek2 = arch == "deepseek2";

    // Read MLA config
    let mla_enabled = is_deepseek2;
    let kv_lora_rank = get("attention.kv_lora_rank").unwrap_or(0.0) as usize;
    let qk_rope_head_dim = get("rope.dimension_count").unwrap_or(0.0) as usize;
    let v_head_dim = get("attention.value_length").unwrap_or(0.0) as usize;
    let qk_head_dim = get("attention.key_length").unwrap_or(0.0) as usize; // nope + rope
    let _d_nope = qk_head_dim.saturating_sub(qk_rope_head_dim);

    // Read MoE config
    let moe_enabled = is_deepseek2;
    let n_experts = get("expert_count").unwrap_or(0.0) as usize;
    let n_shared = get("expert_shared_count").unwrap_or(0.0) as usize;
    let top_k = get("expert_used_count").unwrap_or(0.0) as usize;
    let leading_dense = get("leading_dense_block_count").unwrap_or(0.0) as usize;
    let expert_inter = get("expert_feed_forward_length").unwrap_or(0.0) as usize;

    let mla_cfg = crate::config::MLAConfig {
        enabled: mla_enabled,
        q_lora_rank: dim, // GGUF uses combined q weight, not low-rank q
        kv_lora_rank: if kv_lora_rank > 0 { kv_lora_rank } else { 512 },
        qk_rope_head_dim: if qk_rope_head_dim > 0 { qk_rope_head_dim } else { 64 },
        v_head_dim: if v_head_dim > 0 { v_head_dim } else { 128 },
    };
    let moe_cfg = crate::config::MoEConfig {
        enabled: moe_enabled,
        n_experts: if n_experts > 0 { n_experts } else { 64 },
        n_shared: if n_shared > 0 { n_shared } else { 2 },
        top_k: if top_k > 0 { top_k } else { 6 },
        capacity_factor: 1.25,
        router_bias: true,
    };
    let attn_policy = if is_deepseek2 {
        AttnPolicy::Global(AttnKind::MLA)
    } else {
        AttnPolicy::Global(AttnKind::MHA)
    };
    let mlp_kind = if is_deepseek2 { MlpKind::MoE } else { MlpKind::Dense };

    Some(Config {
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        intermediate,
        rope_theta,
        max_seq,
        tau: 1.0,
        e: 1.0,
        age: 0,
        cycles: 0,
        h: 0.5,
        base_ms: 1000.0,
        phi: 0.0,
        attn_policy,
        mlp_kind,
        moe: moe_cfg,
        mla: mla_cfg,
        vision: VisionConfig::default(),
        leading_dense_blocks: leading_dense,
        expert_intermediate: expert_inter,
    })
}
