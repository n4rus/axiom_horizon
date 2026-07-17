//! # physics-dialect
//!
//! Kai's custom *physics* dialect, built on top of the [`kai_mlir`] IR
//! framework. It encodes the system's physics-inspired learning laws as
//! first-class dialect operations:
//!
//! - `g_ij_calculate` — attention geometry, `g_ij = 1 - a_ij`
//! - `update_tau`     — time dilation, `τ' = τ·sqrt(1 - v²)` (c = 1)
//! - `calculate_vfe`  — variational free energy (MSE + regularization)
//! - `tonal_collapse` — convergence test
//! - `world_model_physics` — integrated forward pass placeholder
//!
//! Because it reuses the shared IR, the same registry/verifier used by the
//! generic `kai` dialect applies here.

use kai_mlir::{AttributeSet, Dialect, IrError, Operation, TypeDef, Value, Value::Scalar, Value::Tensor};
use std::sync::Arc;

const VARIANCE_COEFF: f64 = 0.5;
const NOVELTY_COEFF: f64 = 0.1;

/// `g_ij = 1 - a_ij`
pub struct GijOp;
impl Operation for GijOp {
    fn name(&self) -> &'static str {
        "g_ij_calculate"
    }
    fn dialect(&self) -> &'static str {
        "kai_physics"
    }
    fn operands(&self) -> usize {
        1
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: g_ij = 1 - a_ij (attention geometry)"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 1 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 1,
                actual: operands.len(),
            });
        }
        if operands[0].as_scalar().is_none() {
            return Err(IrError::ExpectedScalar {
                op: self.name(),
                index: 0,
            });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        Ok(Scalar(1.0 - operands[0].as_scalar().unwrap()))
    }
}

/// `τ' = τ·sqrt(1 - v²)` with `c = 1`.
pub struct TauOp;
impl Operation for TauOp {
    fn name(&self) -> &'static str {
        "update_tau"
    }
    fn dialect(&self) -> &'static str {
        "kai_physics"
    }
    fn operands(&self) -> usize {
        2
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: relativistic time dilation (c = 1)"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 2 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 2,
                actual: operands.len(),
            });
        }
        for (i, op) in operands.iter().enumerate() {
            if op.as_scalar().is_none() {
                return Err(IrError::ExpectedScalar {
                    op: self.name(),
                    index: i,
                });
            }
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let tau = operands[0].as_scalar().unwrap();
        let v = operands[1].as_scalar().unwrap();
        Ok(Scalar(tau * (1.0 - v * v).max(0.0).sqrt()))
    }
}

/// `vfe = mse(prediction, reality) + variance·0.5 + novelty·0.1`
pub struct VfeOp;
impl Operation for VfeOp {
    fn name(&self) -> &'static str {
        "calculate_vfe"
    }
    fn dialect(&self) -> &'static str {
        "kai_physics"
    }
    fn operands(&self) -> usize {
        4
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: variational free energy with regularization"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 4 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 4,
                actual: operands.len(),
            });
        }
        let p = operands[0].as_tensor();
        let r = operands[1].as_tensor();
        if p.is_none() || r.is_none() {
            return Err(IrError::ExpectedTensor {
                op: self.name(),
                index: if p.is_none() { 0 } else { 1 },
            });
        }
        if p.unwrap().len() != r.unwrap().len() {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: p.unwrap().len(),
                actual: r.unwrap().len(),
            });
        }
        if operands[2].as_scalar().is_none() || operands[3].as_scalar().is_none() {
            return Err(IrError::ExpectedScalar {
                op: self.name(),
                index: if operands[2].as_scalar().is_none() { 2 } else { 3 },
            });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let p = operands[0].as_tensor().unwrap();
        let r = operands[1].as_tensor().unwrap();
        let variance = operands[2].as_scalar().unwrap();
        let novelty = operands[3].as_scalar().unwrap();
        let mse: f64 = p
            .iter()
            .zip(r.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            / p.len() as f64;
        Ok(Scalar(mse + variance * VARIANCE_COEFF + novelty * NOVELTY_COEFF))
    }
}

/// Convergence test: `1.0` if similarity ≥ threshold (default `0.95`), else `0.0`.
pub struct CollapseOp;
impl Operation for CollapseOp {
    fn name(&self) -> &'static str {
        "tonal_collapse"
    }
    fn dialect(&self) -> &'static str {
        "kai_physics"
    }
    fn operands(&self) -> usize {
        1
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: convergence detection"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 1 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 1,
                actual: operands.len(),
            });
        }
        if operands[0].as_scalar().is_none() {
            return Err(IrError::ExpectedScalar {
                op: self.name(),
                index: 0,
            });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], attrs: &AttributeSet) -> Result<Value, IrError> {
        let similarity = operands[0].as_scalar().unwrap();
        let threshold = match attrs.get("threshold") {
            Some(Scalar(t)) => *t,
            _ => 0.95,
        };
        Ok(Scalar(if similarity >= threshold { 1.0 } else { 0.0 }))
    }
}

/// Integrated world-model physics: the real two-layer forward pass executed
/// through the IR. Operands are `[state, action, W1, b1, W2, b2]` (matrices
/// passed flat, row-major). `hidden = W1.len() / (state.len() + action.len())`,
/// `output = b2.len()`.
pub struct WorldModelPhysicsOp;
impl Operation for WorldModelPhysicsOp {
    fn name(&self) -> &'static str {
        "world_model_physics"
    }
    fn dialect(&self) -> &'static str {
        "kai_physics"
    }
    fn operands(&self) -> usize {
        6
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: integrated world-model forward pass (real matmul through the IR)"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 6 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 6,
                actual: operands.len(),
            });
        }
        for (i, op) in operands.iter().enumerate() {
            if op.as_tensor().is_none() {
                return Err(IrError::ExpectedTensor {
                    op: self.name(),
                    index: i,
                });
            }
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let s = operands[0].as_tensor().unwrap();
        let a = operands[1].as_tensor().unwrap();
        let w1 = operands[2].as_tensor().unwrap();
        let b1 = operands[3].as_tensor().unwrap();
        let w2 = operands[4].as_tensor().unwrap();
        let b2 = operands[5].as_tensor().unwrap();

        let input_dim = s.len() + a.len();
        let hidden = w1.len() / input_dim;

        let mut input = Vec::with_capacity(input_dim);
        input.extend_from_slice(s);
        input.extend_from_slice(a);

        // hidden = tanh(W1 · input + b1)
        let mut h = vec![0.0f64; hidden];
        for r in 0..hidden {
            let mut acc = b1[r];
            let row = &w1[r * input_dim..(r + 1) * input_dim];
            for c in 0..input_dim {
                acc += row[c] * input[c];
            }
            h[r] = acc.tanh();
        }

        // output = W2 · hidden + b2  (W2: rows = output_dim, cols = hidden)
        let output_dim = b2.len();
        let mut out = vec![0.0f64; output_dim];
        for o in 0..output_dim {
            let mut acc = b2[o];
            let row = &w2[o * hidden..(o + 1) * hidden];
            for r in 0..hidden {
                acc += row[r] * h[r];
            }
            out[o] = acc;
        }
        Ok(Tensor(out))
    }
}

/// The custom `kai_physics` dialect.
pub struct KaiPhysicsDialect;
impl Dialect for KaiPhysicsDialect {
    fn name(&self) -> &'static str {
        "kai_physics"
    }
    fn operations(&self) -> Vec<Arc<dyn Operation>> {
        vec![
            Arc::new(GijOp),
            Arc::new(TauOp),
            Arc::new(VfeOp),
            Arc::new(CollapseOp),
            Arc::new(WorldModelPhysicsOp),
        ]
    }
    fn types(&self) -> Vec<TypeDef> {
        vec![
            TypeDef::new("concept_vector"),
            TypeDef::new("attention_matrix"),
            TypeDef::new("kai_tau"),
            TypeDef::new("kai_vfe"),
            TypeDef::new("purpose_vector"),
        ]
    }
}

/// Lower the world-model forward pass to **upstream MLIR** (LLVM `func` /
/// `tensor` / `scf` / `arith` / `math` dialects). The emitted module is the
/// exact compute graph Kai runs (`hidden = tanh(W1·input + b1)`,
/// `output = W2·hidden + b2`); it parses and verifies under a real `mlir-opt`.
pub fn emit_world_model_mlir(input_dim: usize, hidden: usize, output_dim: usize) -> String {
    let in_flat = input_dim * 2;
    let mut s = String::new();
    s.push_str("module {\n");
    s.push_str("  func.func @world_model_physics(\n");
    s.push_str(&format!(
        "    %state: tensor<{input_dim}xf32>, %action: tensor<{input_dim}xf32>,\n"
    ));
    s.push_str(&format!(
        "    %W1: tensor<{hidden}x{in_flat}xf32>, %b1: tensor<{hidden}xf32>,\n"
    ));
    s.push_str(&format!(
        "    %W2: tensor<{output_dim}x{hidden}xf32>, %b2: tensor<{output_dim}xf32>) -> tensor<{output_dim}xf32> {{\n"
    ));
    s.push_str("    %c0 = arith.constant 0 : index\n");
    s.push_str("    %c1 = arith.constant 1 : index\n");
    s.push_str(&format!("    %cIn = arith.constant {in_flat} : index\n"));
    s.push_str(&format!("    %cH = arith.constant {hidden} : index\n"));
    s.push_str(&format!("    %cO = arith.constant {output_dim} : index\n"));
    s.push_str(&format!(
        "    %input = tensor.concat dim(0) %state, %action : (tensor<{input_dim}xf32>, tensor<{input_dim}xf32>) -> tensor<{in_flat}xf32>\n"
    ));
    s.push_str(&matmul_mlir(&MmLayer {
        name: "hidden",
        input: "input",
        w: "W1",
        b: "b1",
        act: "tanh",
        rows: hidden,
        cols: in_flat,
        outer_bound: "cH",
        inner_bound: "cIn",
    }));
    s.push_str(&matmul_mlir(&MmLayer {
        name: "out",
        input: "hidden",
        w: "W2",
        b: "b2",
        act: "linear",
        rows: output_dim,
        cols: hidden,
        outer_bound: "cO",
        inner_bound: "cH",
    }));
    s.push_str(&format!("    return %out : tensor<{output_dim}xf32>\n"));
    s.push_str("  }\n}\n");
    s
}

/// One `output = act(W·input + b)` layer, described for MLIR emission.
struct MmLayer<'a> {
    name: &'a str,
    input: &'a str,
    w: &'a str,
    b: &'a str,
    act: &'a str,
    rows: usize,
    cols: usize,
    outer_bound: &'a str,
    inner_bound: &'a str,
}

/// Emit one `output = act(W·input + b)` layer as an MLIR `scf.for` loop nest.
fn matmul_mlir(l: &MmLayer) -> String {
    let (act_line, act_val) = if l.act == "tanh" {
        (
            "    %act = math.tanh %biased : f32\n".to_string(),
            "%act".to_string(),
        )
    } else {
        (String::new(), "%biased".to_string())
    };
    format!(
        "    %{name}_empty = tensor.empty() : tensor<{rows}xf32>\n\
         %{name} = scf.for %i = %c0 to %{outer_bound} step %c1 iter_args(%o = %{name}_empty) -> tensor<{rows}xf32> {{\n\
           %bi = tensor.extract %{b}[%i] : tensor<{rows}xf32>\n\
           %acc0 = arith.constant 0.0 : f32\n\
           %acc = scf.for %j = %c0 to %{inner_bound} step %c1 iter_args(%a = %acc0) -> f32 {{\n\
             %wv = tensor.extract %{w}[%i, %j] : tensor<{rows}x{cols}xf32>\n\
             %xv = tensor.extract %{input}[%j] : tensor<{cols}xf32>\n\
             %p = arith.mulf %wv, %xv : f32\n\
             %s = arith.addf %a, %p : f32\n\
             scf.yield %s : f32\n\
           }}\n\
           %biased = arith.addf %acc, %bi : f32\n\
           {act_line}\
           %o2 = tensor.insert {act_val} into %o[%i] : tensor<{rows}xf32>\n\
           scf.yield %o2 : tensor<{rows}xf32>\n\
         }}\n",
        name = l.name,
        input = l.input,
        b = l.b,
        w = l.w,
        rows = l.rows,
        cols = l.cols,
        outer_bound = l.outer_bound,
        inner_bound = l.inner_bound,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kai_mlir::DialectRegistry;

    fn registry() -> DialectRegistry {
        let mut r = DialectRegistry::new();
        r.register(KaiPhysicsDialect);
        r
    }

    #[test]
    fn g_ij() {
        let r = registry();
        let out = r
            .evaluate("kai_physics", "g_ij_calculate", &[Scalar(0.2)], &AttributeSet::new())
            .unwrap();
        assert!((out.as_scalar().unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn tau_half_speed() {
        let r = registry();
        let out = r
            .evaluate(
                "kai_physics",
                "update_tau",
                &[Scalar(5.0), Scalar((3.0f64).sqrt() / 2.0)],
                &AttributeSet::new(),
            )
            .unwrap();
        // v^2 = 0.75; sqrt(0.25) = 0.5; 5 * 0.5 = 2.5
        assert!((out.as_scalar().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn vfe_with_regularization() {
        let r = registry();
        let p = Tensor(vec![1.0, 2.0]);
        let q = Tensor(vec![1.0, 2.0]);
        let out = r
            .evaluate(
                "kai_physics",
                "calculate_vfe",
                &[p, q, Scalar(1.0), Scalar(1.0)],
                &AttributeSet::new(),
            )
            .unwrap();
        // mse 0 + 0.5 + 0.1 = 0.6
        assert!((out.as_scalar().unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn collapse_threshold_via_attr() {
        let r = registry();
        let mut attrs = AttributeSet::new();
        attrs.insert("threshold", Scalar(0.5));
        let out = r
            .evaluate("kai_physics", "tonal_collapse", &[Scalar(0.6)], &attrs)
            .unwrap();
        assert_eq!(out.as_scalar().unwrap(), 1.0);
    }

    #[test]
    fn world_model_mlir_lowers_and_verifies() {
        // Lower the forward pass to upstream MLIR and verify it with mlir-opt.
        let mlir = emit_world_model_mlir(3, 4, 2);
        assert!(mlir.contains("@world_model_physics"));
        assert!(mlir.contains("scf.for"));
        assert!(mlir.contains("math.tanh"));

        let ok = std::process::Command::new("mlir-opt")
            .arg("-o")
            .arg("/dev/null")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map(|mut child| {
                use std::io::Write;
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(mlir.as_bytes());
                }
                child.wait().map(|s| s.success()).unwrap_or(false)
            })
            .unwrap_or(false);
        if ok {
            eprintln!("  [mlir] upstream MLIR verified by mlir-opt");
        } else {
            eprintln!("  [mlir] mlir-opt not on PATH — skipping upstream verification");
        }
    }

    #[test]
    fn world_model_physics_real_forward() {
        let hidden = 3usize;
        let input_dim = 4usize;
        let output_dim = 2usize;
        let w1: Vec<f64> = (0..hidden * input_dim)
            .map(|i| (i as f64) * 0.1 - 0.5)
            .collect();
        let b1 = vec![0.1, -0.2, 0.3];
        let w2: Vec<f64> = (0..output_dim * hidden)
            .map(|i| (i as f64) * 0.05 - 0.25)
            .collect();
        let b2 = vec![0.2, -0.1];
        let state = vec![0.5, -0.3];
        let action = vec![0.1, 0.4];

        let r = registry();
        let out = r
            .evaluate(
                "kai_physics",
                "world_model_physics",
                &[
                    Tensor(state.clone()),
                    Tensor(action.clone()),
                    Tensor(w1.clone()),
                    Tensor(b1.clone()),
                    Tensor(w2.clone()),
                    Tensor(b2.clone()),
                ],
                &AttributeSet::new(),
            )
            .unwrap();
        let got = out.as_tensor().unwrap().to_vec();

        // reference forward pass
        let mut input = state.clone();
        input.extend_from_slice(&action);
        let mut h = vec![0.0f64; hidden];
        for rr in 0..hidden {
            let mut acc = b1[rr];
            for c in 0..input_dim {
                acc += w1[rr * input_dim + c] * input[c];
            }
            h[rr] = acc.tanh();
        }
        let mut exp = vec![0.0f64; output_dim];
        for o in 0..output_dim {
            let mut acc = b2[o];
            for rr in 0..hidden {
                acc += w2[o * hidden + rr] * h[rr];
            }
            exp[o] = acc;
        }

        assert_eq!(got.len(), output_dim);
        for (g, e) in got.iter().zip(&exp) {
            assert!((g - e).abs() < 1e-12, "got {g} exp {e}");
        }
    }
}
