//! # kai-mlir
//!
//! A small, self-contained intermediate representation (IR) modeled on the
//! structural concepts of MLIR: dialects, operations, types, attributes, and a
//! registry. It is intentionally dependency-free so it compiles anywhere Rust
//! does.
//!
//! The long-term intent is to *lower* these dialects to upstream MLIR. Until
//! that toolchain is available in the build environment, this standalone IR
//! gives Kai a typed, verifiable representation of its compute graph.
//!
//! Two dialects are provided here:
//! - [`KaiDialect`] — the generic reference dialect (world model, attention
//!   geometry, time dilation, free energy, tonal collapse).

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// A typed value that flows through the IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A single scalar (used for `g_ij`, `τ`, VFE, …).
    Scalar(f64),
    /// A flat tensor (used for state/action/prediction vectors).
    Tensor(Vec<f64>),
}

impl Value {
    pub fn scalar(x: f64) -> Self {
        Value::Scalar(x)
    }

    pub fn tensor(t: Vec<f64>) -> Self {
        Value::Tensor(t)
    }

    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(x) => Some(*x),
            Value::Tensor(_) => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&[f64]> {
        match self {
            Value::Tensor(t) => Some(t),
            Value::Scalar(_) => None,
        }
    }
}

/// Identifier of an operation within its dialect.
pub type OpId = &'static str;
/// Identifier of a type within its dialect.
pub type TypeId = &'static str;

/// A dialect-level type definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeDef {
    pub name: TypeId,
}

impl TypeDef {
    pub const fn new(name: TypeId) -> Self {
        TypeDef { name }
    }
}

/// A compile-time constant attached to an operation.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub key: &'static str,
    pub value: Value,
}

/// An ordered, keyed collection of attributes.
#[derive(Debug, Clone, Default)]
pub struct AttributeSet {
    attrs: Vec<Attribute>,
}

impl AttributeSet {
    pub fn new() -> Self {
        AttributeSet::default()
    }

    pub fn insert(&mut self, key: &'static str, value: Value) {
        if let Some(slot) = self.attrs.iter_mut().find(|a| a.key == key) {
            slot.value = value;
        } else {
            self.attrs.push(Attribute { key, value });
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.attrs.iter().find(|a| a.key == key).map(|a| &a.value)
    }
}

/// Errors raised during IR verification or evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrError {
    UnknownDialect(&'static str),
    UnknownOperation { dialect: &'static str, op: OpId },
    OperandCountMismatch { op: OpId, expected: usize, actual: usize },
    ExpectedScalar { op: OpId, index: usize },
    ExpectedTensor { op: OpId, index: usize },
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrError::UnknownDialect(d) => write!(f, "unknown dialect: {d}"),
            IrError::UnknownOperation { dialect, op } => {
                write!(f, "unknown operation: {dialect}.{op}")
            }
            IrError::OperandCountMismatch { op, expected, actual } => write!(
                f,
                "operation {op} expects {expected} operands, got {actual}"
            ),
            IrError::ExpectedScalar { op, index } => {
                write!(f, "operation {op} operand {index} must be a scalar")
            }
            IrError::ExpectedTensor { op, index } => {
                write!(f, "operation {op} operand {index} must be a tensor")
            }
        }
    }
}

impl Error for IrError {}

/// An operation in a dialect.
pub trait Operation: Send + Sync {
    /// The operation's mnemonic (e.g. `"g_ij_calculate"`).
    fn name(&self) -> OpId;
    /// The dialect that owns this operation.
    fn dialect(&self) -> &'static str;
    /// Number of operand values consumed.
    fn operands(&self) -> usize;
    /// Number of result values produced.
    fn results(&self) -> usize;
    /// Human-readable description of the optimization/semantics it provides.
    fn benefits(&self) -> &'static str;
    /// Structural verification (operand count and shapes/types).
    fn verify(&self, operands: &[Value]) -> Result<(), IrError>;
    /// Semantic evaluation on concrete operands.
    fn apply(&self, operands: &[Value], attrs: &AttributeSet) -> Result<Value, IrError>;
}

/// A dialect: a namespace of operations and types.
pub trait Dialect: Send + Sync {
    fn name(&self) -> &'static str;
    fn operations(&self) -> Vec<Arc<dyn Operation>>;
    fn types(&self) -> Vec<TypeDef>;
}

/// A registry of dialects keyed by name.
#[derive(Default)]
pub struct DialectRegistry {
    dialects: HashMap<&'static str, Arc<dyn Dialect>>,
}

impl DialectRegistry {
    pub fn new() -> Self {
        DialectRegistry::default()
    }

    /// Register a dialect, taking ownership of it.
    pub fn register<D: Dialect + 'static>(&mut self, dialect: D) {
        self.dialects.insert(dialect.name(), Arc::new(dialect));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Dialect>> {
        self.dialects.get(name).cloned()
    }

    /// Look up a specific operation by `dialect.op`.
    pub fn operation(&self, dialect: &str, op: OpId) -> Option<Arc<dyn Operation>> {
        self.get(dialect)?
            .operations()
            .into_iter()
            .find(|o| o.name() == op)
    }

    /// Convenience: verify and apply an operation by name.
    pub fn evaluate(
        &self,
        dialect: &'static str,
        op: OpId,
        operands: &[Value],
        attrs: &AttributeSet,
    ) -> Result<Value, IrError> {
        let operation = self
            .operation(dialect, op)
            .ok_or(IrError::UnknownOperation { dialect, op })?;
        operation.verify(operands)?;
        operation.apply(operands, attrs)
    }
}

// ---------------------------------------------------------------------------
// Reference `kai` dialect.
// ---------------------------------------------------------------------------

/// `world_model_predict` — alias for the two-layer forward pass (shape-checked).
pub struct WorldModelOp;
impl Operation for WorldModelOp {
    fn name(&self) -> OpId {
        "world_model_predict"
    }
    fn dialect(&self) -> &'static str {
        "kai"
    }
    fn operands(&self) -> usize {
        2
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Vectorized world-model forward pass"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 2 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 2,
                actual: operands.len(),
            });
        }
        if operands[0].as_tensor().is_none() || operands[1].as_tensor().is_none() {
            return Err(IrError::ExpectedTensor {
                op: self.name(),
                index: if operands[0].as_tensor().is_none() { 0 } else { 1 },
            });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        // Returns the concatenated input as a stand-in prediction; the numeric
        // kernel in `kai-core` performs the actual matmul.
        let s = operands[0].as_tensor().unwrap();
        let a = operands[1].as_tensor().unwrap();
        let mut v = Vec::with_capacity(s.len() + a.len());
        v.extend_from_slice(s);
        v.extend_from_slice(a);
        Ok(Value::Tensor(v))
    }
}

/// `g_ij_calculate` — attention geometry: `g_ij = 1 - a_ij`.
pub struct AttentionGeometryOp;
impl Operation for AttentionGeometryOp {
    fn name(&self) -> OpId {
        "g_ij_calculate"
    }
    fn dialect(&self) -> &'static str {
        "kai"
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
        let a = operands[0].as_scalar().unwrap();
        Ok(Value::Scalar(1.0 - a))
    }
}

/// `update_tau` — time dilation: `τ' = τ · sqrt(1 - v²)` with `c = 1`.
pub struct TimeDilationOp;
impl Operation for TimeDilationOp {
    fn name(&self) -> OpId {
        "update_tau"
    }
    fn dialect(&self) -> &'static str {
        "kai"
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
        let factor = (1.0 - v * v).max(0.0).sqrt();
        Ok(Value::Scalar(tau * factor))
    }
}

/// `compute_vfe` — variational free energy proxy from prediction vs. reality.
pub struct VfeComputeOp;
impl Operation for VfeComputeOp {
    fn name(&self) -> OpId {
        "compute_vfe"
    }
    fn dialect(&self) -> &'static str {
        "kai"
    }
    fn operands(&self) -> usize {
        2
    }
    fn results(&self) -> usize {
        1
    }
    fn benefits(&self) -> &'static str {
        "Physics: variational free energy (mean squared error)"
    }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 2 {
            return Err(IrError::OperandCountMismatch {
                op: self.name(),
                expected: 2,
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
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let p = operands[0].as_tensor().unwrap();
        let r = operands[1].as_tensor().unwrap();
        let mse: f64 = p
            .iter()
            .zip(r.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            / p.len() as f64;
        Ok(Value::Scalar(mse))
    }
}

/// `tonal_collapse` — convergence test from an attention similarity value.
pub struct TonalCollapseOp;
impl Operation for TonalCollapseOp {
    fn name(&self) -> OpId {
        "tonal_collapse"
    }
    fn dialect(&self) -> &'static str {
        "kai"
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
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        // Collapse is reached when similarity exceeds the (default) threshold 0.95.
        let similarity = operands[0].as_scalar().unwrap();
        Ok(Value::Scalar(if similarity >= 0.95 { 1.0 } else { 0.0 }))
    }
}

/// `surprisal_vfe` — VFE proxy from token probability: `-ln(p)`.
/// Used during generation: high surprisal = model is uncertain about its prediction.
pub struct SurprisalVfeOp;
impl Operation for SurprisalVfeOp {
    fn name(&self) -> OpId { "surprisal_vfe" }
    fn dialect(&self) -> &'static str { "kai" }
    fn operands(&self) -> usize { 1 }
    fn results(&self) -> usize { 1 }
    fn benefits(&self) -> &'static str { "Physics: VFE proxy from surprisal -ln(p)" }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 1 {
            return Err(IrError::OperandCountMismatch { op: self.name(), expected: 1, actual: operands.len() });
        }
        if operands[0].as_scalar().is_none() {
            return Err(IrError::ExpectedScalar { op: self.name(), index: 0 });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let p = operands[0].as_scalar().unwrap().max(1e-12);
        Ok(Value::Scalar(-p.ln()))
    }
}

/// `compute_curvature` — attention curvature: `1 - max(softmax)`.
/// High when attention is diffuse across many tokens; low when focused on one.
pub struct ComputeCurvatureOp;
impl Operation for ComputeCurvatureOp {
    fn name(&self) -> OpId { "compute_curvature" }
    fn dialect(&self) -> &'static str { "kai" }
    fn operands(&self) -> usize { 1 }
    fn results(&self) -> usize { 1 }
    fn benefits(&self) -> &'static str { "Physics: attention curvature 1 - max(p)" }
    fn verify(&self, operands: &[Value]) -> Result<(), IrError> {
        if operands.len() != 1 {
            return Err(IrError::OperandCountMismatch { op: self.name(), expected: 1, actual: operands.len() });
        }
        if operands[0].as_tensor().is_none() {
            return Err(IrError::ExpectedTensor { op: self.name(), index: 0 });
        }
        Ok(())
    }
    fn apply(&self, operands: &[Value], _attrs: &AttributeSet) -> Result<Value, IrError> {
        let probs = operands[0].as_tensor().unwrap();
        let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
        Ok(Value::Scalar(1.0 - max_p))
    }
}

/// The generic `kai` dialect.
pub struct KaiDialect;
impl Dialect for KaiDialect {
    fn name(&self) -> &'static str {
        "kai"
    }
    fn operations(&self) -> Vec<Arc<dyn Operation>> {
        vec![
            Arc::new(WorldModelOp),
            Arc::new(AttentionGeometryOp),
            Arc::new(TimeDilationOp),
            Arc::new(VfeComputeOp),
            Arc::new(TonalCollapseOp),
            Arc::new(SurprisalVfeOp),
            Arc::new(ComputeCurvatureOp),
        ]
    }
    fn types(&self) -> Vec<TypeDef> {
        vec![
            TypeDef::new("kai_state"),
            TypeDef::new("kai_action"),
            TypeDef::new("kai_params"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> DialectRegistry {
        let mut r = DialectRegistry::new();
        r.register(KaiDialect);
        r
    }

    #[test]
    fn registry_knows_kai() {
        let r = registry();
        assert!(r.get("kai").is_some());
        assert!(r.get("nonexistent").is_none());
    }

    #[test]
    fn attention_geometry_g_ij() {
        let r = registry();
        let out = r
            .evaluate("kai", "g_ij_calculate", &[Value::scalar(0.7)], &AttributeSet::new())
            .unwrap();
        assert!((out.as_scalar().unwrap() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn attention_geometry_rejects_tensor() {
        let r = registry();
        let err = r
            .evaluate(
                "kai",
                "g_ij_calculate",
                &[Value::tensor(vec![0.7])],
                &AttributeSet::new(),
            )
            .unwrap_err();
        assert!(matches!(err, IrError::ExpectedScalar { .. }));
    }

    #[test]
    fn time_dilation_at_rest() {
        let r = registry();
        let out = r
            .evaluate("kai", "update_tau", &[Value::scalar(10.0), Value::scalar(0.0)], &AttributeSet::new())
            .unwrap();
        assert!((out.as_scalar().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn time_dilation_slows() {
        let r = registry();
        let out = r
            .evaluate("kai", "update_tau", &[Value::scalar(10.0), Value::scalar(0.6)], &AttributeSet::new())
            .unwrap();
        // 0.6^2 = 0.36; sqrt(0.64) = 0.8; 10 * 0.8 = 8
        assert!((out.as_scalar().unwrap() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn vfe_matches_manual() {
        let r = registry();
        let p = Value::tensor(vec![1.0, 2.0, 3.0]);
        let q = Value::tensor(vec![1.0, 2.0, 3.0]);
        let out = r.evaluate("kai", "compute_vfe", &[p, q], &AttributeSet::new()).unwrap();
        assert!(out.as_scalar().unwrap().abs() < 1e-12);
    }

    #[test]
    fn tonal_collapse_threshold() {
        let r = registry();
        let low = r
            .evaluate("kai", "tonal_collapse", &[Value::scalar(0.5)], &AttributeSet::new())
            .unwrap();
        let high = r
            .evaluate("kai", "tonal_collapse", &[Value::scalar(0.99)], &AttributeSet::new())
            .unwrap();
        assert_eq!(low.as_scalar().unwrap(), 0.0);
        assert_eq!(high.as_scalar().unwrap(), 1.0);
    }

    #[test]
    fn unknown_operation_errors() {
        let r = registry();
        let err = r
            .evaluate("kai", "does_not_exist", &[], &AttributeSet::new())
            .unwrap_err();
        assert!(matches!(
            err,
            IrError::UnknownOperation {
                dialect: "kai",
                op: "does_not_exist"
            }
        ));
    }

    #[test]
    fn surprisal_vfe_certain() {
        let r = registry();
        // p = 1.0 → -ln(1) = 0
        let out = r.evaluate("kai", "surprisal_vfe", &[Value::scalar(1.0)], &AttributeSet::new()).unwrap();
        assert!(out.as_scalar().unwrap().abs() < 1e-12);
    }

    #[test]
    fn surprisal_vfe_uncertain() {
        let r = registry();
        // p = 0.5 → -ln(0.5) ≈ 0.693
        let out = r.evaluate("kai", "surprisal_vfe", &[Value::scalar(0.5)], &AttributeSet::new()).unwrap();
        assert!((out.as_scalar().unwrap() - 0.693_147).abs() < 1e-4);
    }

    #[test]
    fn compute_curvature_sharp() {
        let r = registry();
        // One token dominates with p ≈ 1.0 → curvature ≈ 0
        let probs = vec![0.001, 0.998, 0.001];
        let out = r.evaluate("kai", "compute_curvature", &[Value::tensor(probs)], &AttributeSet::new()).unwrap();
        assert!((out.as_scalar().unwrap() - (1.0 - 0.998)).abs() < 1e-4);
    }

    #[test]
    fn compute_curvature_diffuse() {
        let r = registry();
        // All tokens similar probability → curvature closer to 1
        let probs = vec![0.33, 0.34, 0.33];
        let out = r.evaluate("kai", "compute_curvature", &[Value::tensor(probs)], &AttributeSet::new()).unwrap();
        assert!((out.as_scalar().unwrap() - (1.0 - 0.34)).abs() < 1e-4);
    }
}
