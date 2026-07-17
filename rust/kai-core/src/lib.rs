//! # kai-core
//!
//! High-performance kernels backing Kai's world model and its physics-inspired
//! free-energy objective.
//!
//! The crate is written in pure Rust (no Python dependency) so the numerical
//! core can be unit-tested in isolation. Python bindings are provided behind
//! the `python` feature and are intended to be loaded by `kai_mind.py`.
//!
//! All public functions validate tensor shapes and report structured errors
//! instead of panicking on mismatched dimensions.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis};
use std::fmt;

/// Errors returned by the numerical kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KaiCoreError {
    /// A tensor had the wrong number of elements for its role.
    DimensionMismatch {
        /// Human-readable context (e.g. `"w1 input width"`).
        what: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// A required tensor was empty.
    EmptyInput(&'static str),
}

impl fmt::Display for KaiCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KaiCoreError::DimensionMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in {what}: expected {expected} elements, got {actual}"
            ),
            KaiCoreError::EmptyInput(what) => write!(f, "empty input: {what} must not be empty"),
        }
    }
}

impl std::error::Error for KaiCoreError {}

/// Parameters of the two-layer world-model network.
///
/// `w1` maps the concatenated `[state, action]` vector to the hidden layer,
/// `w2` maps the hidden layer to the prediction.
#[derive(Debug, Clone)]
pub struct WorldModelParams {
    /// Hidden weights, shape `(hidden, state + action)`.
    pub w1: Array2<f32>,
    /// Hidden bias, length `hidden`.
    pub b1: Array1<f32>,
    /// Output weights, shape `(output, hidden)`.
    pub w2: Array2<f32>,
    /// Output bias, length `output`.
    pub b2: Array1<f32>,
}

impl WorldModelParams {
    /// Number of input units (`state + action`) inferred from `w1`.
    pub fn input_dim(&self) -> usize {
        self.w1.ncols()
    }

    /// Number of hidden units inferred from `w1`.
    pub fn hidden_dim(&self) -> usize {
        self.w1.nrows()
    }

    /// Number of output units inferred from `w2`.
    pub fn output_dim(&self) -> usize {
        self.w2.nrows()
    }

    /// Validate that every tensor has a consistent shape.
    pub fn validate(&self) -> Result<(), KaiCoreError> {
        let in_dim = self.input_dim();
        let hid = self.hidden_dim();
        let out = self.output_dim();

        if in_dim == 0 {
            return Err(KaiCoreError::EmptyInput("w1 columns"));
        }
        if hid == 0 {
            return Err(KaiCoreError::EmptyInput("w1 rows / b1"));
        }
        if out == 0 {
            return Err(KaiCoreError::EmptyInput("w2 rows / b2"));
        }
        if self.b1.len() != hid {
            return Err(KaiCoreError::DimensionMismatch {
                what: "b1 length",
                expected: hid,
                actual: self.b1.len(),
            });
        }
        if self.w2.ncols() != hid {
            return Err(KaiCoreError::DimensionMismatch {
                what: "w2 input width",
                expected: hid,
                actual: self.w2.ncols(),
            });
        }
        if self.b2.len() != out {
            return Err(KaiCoreError::DimensionMismatch {
                what: "b2 length",
                expected: out,
                actual: self.b2.len(),
            });
        }
        Ok(())
    }
}

/// Forward pass of the world model: `tanh(w1 · [state, action] + b1) → w2 · h + b2`.
///
/// Returns a prediction vector of length `output_dim`.
pub fn predict(
    state: &Array1<f32>,
    action: &Array1<f32>,
    params: &WorldModelParams,
) -> Result<Array1<f32>, KaiCoreError> {
    params.validate()?;

    let s = state.len();
    let a = action.len();
    if s + a != params.input_dim() {
        return Err(KaiCoreError::DimensionMismatch {
            what: "state+action width",
            expected: params.input_dim(),
            actual: s + a,
        });
    }

    // Concatenate state and action into the network input (stays rank-1).
    let input = ndarray::concatenate(Axis(0), &[state.view(), action.view()])
        .expect("state and action share dtype and axis; concatenation cannot fail");

    let hidden = params
        .w1
        .dot(&input)
        .mapv(|v| v.tanh());
    let hidden = &hidden + &params.b1;

    let output = params.w2.dot(&hidden) + &params.b2;
    Ok(output)
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` when either vector has zero norm (rather than dividing by zero).
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> Result<f32, KaiCoreError> {
    if a.is_empty() || b.is_empty() {
        return Err(KaiCoreError::EmptyInput("cosine input"));
    }
    if a.len() != b.len() {
        return Err(KaiCoreError::DimensionMismatch {
            what: "cosine lengths",
            expected: a.len(),
            actual: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (na * nb))
    }
}

/// Variational Free Energy proxy used by Kai as a learning signal.
///
/// `vfe = mse(prediction, reality) + variance * VARIANCE_COEFF + novelty * NOVELTY_COEFF`
///
/// where `mse` is the mean squared error over the prediction vector. Lower is better:
/// the model reduces surprise by both predicting reality accurately and keeping its
/// internal states parsimonious.
pub const VARIANCE_COEFF: f32 = 0.5;
pub const NOVELTY_COEFF: f32 = 0.1;

pub fn compute_vfe(
    prediction: &Array1<f32>,
    reality: &Array1<f32>,
    variance: f32,
    novelty: f32,
) -> Result<f32, KaiCoreError> {
    if prediction.is_empty() || reality.is_empty() {
        return Err(KaiCoreError::EmptyInput("vfe input"));
    }
    if prediction.len() != reality.len() {
        return Err(KaiCoreError::DimensionMismatch {
            what: "vfe lengths",
            expected: prediction.len(),
            actual: reality.len(),
        });
    }
    let mse: f32 = prediction
        .iter()
        .zip(reality.iter())
        .map(|(p, r)| {
            let d = p - r;
            d * d
        })
        .sum::<f32>()
        / prediction.len() as f32;

    Ok(mse + variance * VARIANCE_COEFF + novelty * NOVELTY_COEFF)
}

/// Batched forward pass: `states` shape `(n, state)`, `actions` shape `(n, action)`.
///
/// Returns predictions shape `(n, output)`.
pub fn batch_predict(
    states: &Array2<f32>,
    actions: &Array2<f32>,
    params: &WorldModelParams,
) -> Result<Array2<f32>, KaiCoreError> {
    params.validate()?;

    let n = states.nrows();
    if actions.nrows() != n {
        return Err(KaiCoreError::DimensionMismatch {
            what: "batch rows",
            expected: n,
            actual: actions.nrows(),
        });
    }
    if states.ncols() + actions.ncols() != params.input_dim() {
        return Err(KaiCoreError::DimensionMismatch {
            what: "batch input width",
            expected: params.input_dim(),
            actual: states.ncols() + actions.ncols(),
        });
    }

    let mut out = Array2::zeros((n, params.output_dim()));
    for i in 0..n {
        let pred = predict(&states.row(i).to_owned(), &actions.row(i).to_owned(), params)?;
        out.row_mut(i).assign(&pred);
    }
    Ok(out)
}

/// One in-place SGD training step of the two-layer world model.
///
/// This is a faithful, allocation-light port of `KaiMind.train_wm`'s backward
/// pass (pure-Python hotspot: `hidden * (state+action) + output * hidden` MACs
/// per step). The mathematics — including the network's non-standard update
/// rule — is replicated *exactly* so results match the Python path bit-for-bit
/// in intent:
///
/// * `inp` = `[prev, action]` truncated/zero-padded to the input width `D`.
/// * `h`   = `tanh(W1 · inp + b1)`, `pred` = `W2 · h + b2`.
/// * `err[i]` = `pred[i] - actual[i]` (missing `actual` entries treated as 0).
/// * **W2 is updated first**, then `dh` is derived from the *already-updated* W2
///   (matching the Python statement order, which reads the mutated `self.W2`).
/// * W2 update uses factor `(1 - h²)·h`; `dh` uses factor `(0.8729 - h²)`.
///
/// `lr` (learning-rate schedule) and `actual` normalization stay in Python, so
/// this kernel introduces *zero* change to the learning dynamics. Returns the
/// mean-squared error over the output width, matching `mse(pred, actual)`.
#[allow(clippy::too_many_arguments)]
pub fn train_step(
    prev: ArrayView1<f32>,
    action: ArrayView1<f32>,
    actual: ArrayView1<f32>,
    mut w1: ArrayViewMut2<f32>,
    mut b1: ArrayViewMut1<f32>,
    mut w2: ArrayViewMut2<f32>,
    mut b2: ArrayViewMut1<f32>,
    lr: f32,
) -> Result<f32, KaiCoreError> {
    let hh = w1.nrows();
    let d = w1.ncols();
    let o = w2.nrows();
    if d == 0 {
        return Err(KaiCoreError::EmptyInput("w1 columns"));
    }
    if hh == 0 {
        return Err(KaiCoreError::EmptyInput("w1 rows / b1"));
    }
    if o == 0 {
        return Err(KaiCoreError::EmptyInput("w2 rows / b2"));
    }
    if b1.len() != hh {
        return Err(KaiCoreError::DimensionMismatch {
            what: "b1 length",
            expected: hh,
            actual: b1.len(),
        });
    }
    if w2.ncols() != hh {
        return Err(KaiCoreError::DimensionMismatch {
            what: "w2 input width",
            expected: hh,
            actual: w2.ncols(),
        });
    }
    if b2.len() != o {
        return Err(KaiCoreError::DimensionMismatch {
            what: "b2 length",
            expected: o,
            actual: b2.len(),
        });
    }

    // inp = [prev, action] truncated / zero-padded to width D.
    let plen = prev.len();
    let mut inp = vec![0.0f32; d];
    for (k, slot) in inp.iter_mut().enumerate() {
        *slot = if k < plen {
            prev[k]
        } else if k - plen < action.len() {
            action[k - plen]
        } else {
            0.0
        };
    }

    // Forward: h = tanh(W1 inp + b1); pred = W2 h + b2.
    let mut h = vec![0.0f32; hh];
    for j in 0..hh {
        let mut s = b1[j];
        for k in 0..d {
            s += w1[[j, k]] * inp[k];
        }
        h[j] = s.tanh();
    }
    let mut pred = vec![0.0f32; o];
    for i in 0..o {
        let mut s = b2[i];
        for j in 0..hh {
            s += w2[[i, j]] * h[j];
        }
        pred[i] = s;
    }

    // Error and MSE (missing actual entries -> 0.0).
    let mut mse = 0.0f32;
    let mut err = vec![0.0f32; o];
    for i in 0..o {
        let target = if i < actual.len() { actual[i] } else { 0.0 };
        let e = pred[i] - target;
        err[i] = e;
        mse += e * e;
    }
    mse /= o as f32;

    // Backward — W2 / b2 first (dh below reads the updated W2, as in Python).
    for i in 0..o {
        let ei = err[i];
        for j in 0..hh {
            w2[[i, j]] -= lr * ei * (1.0 - h[j] * h[j]) * h[j];
        }
        b2[i] -= lr * ei;
    }
    for j in 0..hh {
        let mut s = 0.0f32;
        for k in 0..o {
            s += err[k] * w2[[k, j]];
        }
        let dhj = s * (0.8729 - h[j] * h[j]);
        let ljk = lr * dhj;
        for k in 0..d {
            w1[[j, k]] -= ljk * inp[k];
        }
        b1[j] -= ljk;
    }

    Ok(mse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Build a tiny, deterministic parameter set for tests (dims 4 / 8 / 4).
    fn tiny_params() -> WorldModelParams {
        WorldModelParams {
            w1: array![
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            b1: array![0.0, 0.0, 0.0, 0.0],
            w2: array![
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
            ],
            b2: array![0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn params_validate_ok() {
        assert!(tiny_params().validate().is_ok());
    }

    #[test]
    fn params_validate_bad_bias() {
        let mut p = tiny_params();
        p.b1 = array![0.0, 0.0]; // wrong length
        assert_eq!(
            p.validate(),
            Err(KaiCoreError::DimensionMismatch {
                what: "b1 length",
                expected: 4,
                actual: 2
            })
        );
    }

    #[test]
    fn predict_shape_and_finite() {
        let p = tiny_params();
        let s = array![1.0, 0.0, 1.0, 0.0];
        let a = array![0.0, 1.0, 0.0, 1.0];
        let out = predict(&s, &a, &p).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn predict_rejects_width_mismatch() {
        let p = tiny_params();
        let s = array![1.0, 0.0];
        let a = array![0.0, 1.0];
        assert!(matches!(
            predict(&s, &a, &p),
            Err(KaiCoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = array![0.3, 0.4, 0.5];
        let b = array![0.3, 0.4, 0.5];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_is_zero() {
        let a = array![0.0, 0.0, 0.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn vfe_zero_when_prediction_matches() {
        let p = array![1.0, 2.0, 3.0];
        let r = array![1.0, 2.0, 3.0];
        let vfe = compute_vfe(&p, &r, 0.0, 0.0).unwrap();
        assert!(vfe.abs() < 1e-6);
    }

    #[test]
    fn vfe_increases_with_novelty() {
        let p = array![1.0, 2.0];
        let r = array![1.0, 2.0];
        let base = compute_vfe(&p, &r, 0.0, 0.0).unwrap();
        let with_novelty = compute_vfe(&p, &r, 0.0, 1.0).unwrap();
        assert!((with_novelty - base - NOVELTY_COEFF).abs() < 1e-6);
    }

    /// Independent reference implementation of one training step, used to lock
    /// down `train_step`'s exact update rule and statement ordering.
    fn train_step_reference(
        prev: &Array1<f32>,
        action: &Array1<f32>,
        actual: &Array1<f32>,
        p: &mut WorldModelParams,
        lr: f32,
    ) -> f32 {
        let d = p.w1.ncols();
        let hh = p.w1.nrows();
        let o = p.w2.nrows();
        let plen = prev.len();
        let inp: Vec<f32> = (0..d)
            .map(|k| {
                if k < plen {
                    prev[k]
                } else if k - plen < action.len() {
                    action[k - plen]
                } else {
                    0.0
                }
            })
            .collect();
        let h: Vec<f32> = (0..hh)
            .map(|j| {
                let mut s = p.b1[j];
                for k in 0..d {
                    s += p.w1[[j, k]] * inp[k];
                }
                s.tanh()
            })
            .collect();
        let pred: Vec<f32> = (0..o)
            .map(|i| {
                let mut s = p.b2[i];
                for j in 0..hh {
                    s += p.w2[[i, j]] * h[j];
                }
                s
            })
            .collect();
        let err: Vec<f32> = (0..o)
            .map(|i| pred[i] - if i < actual.len() { actual[i] } else { 0.0 })
            .collect();
        let mse = err.iter().map(|e| e * e).sum::<f32>() / o as f32;
        for i in 0..o {
            for j in 0..hh {
                p.w2[[i, j]] -= lr * err[i] * (1.0 - h[j] * h[j]) * h[j];
            }
            p.b2[i] -= lr * err[i];
        }
        for j in 0..hh {
            let mut s = 0.0f32;
            for k in 0..o {
                s += err[k] * p.w2[[k, j]];
            }
            let dhj = s * (0.8729 - h[j] * h[j]);
            for k in 0..d {
                p.w1[[j, k]] -= lr * dhj * inp[k];
            }
            p.b1[j] -= lr * dhj;
        }
        mse
    }

    #[test]
    fn train_step_matches_reference() {
        let s = array![1.0, 0.0, 1.0, 0.0];
        let a = array![0.0, 1.0, 0.0, 1.0];
        let actual = array![0.5, -0.5, 0.25, -0.25];
        let lr = 0.01f32;

        let mut p = tiny_params();
        let mse = train_step(
            s.view(),
            a.view(),
            actual.view(),
            p.w1.view_mut(),
            p.b1.view_mut(),
            p.w2.view_mut(),
            p.b2.view_mut(),
            lr,
        )
        .unwrap();

        let mut r = tiny_params();
        let mse_ref = train_step_reference(&s, &a, &actual, &mut r, lr);

        assert!((mse - mse_ref).abs() < 1e-6);
        assert!(p
            .w1
            .iter()
            .zip(r.w1.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6));
        assert!(p
            .w2
            .iter()
            .zip(r.w2.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6));
        assert!(p
            .b1
            .iter()
            .zip(r.b1.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6));
        assert!(p
            .b2
            .iter()
            .zip(r.b2.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6));
    }

    #[test]
    fn train_step_reduces_error_over_steps() {
        let s = array![0.5, -0.2, 0.1, 0.3];
        let a = array![0.1, 0.2, -0.3, 0.4];
        let actual = array![0.2, -0.1, 0.15, -0.05];
        let mut p = tiny_params();

        let first_mse = train_step(
            s.view(),
            a.view(),
            actual.view(),
            p.w1.view_mut(),
            p.b1.view_mut(),
            p.w2.view_mut(),
            p.b2.view_mut(),
            0.05,
        )
        .unwrap();
        let mut last = first_mse;
        for _ in 0..200 {
            last = train_step(
                s.view(),
                a.view(),
                actual.view(),
                p.w1.view_mut(),
                p.b1.view_mut(),
                p.w2.view_mut(),
                p.b2.view_mut(),
                0.05,
            )
            .unwrap();
        }
        assert!(last < first_mse);
        assert!(last.is_finite());
    }

    #[test]
    fn train_step_rejects_bad_bias() {
        let s = array![1.0, 0.0, 1.0, 0.0];
        let a = array![0.0, 1.0, 0.0, 1.0];
        let actual = array![0.0, 0.0, 0.0, 0.0];
        let mut p = tiny_params();
        let mut bad_b1 = array![0.0, 0.0]; // wrong length
        assert!(matches!(
            train_step(
                s.view(),
                a.view(),
                actual.view(),
                p.w1.view_mut(),
                bad_b1.view_mut(),
                p.w2.view_mut(),
                p.b2.view_mut(),
                0.01,
            ),
            Err(KaiCoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_predict_matches_loop() {
        let p = tiny_params();
        let states = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]];
        let actions = array![[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]];
        let batched = batch_predict(&states, &actions, &p).unwrap();
        assert_eq!(batched.shape(), &[2, 4]);
        for i in 0..2 {
            let single = predict(&states.row(i).to_owned(), &actions.row(i).to_owned(), &p).unwrap();
            assert!(single
                .iter()
                .zip(batched.row(i).iter())
                .all(|(x, y)| (x - y).abs() < 1e-6));
        }
    }
}

// ---------------------------------------------------------------------------
// Optional Python bindings (gated behind the `python` feature).
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
mod python {
    use super::*;
    use numpy::{PyArray1, PyArray2};
    use pyo3::prelude::*;

    fn params_from_py(
        w1: &PyArray2<f32>,
        b1: &PyArray1<f32>,
        w2: &PyArray2<f32>,
        b2: &PyArray1<f32>,
    ) -> WorldModelParams {
        WorldModelParams {
            w1: w1.to_owned_array(),
            b1: b1.to_owned_array(),
            w2: w2.to_owned_array(),
            b2: b2.to_owned_array(),
        }
    }

    #[pyfunction]
    #[pyo3(name = "predict_world_model")]
    fn py_predict<'py>(
        py: Python<'py>,
        state: &'py PyArray1<f32>,
        action: &'py PyArray1<f32>,
        w1: &'py PyArray2<f32>,
        b1: &'py PyArray1<f32>,
        w2: &'py PyArray2<f32>,
        b2: &'py PyArray1<f32>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let params = params_from_py(w1, b1, w2, b2);
        let out = predict(&state.to_owned_array(), &action.to_owned_array(), &params)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_owned_array(py, out))
    }

    #[pyfunction]
    #[pyo3(name = "train_step_world_model")]
    #[allow(clippy::too_many_arguments)]
    fn py_train_step(
        state: &PyArray1<f32>,
        action: &PyArray1<f32>,
        actual: &PyArray1<f32>,
        w1: &PyArray2<f32>,
        b1: &PyArray1<f32>,
        w2: &PyArray2<f32>,
        b2: &PyArray1<f32>,
        lr: f32,
    ) -> PyResult<f32> {
        // Mutate the caller's weight arrays in place (zero-copy), returning MSE.
        let state = state.to_owned_array();
        let action = action.to_owned_array();
        let actual = actual.to_owned_array();
        let mut w1 = unsafe { w1.as_array_mut() };
        let mut b1 = unsafe { b1.as_array_mut() };
        let mut w2 = unsafe { w2.as_array_mut() };
        let mut b2 = unsafe { b2.as_array_mut() };
        train_step(
            state.view(),
            action.view(),
            actual.view(),
            w1.view_mut(),
            b1.view_mut(),
            w2.view_mut(),
            b2.view_mut(),
            lr,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    #[pyo3(name = "compute_vfe")]
    fn py_vfe(
        prediction: &PyArray1<f32>,
        reality: &PyArray1<f32>,
        variance: f32,
        novelty: f32,
    ) -> PyResult<f32> {
        compute_vfe(
            &prediction.to_owned_array(),
            &reality.to_owned_array(),
            variance,
            novelty,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    #[pyo3(name = "cosine_similarity")]
    fn py_cosine(a: &PyArray1<f32>, b: &PyArray1<f32>) -> PyResult<f32> {
        cosine_similarity(&a.to_owned_array(), &b.to_owned_array())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pymodule]
    fn kai_core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_predict, m)?)?;
        m.add_function(wrap_pyfunction!(py_train_step, m)?)?;
        m.add_function(wrap_pyfunction!(py_vfe, m)?)?;
        m.add_function(wrap_pyfunction!(py_cosine, m)?)?;
        let _ = py;
        Ok(())
    }
}
