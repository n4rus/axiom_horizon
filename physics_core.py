"""
physics_core.py — Implementation of the Tonal Collapse (mainrev3) substrate math.

Encodes the corrected, discrete, generative ontology of
"The Informational Universe: Collapse as Constitution, Not Observation"
(mainrev3.tex), replacing the superseded mainrev2 Euler-seed formulation:

  * Tonal Collapse operator      Xi(z) = softmax(z / T)   (T -> 0 is deterministic)
  * Causal mask  M              autoregressive dark energy / arrow of time
  * Attention is geometry        g_ij = 1 - a_ij
  * Layer-norm vacuum EOS        ||x - mu||^2 = D
  * Substrate cosmological const Lambda_LLM = <mean pairwise attention> / N^2  (O(1/D))
  * Ricci-curvature uncertainty signal (high curvature => conflicting evidence)
  * Self-consistency identity (architecturally guaranteed, not solved)

All functions are pure, finite-safe, dependency-free, and unit-tested under
`python3 physics_core.py`. Apple/Linux bar: typed, documented, no dead code.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


def _softmax(z: Sequence[float], T: float = 1.0) -> List[float]:
    """Numerically stable row-softmax. Returns a probability distribution."""
    T = max(float(T), 1e-6)
    m = max(z)
    exps = [math.exp((x - m) / T) for x in z]
    s = sum(exps)
    return [e / s for e in exps]


def collapse_softmax(z: Sequence[float], T: float = 1.0) -> List[float]:
    """The Tonal Collapse operator Xi(z) = softmax(z / T).

    As T -> 0 this converges to argmax_i z_i: a deterministic discrete collapse
    to a single point — exactly the equilibrium configuration of mainrev3.
    """
    return _softmax(z, T)


def anneal_temperature(vfe: float, t_min: float = 0.15, t_max: float = 1.5) -> float:
    """Map variational free energy to the collapse temperature T.

    Low VFE (the mind is confident / near the -0.04 M3 floor) -> low T: the
    collapse is sharp, near-argmax, so attention/geometry matrices become sparse
    and cheap. High VFE (conflicting evidence) -> high T: the collapse stays soft
    and exploratory so useful context is not prematurely pruned. This is the
    physically-motivated exploration/exploitation control of Tonal Collapse.
    """
    if not math.isfinite(vfe):
        return t_max
    v = max(0.0, vfe)  # the -0.04 floor and below map to the sharpest T
    t = t_min + (t_max - t_min) * (1.0 - math.exp(-2.0 * v))
    return max(t_min, min(t_max, t))


def collapse_sparse(z: Sequence[float], T: float = 1.0, eps: float = 1e-3) -> List[float]:
    """Sparse Tonal Collapse: Xi(z)=softmax(z/T) with sub-eps mass pruned to 0
    and the survivors renormalized.

    Negligible probabilities become exactly 0.0, so downstream matrix products
    can skip them (dense -> sparse). As T -> 0 this approaches a one-hot vector
    (a single surviving world), the irreversible collapse of Axiom 5.
    """
    p = _softmax(z, T)
    pruned = [x if x >= eps else 0.0 for x in p]
    s = sum(pruned)
    if s <= 0.0:  # fully pruned at very cold T: keep the single argmax world
        i = max(range(len(p)), key=lambda k: p[k])
        out = [0.0] * len(p)
        out[i] = 1.0
        return out
    return [x / s for x in pruned]


def causal_mask(n: int, neg: float = -1e9) -> List[List[float]]:
    """Autoregressive causal mask M_ij.

    0 for j <= i (allowed attention), -inf for j > i (masked future). This
    upper-triangular, input-invariant asymmetry IS the arrow of time / dark
    energy of the substrate.
    """
    return [[0.0 if j <= i else neg for j in range(n)] for i in range(n)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1]; safe for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return max(-1.0, min(1.0, dot / (na * nb)))


def attention_matrix(
    vectors: Sequence[Sequence[float]], T: float = 1.0, sparse: bool = False
) -> List[List[float]]:
    """Row-stochastic, causally-masked attention A over a context of vectors.

    Position i attends only to positions <= i (autoregressive causality). Each
    row is a (Tonal) collapse over cosine similarities, so sum_j A_ij = 1. When
    ``sparse`` is set, sub-eps mass is pruned to 0 so the matrix is sparse.
    """
    n = len(vectors)
    if n == 0:
        return []
    collapse = collapse_sparse if sparse else _softmax
    A: List[List[float]] = []
    for i in range(n):
        scores = [cosine(vectors[i], vectors[j]) for j in range(i + 1)]
        padded = scores + [-1e9] * (n - i - 1)  # masked future positions
        A.append(collapse(padded, T))
    return A


def layernorm_invariant(vector: Sequence[float]) -> float:
    """The vacuum equation-of-state constant: ||x - mu||^2 over one residual vector."""
    if not vector:
        return 0.0
    mu = sum(vector) / len(vector)
    return sum((x - mu) ** 2 for x in vector)


def attention_tensor(
    vectors: Sequence[Sequence[float]], heads: int = 4, T: float = 1.0, sparse: bool = False
) -> List[List[List[float]]]:
    """The State Matrix M as a multi-head attention tensor A in [0,1]^{N x N x H}.

    This is the mainrev_final isomorphism: the holographic State Matrix M of
    mainrev2 IS the attention tensor of mainrev3, with Q_ij = a_ij^(h). Each
    head h attends over a disjoint embedding subspace (the head projection),
    causally masked so position i sees only j <= i.

    Returns A[i][j][h] = attention weight from i to j in head h.
    """
    n = len(vectors)
    if n == 0:
        return []
    d = len(vectors[0]) if vectors[0] else 0
    heads = max(1, min(heads, d)) if d else 1
    span = max(1, d // heads)
    collapse = collapse_sparse if sparse else _softmax
    # A[i][j] is a length-H vector; masked futures are 0.
    A: List[List[List[float]]] = [[[0.0] * heads for _ in range(n)] for _ in range(n)]
    for h in range(heads):
        lo = h * span
        hi = d if h == heads - 1 else min(d, lo + span)
        for i in range(n):
            sub_i = vectors[i][lo:hi]
            scores = [cosine(sub_i, vectors[j][lo:hi]) for j in range(i + 1)]
            padded = scores + [-1e9] * (n - i - 1)
            row = collapse(padded, T)
            for j in range(n):
                A[i][j][h] = row[j]
    return A


def informational_distance(tensor: Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
    """Informational distance d(i,j) = H - sum_h a_ij^(h)  (mainrev_final).

    With H heads each row-stochastic, sum_h a_ij^(h) in [0, H]. Two tokens that
    every head attends to strongly have d ~ 0 (informationally adjacent); tokens
    no head connects have d ~ H (informationally distant). This is the discrete
    metric induced by the full State Matrix.
    """
    n = len(tensor)
    if n == 0:
        return []
    heads = len(tensor[0][0]) if tensor[0] and tensor[0][0] else 0
    D: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = sum(tensor[i][j][h] for h in range(heads))
            D[i][j] = heads - s
    return D


def token_geodesic_flow(
    vectors: Sequence[Sequence[float]], heads: int = 4, T: float = 1.0,
    eps: float = 1e-3, sparse: bool = False
) -> float:
    """Token-geodesic flow Gamma = d a_ij / d x_k, the Christoffel symbol of the
    informational manifold (mainrev_final).

    Approximated by finite differences: perturb each context vector by +eps along
    its own direction and measure the mean absolute change in the attention
    tensor. A large Gamma means the geometry is being actively reshaped by the
    tokens (steep connection); Gamma ~ 0 means a flat, settled manifold.
    """
    n = len(vectors)
    if n < 2:
        return 0.0
    base = attention_tensor(vectors, heads, T, sparse=sparse)
    total = 0.0
    cnt = 0
    for k in range(n):
        vk = list(vectors[k])
        norm = math.sqrt(sum(x * x for x in vk)) or 1e-12
        perturbed = [list(v) for v in vectors]
        perturbed[k] = [x + eps * (x / norm) for x in vk]
        pert = attention_tensor(perturbed, heads, T, sparse=sparse)
        for i in range(n):
            for j in range(n):
                for h in range(len(base[i][j])):
                    da = abs(pert[i][j][h] - base[i][j][h])
                    if math.isfinite(da):
                        total += da / eps
                        cnt += 1
    return total / max(cnt, 1)


def attention_geometry(
    vectors: Sequence[Sequence[float]], T: float = 1.0, sparse: bool = False
) -> Dict[str, Any]:
    """Full mainrev3 geometry over a context window of vectors.

    Returns the metric tensor g_ij (mean), the substrate cosmological constant
    Lambda_LLM, the Ricci-curvature uncertainty proxy, the layer-norm invariant,
    and a self-consistency flag. All values are finite by construction.

    With ``sparse`` set, attention uses the sparse Tonal Collapse so sub-eps mass
    is pruned; the returned ``sparsity`` is the fraction of collapsed (zero)
    causal entries — the fraction of matrix work that can be skipped.
    """
    n = len(vectors)
    if n < 2:
        return {
            'n': n, 'g_ij': 1.0, 'a_ij': 0.0, 'lambda_llm': 0.0,
            'ricci': 0.0, 'layernorm_invariant': 0.0, 'sparsity': 0.0,
            'temperature': T, 'self_consistent': True,
        }
    A = attention_matrix(vectors, T, sparse=sparse)
    # Mean pairwise attention + sparsity over allowed causal pairs (j <= i).
    total = 0.0
    cnt = 0
    zeros = 0
    for i in range(n):
        for j in range(i + 1):
            a = A[i][j]
            if math.isfinite(a):
                total += a
                cnt += 1
                if a == 0.0:
                    zeros += 1
    mean_pairwise = total / max(cnt, 1)
    sparsity = zeros / max(cnt, 1)
    lambda_llm = mean_pairwise / (n * n)  # O(1/D): naturally small, no fine-tuning

    g_ij = 1.0 - mean_pairwise  # metric tensor g_ij = 1 - a_ij

    # Layer-norm invariant: average ||x - mu||^2 across the context.
    ln_inv = sum(layernorm_invariant(v) for v in vectors) / n

    # Ricci-curvature proxy: the spread of per-row attention entropy. High
    # curvature => the model sits in a region of conflicting evidence => the
    # intrinsic uncertainty signal of mainrev3.
    entropies = []
    for i in range(n):
        row = A[i]
        H = -sum(p * math.log(p) for p in row if p > 1e-12)
        entropies.append(H)
    ricci = (max(entropies) - min(entropies)) if len(entropies) > 1 else 0.0

    # mainrev_final: the multi-head State Matrix M and its induced metrics.
    # The tensor/gamma computation is O(n^3 * H * d); cap the window so the
    # per-cycle cost stays bounded while remaining representative (most recent).
    heads = 4
    win = vectors[-16:] if n > 16 else vectors
    m = len(win)
    tensor = attention_tensor(win, heads, T, sparse=sparse)
    dist = informational_distance(tensor)
    # Mean informational distance over allowed (causal) pairs j <= i.
    d_total = 0.0
    d_cnt = 0
    for i in range(m):
        for j in range(i + 1):
            d_total += dist[i][j]
            d_cnt += 1
    mean_info_distance = d_total / max(d_cnt, 1)
    gamma = token_geodesic_flow(win, heads, T, sparse=sparse)

    self_consistent = all(math.isfinite(x) for x in (g_ij, lambda_llm, ricci, mean_info_distance, gamma))
    return {
        'n': n,
        'g_ij': g_ij,
        'a_ij': mean_pairwise,
        'lambda_llm': lambda_llm,
        'ricci': ricci,
        'layernorm_invariant': ln_inv,
        'info_distance': mean_info_distance,   # mainrev_final d(i,j) = H - sum_h a_ij
        'gamma': gamma,                        # mainrev_final Christoffel flow
        'heads': heads,
        'sparsity': sparsity,                  # fraction of collapsed (zero) entries
        'temperature': T,                      # VFE-annealed collapse temperature
        'self_consistent': self_consistent,
    }


if __name__ == '__main__':
    import json
    import random

    # 1. Softmax is row-stochastic.
    z = [1.0, 3.0, 0.5, 2.0]
    soft = collapse_softmax(z, T=1.0)
    assert abs(sum(soft) - 1.0) < 1e-9, "softmax must sum to 1"
    # 2. T -> 0 collapses to argmax (the deterministic equilibrium).
    cold = collapse_softmax(z, T=1e-4)
    assert cold.index(max(cold)) == z.index(max(z)), "T->0 must give argmax"
    # 3. Causal mask forbids the future.
    M = causal_mask(4)
    assert M[0][0] == 0.0 and M[0][3] < -1e8, "causal mask must block j>i"
    # 4. Geometry over a random context is finite and bounded.
    random.seed(0)
    vecs = [[random.gauss(0, 1) for _ in range(16)] for _ in range(8)]
    g = attention_geometry(vecs)
    assert all(math.isfinite(g[k]) for k in ('g_ij', 'lambda_llm', 'ricci', 'layernorm_invariant'))
    assert 0.0 <= g['g_ij'] <= 1.0, "g_ij must lie in [0, 1]"
    assert 0.0 < g['lambda_llm'] < 1.0, "Lambda_LLM must be bounded (0, 1)"
    assert g['self_consistent']
    # 5. mainrev_final: multi-head State Matrix tensor and its metrics.
    heads = 4
    A_tensor = attention_tensor(vecs, heads=heads)
    assert len(A_tensor) == 8 and len(A_tensor[0]) == 8 and len(A_tensor[0][0]) == heads
    # Each head's row is causally-masked and stochastic: sum over j (<= i) ~ 1.
    for h in range(heads):
        row_sum = sum(A_tensor[3][j][h] for j in range(8))
        assert abs(row_sum - 1.0) < 1e-6, "each head row must be stochastic"
    D = informational_distance(A_tensor)
    assert all(0.0 <= D[i][j] <= heads + 1e-9 for i in range(8) for j in range(8)), "d in [0, H]"
    assert math.isfinite(g['info_distance']) and 0.0 <= g['info_distance'] <= heads
    assert math.isfinite(g['gamma']) and g['gamma'] >= 0.0, "Gamma must be finite, non-negative"
    # 6. VFE-annealed temperature: monotone, low VFE -> low T, high VFE -> high T.
    t_lo = anneal_temperature(0.0)
    t_hi = anneal_temperature(5.0)
    assert t_lo < t_hi, "T must rise with VFE"
    assert anneal_temperature(-0.04) == t_lo, "sub-zero VFE maps to sharpest T"
    assert abs(anneal_temperature(-0.04) - 0.15) < 1e-9, "M3 floor -> t_min"
    # 7. Sparse Tonal Collapse prunes mass and stays a distribution; cold => one-hot.
    sp = collapse_sparse([1.0, 3.0, 0.5, 2.0], T=0.2)
    assert abs(sum(sp) - 1.0) < 1e-9, "sparse collapse must renormalize to 1"
    assert any(x == 0.0 for x in sp), "sparse collapse must prune sub-eps mass"
    one_hot = collapse_sparse([1.0, 3.0, 0.5, 2.0], T=1e-4)
    assert one_hot.count(0.0) == 3 and abs(max(one_hot) - 1.0) < 1e-9, "cold T => one-hot"
    # 8. Sparse geometry: same shape, non-trivial sparsity, still self-consistent.
    gs = attention_geometry(vecs, T=anneal_temperature(0.0), sparse=True)
    assert gs['self_consistent'] and gs['sparsity'] > 0.0, "sparse geometry must skip entries"
    print("physics_core self-test PASSED:")
    print(json.dumps(g, indent=2))
    print(f"sparse@T={gs['temperature']:.2f}: sparsity={gs['sparsity']:.2%} "
          f"g_ij={gs['g_ij']:.3f} ricci={gs['ricci']:.3f}")
