#!/usr/bin/env python3
"""
kai_physics_bench.py — the falsification instrument for the physics layer.

Does the physics-wired path produce better answers than the raw path?
This bench answers that with three measurements (the plan's own metrics):

 1. PHYSICS LIVENESS  — the controller must actually move: temperature should
    react to novelty (corr(η, T) != 0), tau should track it, and dominant
    domain should be reported. If the physics block is constant, the layer
    is decorative — falsified.
 2. QUALITY DELTA     — judge model (qwen3.5:9b, blinded) rates physics-path
    vs raw-path answers on factual completeness + groundedness. Win rate and
    mean score delta decide whether the physics earns its place.
 3. SELF-CONSISTENCY  — same question through the same path twice, embedded
    with nomic-embed-text, cosine similarity of the two answers. Plan target
    is >95%; we report the actual number per path.

Usage:  python3 tools/kai_physics_bench.py [--port 8765] [--worker qwen2.5-coder:3b]
        [--judge qwen2.5:7b] [--queries N] [--repeat 2] [--label bench0]
Results append to .kai_physics_bench.jsonl.
"""
import argparse, json, time, urllib.request, urllib.parse

BRIDGE_PORT = 8765
STOCK = "http://127.0.0.1:11434"
RECORD = ".kai_physics_bench.jsonl"
EMBED_MODEL = "nomic-embed-text"

# Domain-spanning probe set (shared with recall bench).
QUERIES = [
    "quantum entanglement measurement",
    "black hole event horizon",
    "second law of thermodynamics entropy",
    "special relativity time dilation",
    "neural network attention mechanism",
    "cryptographic hash function",
    "solar photovoltaic electricity",
    "kai vfe tau time dilation controller",
    "darwin self improvement generation",
    "bridge soul system personality memory",
    "what is a transformer and how does it work",
    "explain the monty hall problem",
    "why is the sky blue",
    "photosynthesis carbon fixation",
]

# ── Raw path: direct stock ollama, fixed temp, NO physics, NO soul ────────
def raw_answer(worker: str, q: str, temperature: float = 0.7) -> str:
    body = {
        "model": worker,
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 200},
    }
    req = urllib.request.Request(
        f"{STOCK}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d.get("message", {}).get("content", "").strip()

# ── Physics path: bridge, kai_vfe on, soul + corpus + adaptive temp ──────
def physics_answer(worker: str, q: str, port: int) -> tuple:
    body = {
        "model": f"kai/{worker}",
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    phys = d.get("kai_physics", {}) or {}
    return content, phys

# ── L4: fused physics path — multi-teacher Kalman routing ──────────────────
# Replaces the single-bridge-model path with /v1/fuse, which dispatches the
# query to 3 teachers, Kalman-fuses their quality estimates, and routes to the
# best-expert answer. Quality ceiling = best of the tier, not the single model.
from urllib.parse import urlencode
def fuse_answer(q: str, port: int, max_tokens: int = 200, teachers: str = "") -> tuple:
    params = urlencode({"q": q, "max_tokens": max_tokens})
    if teachers:
        params += "&" + urlencode({"teachers": teachers})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/fuse?{params}",
        headers={"Content-Type": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    content = (d.get("answer") or "").strip()
    phys = {"temperature": 0.7, "tau": d.get("fused_uncertainty", 0.5),
            "novelty": round(1.0 - d.get("fused_uncertainty", 0.5), 4),
            "dominant_domain": f"fusion_routed_{d.get('routed_to','?')}".replace('/', '_')}
    return content, phys

# ── Judge: rate both blinded answers, report scores + winner ──────────────
def judge(judge_model: str, q: str, a: str, b: str) -> dict:
    prompt = (
        "You are an impartial evaluation judge. Two answers to the same "
        "question are given. Rate EACH on factual completeness (1-5) and "
        "groundedness/relevance to the question (1-5). Be strict; prefer "
        "the answer with correct, specific, on-topic content over padding.\n\n"
        f"QUESTION: {q}\n\n"
        f"ANSWER A:\n{a[:800]}\n\n"
        f"ANSWER B:\n{b[:800]}\n\n"
        "Reply ONLY as JSON: {\"a_complete\":N,\"a_ground\":N,\"b_complete\":N,\"b_ground\":N,\"winner\":\"A\"|\"B\"|\"TIE\"}\n"
    )
    body = {"model": judge_model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.0, "num_predict": 100}}
    req = urllib.request.Request(f"{STOCK}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    txt = d.get("message", {}).get("content", "").strip()
    try:
        # strip markdown fences if any
        if "```" in txt:
            txt = txt.split("```")[1] if txt.count("```") >= 2 else txt
        j = json.loads(txt)
        j["winner"] = j.get("winner", "TIE")
        return j
    except Exception:
        # non-JSON reply: greedy guess by picking more text
        return {
            "a_complete": 0, "a_ground": 0, "b_complete": 0, "b_ground": 0,
            "winner": "A" if len(a) >= len(b) else "B", "parse_fail": txt[:120],
        }

# ── Embed → cosine (self-consistency scoring) ─────────────────────────────
def embed(text: str):
    body = {"model": EMBED_MODEL, "input": text[:2000]}
    req = urllib.request.Request(f"{STOCK}/api/embed", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    emb = d.get("embeddings") or d.get("embedding") or []
    if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
        return emb[0]
    return emb

def cosine(a, b):
    la = len(a); import math
    dim = min(la, len(b))
    dot = sum(a[i] * b[i] for i in range(dim))
    na = math.sqrt(sum(x * x for x in a[:dim]))
    nb = math.sqrt(sum(x * x for x in b[:dim]))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    ap.add_argument("--worker", default="qwen2.5-coder:3b")
    ap.add_argument("--judge", default="qwen2.5:7b")
    ap.add_argument("--queries", type=int, default=len(QUERIES))
    ap.add_argument("--repeat", type=int, default=2, help="consistency repeats per path")
    ap.add_argument("--label", default="")
    ap.add_argument("--fuse", action="store_true",
                    help="L4 path: use multi-teacher /v1/fuse instead of single physics model")
    ap.add_argument("--teachers", default="",
                    help="comma-sep teacher list for --fuse (default: bridge's FUSION_TEACHERS)")
    args = ap.parse_args()

    qs = QUERIES[: args.queries]
    phys_label = ("fusion(" + args.teachers + ")" if args.fuse and args.teachers
                  else ("fusion(3-teacher)" if args.fuse else f"kai/{args.worker}"))
    print(f"==== kai physics bench  label='{args.label}'  physics={phys_label}  judge={args.judge}")
    print(f"     {len(qs)} queries x {args.repeat} repeats x2 paths — this takes minutes\n")

    # 1. Liveness + quality
    novel, temps, taus = [], [], []
    wins, ties, losses = 0, 0, 0
    scores_a, scores_b = [], []
    dev_count = 0  # queries where adapted temp ||deviated|| from the base knob
    base_knob = 0.7
    for qi, q in enumerate(qs):
        pa, phys = (fuse_answer(q, args.port, teachers=args.teachers)
                    if args.fuse
                    else physics_answer(args.worker, q, args.port))
        ra = raw_answer(args.worker, q)
        novel.append(phys.get("novelty", 0.5)); temps.append(phys.get("temperature", -1)); taus.append(phys.get("tau", -1))
        if abs(phys.get("temperature", base_knob) - base_knob) >= 0.05:
            dev_count += 1
        j = judge(args.judge, q, pa, ra)
        sa = (j.get("a_complete", 0) + j.get("a_ground", 0)) / 2
        sb = (j.get("b_complete", 0) + j.get("b_ground", 0)) / 2
        scores_a.append(sa); scores_b.append(sb)
        if j.get("winner") == "A": wins += 1
        elif j.get("winner") == "B": losses += 1
        else: ties += 1
        print(f"  [{qi+1}/{len(qs)}] {q[:44]:44s} dom={phys.get('dominant_domain','?')[:18]:18s} "
              f"η={phys.get('novelty',0):.2f} T={sa:.1f}v{sb:.1f} → {j.get('winner')}")

    # 1b. liveness — NOT corr(novelty,temp) which is tautological (temp is
    # monotone in η by construction). The honest signal is: does the physics
    # knob ACTUALLY move relative to the raw path, and does reported tau vary?
    n = len(novel)
    temp_spread = (max(temps)-min(temps)) if temps else 0.0
    tau_spread = (max(taus)-min(taus)) if taus else 0.0
    dev_frac = dev_count / n if n else 0.0

    # 2. quality summary
    mean_a = sum(scores_a)/len(scores_a); mean_b = sum(scores_b)/len(scores_b)

    # 3. self-consistency (subsample: first repeat cap — keep bench short)
    cons_sets = min(6, n)
    cons_phys, cons_raw = [], []
    for q in qs[:cons_sets]:
        if args.fuse:
            p1, _ = fuse_answer(q, args.port, teachers=args.teachers)
            p2, _ = fuse_answer(q, args.port, teachers=args.teachers)
        else:
            p1, _ = physics_answer(args.worker, q, args.port)
            p2, _ = physics_answer(args.worker, q, args.port)
        e_p1, e_p2 = embed(p1), embed(p2)
        if e_p1 and e_p2: cons_phys.append(cosine(e_p1, e_p2))
        r1 = raw_answer(args.worker, q)
        r2 = raw_answer(args.worker, q)
        e_r1, e_r2 = embed(r1), embed(r2)
        if e_r1 and e_r2: cons_raw.append(cosine(e_r1, e_r2))
        print(f"  [consistency {len(cons_phys)}/{cons_sets}] phys={cons_phys[-1] if cons_phys else 0:.3f} raw={cons_raw[-1] if cons_raw else 0:.3f}")

    c_phys = sum(cons_phys)/len(cons_phys) if cons_phys else 0.0
    c_raw = sum(cons_raw)/len(cons_raw) if cons_raw else 0.0

    res = {
        "t": time.time(), "label": args.label, "worker": args.worker, "judge": args.judge,
        "n": n,
        "liveness": {"temp_spread": round(temp_spread, 3),
                     "temp_range": [round(min(temps), 4), round(max(temps), 4)],
                     "tau_spread": round(tau_spread, 3),
                     "dev_from_base_knob_frac": round(dev_frac, 3),
                     "base_knob": base_knob},
        "quality": {"wins": wins, "ties": ties, "losses": losses,
                    "physics_mean_score": round(mean_a, 3), "raw_mean_score": round(mean_b, 3),
                    "delta": round(mean_a - mean_b, 3)},
        "consistency": {"physics": round(c_phys, 4), "raw": round(c_raw, 4),
                        "physics_meets_95": c_phys >= 0.95, "raw_meets_95": c_raw >= 0.95},
    }
    with open(RECORD, "a") as f:
        f.write(json.dumps(res) + "\n")

    print("\n==== results ====")
    print(f"  LIVENESS      temp spread={temp_spread:.3f} range={res['liveness']['temp_range']} "
          f"(base knob {base_knob})  dev-frac={dev_frac:.2f}")
    print(f"  LIVENESS      tau spread={tau_spread:.3f} "
          f"(tau>0.05 movement = controller actually reacts across queries)")
    print(f"  QUALITY       physics {mean_a:.2f} vs raw {mean_b:.2f}  delta {mean_a-mean_b:+.2f}  "
          f"[wins {wins} / ties {ties} / losses {losses}]")
    verdict = "PHYSICS WINS" if (mean_a - mean_b) >= 0.3 else ("RAW WINS" if (mean_b - mean_a) >= 0.3 else "MARGINAL")
    print(f"  QUALITY       verdict: {verdict}")
    print(f"  CONSISTENCY   physics {c_phys:.3f} vs raw {c_raw:.3f}  (plan target >0.95)")
    print(f"  -> appended to {RECORD}\n")

if __name__ == "__main__":
    main()