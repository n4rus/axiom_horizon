#!/usr/bin/env python3
"""measure_curvature.py — empirical supplement for mainrev4.tex (Tables S1, S2).

S1: Spearman rho between attention curvature kappa_i and next-token
    surprisal -log p(t_i), >=10^4 positions, >=2 open models.
S2: Lambda_LLM = mean pairwise attention / N^2 across >=3 model sizes,
    check O(1/D) scaling.

kappa_i (paper Eq.4): mean_j |g_ij - rowmean_i|, g_ij = 1 - a_ij,
averaged over heads, then averaged over layers.

Corpus: plain text streamed from local enwiki bz2 dump (first articles
until token budget). Method note recorded in output JSON.

Usage:
  python3 tools/measure_curvature.py [--device cuda] [--tokens 12000]
                                     [--chunk 512] [--out tools/curv_results.json]
"""
import argparse, bz2, gc, json, os, re, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WIKI = "enwiki-20260601-pages-articles-multistream.xml.bz2"
MODELS_GPU = ["Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"]
MODEL_CPU3B = None  # disabled: fp16 overflow on 6GB Turing (see log); 3B CPU too slow


def stream_wiki_text(path, max_chars=400000):
    """Decompress from stream start, strip tags crudely, return text."""
    out, total = [], 0
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        while total < max_chars:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            chunk = re.sub(r"<[^>]+>", " ", chunk)
            chunk = re.sub(r"\{\{[^{}]*\}\}", " ", chunk)
            chunk = re.sub(r"\s+", " ", chunk)
            out.append(chunk)
            total += len(chunk)
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def measure_model(model_id, texts, device, chunk=512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",  # sdpa can't output_attentions
    ).to(device).eval()
    cfg = model.config
    D = getattr(cfg, "hidden_size", -1)
    n_layers = getattr(cfg, "num_hidden_layers", -1)
    kappas, surps = [], []
    ent_all = []
    skipped_nonfinite = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt",
                      truncation=True, max_length=chunk)["input_ids"].to(device)
            n = ids.shape[1]
            if n < 16:
                continue
            out = model(ids, output_attentions=True, use_cache=False)
            logits = out.logits.float()
            # surprisal of each next token
            lp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = ids[0, 1:]
            surps.extend((-lp[torch.arange(len(tgt)), tgt]).cpu().tolist())
            # kappa per position i (predicting i+1): mean over layers+heads
            kl = []
            ent = []
            layer_ok = True
            for attn in out.attentions:  # [1,H,n,n]
                a = attn[0].float()  # H,n,n
                if not bool(torch.isfinite(a).all()):
                    layer_ok = False
                    break
                g = 1.0 - a
                rowmean = g.mean(dim=-1, keepdim=True)
                kl.append((g - rowmean).abs().mean(dim=-1).mean(dim=0))  # n
                ent.append((-a.clamp_min(1e-12) *
                            a.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=0))
            if not layer_ok:
                skipped_nonfinite += 1
                del out, logits
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            kappas.extend(torch.stack(kl).mean(dim=0)[:-1].cpu().tolist())
            ents = torch.stack(ent).mean(dim=0)
            ent_all.append(ents.mean().item())
            del out, logits
            if device == "cuda":
                torch.cuda.empty_cache()
    rho, pval = spearmanr(kappas, surps)
    # Lambda_LLM proxy: mean attention mass per pair ~ 1/n^2 row-stochastic
    # mean over all pairs of mean attention = 1/n per row-sum=1 -> report 1/n^2 scaled
    info = {"model": model_id, "device": device, "D": D, "layers": n_layers,
            "positions": len(kappas), "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "mean_kappa": float(sum(kappas) / len(kappas)),
            "mean_surprisal": float(sum(surps) / len(surps)),
            "mean_entropy": float(sum(ent_all) / len(ent_all)),
            "skipped_nonfinite": skipped_nonfinite}
    del model, tok
    gc.collect()
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    return info


def lambda_scaling(model_ids_devices, texts, chunk=256):
    """Mean pairwise attention / N^2 for S2 (smaller chunks, fewer texts)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = []
    for mid, dev in model_ids_devices:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True,
            torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
            attn_implementation="eager",
        ).to(dev).eval()
        D = getattr(model.config, "hidden_size", -1)
        vals, ents = [], []
        with torch.no_grad():
            for t in texts[:6]:
                ids = tok(t, return_tensors="pt",
                          truncation=True, max_length=chunk)["input_ids"].to(dev)
                n = ids.shape[1]
                if n < 16:
                    continue
                if not bool(torch.isfinite(ids.float()).all()):
                    continue
                out = model(ids, output_attentions=True, use_cache=False)
                ms, es = [], []
                for a in out.attentions:
                    af = a[0].float()
                    if not bool(torch.isfinite(af).all()):
                        continue
                    ms.append(af.mean().item())
                    es.append(float((-af.clamp_min(1e-12) *
                                     af.clamp_min(1e-12).log()).sum(dim=-1).mean().item()))
                if not ms:
                    continue
                m = sum(ms) / len(ms)
                vals.append(m / (n * n))
                ents.append(sum(es) / len(es))
                del out
        rows.append({"model": mid, "device": dev, "D": D,
                     "lambda_llm": sum(vals) / len(vals),
                     "mean_entropy": sum(ents) / len(ents)})
        del model, tok
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default="tools/curv_results.json")
    ap.add_argument("--tex", default="tools/curv_tables.tex")
    a = ap.parse_args()
    t0 = time.time()
    print("[1/4] streaming wiki corpus...", flush=True)
    raw = stream_wiki_text(WIKI)
    print("  chars:", len(raw), flush=True)
    # split into ~chunk-token texts (rough: 4 chars/token)
    per = a.chunk * 4
    texts = [raw[i:i + per] for i in range(0, len(raw), per)]
    # trim to token budget
    keep, total = [], 0
    for t in texts:
        keep.append(t)
        total += len(t) // 4
        if total >= a.tokens * 2:  # x2 models share corpus; per-model ~tokens
            break
    print("  texts:", len(keep), flush=True)
    res = {"corpus": "enwiki-20260601 stream head, tags stripped",
           "chunk": a.chunk, "note": "kappa = Eq.4 mean over heads then layers"}
    print("[2/4] S1 on GPU models...", flush=True)
    res["S1"] = [measure_model(m, keep, a.device, a.chunk)
                 for m in MODELS_GPU]
    for r in res["S1"]:
        print("  ", r["model"], "n=%d rho=%.4f p=%.2g" %
              (r["positions"], r["spearman_rho"], r["spearman_p"]), flush=True)
    print("[3/4] S2 scaling (gpu models)...", flush=True)
    res["S2"] = lambda_scaling(
        [(m, a.device) for m in MODELS_GPU], keep)
    for r in res["S2"]:
        print("  ", r["model"], "D=%d lambda=%.3e entropy=%.4f" %
              (r["D"], r["lambda_llm"], r["mean_entropy"]), flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    # LaTeX snippet
    L = ["% auto-generated by tools/measure_curvature.py — paste into mainrev4 Supplement"]
    L.append("% Table S1")
    for r in res["S1"]:
        L.append("%% %s: n=%d rho=%.4f p=%.2g mean_kappa=%.4f mean_surprisal=%.4f mean_entropy=%.4f" % (
            r["model"], r["positions"], r["spearman_rho"], r["spearman_p"],
            r["mean_kappa"], r["mean_surprisal"], r["mean_entropy"]))
    L.append("% Table S2")
    for r in res["S2"]:
        L.append("%% %s: D=%d lambda=%.3e mean_entropy=%.4f" % (
            r["model"], r["D"], r["lambda_llm"], r["mean_entropy"]))
    open(a.tex, "w").write("\n".join(L) + "\n")
    print("[4/4] wrote", a.out, "and", a.tex,
          "(%.0fs)" % res["seconds"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
