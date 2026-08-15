#!/usr/bin/env python3
"""
kai_darwin_corpus.py — Darwin evolution over the corpus attractor.

Evolves VFE physics params (tau_min/max, novelty_scale, vfe_tau_rate)
by evaluating each candidate on samples from the real 290k-file corpus.

Fitness = low VFE (accuracy) + high novelty×curvature (exploration) + speed
Weighted against the attractor embedding manifold.

Usage:
  python3 tools/kai_darwin_corpus.py [--generations 20] [--population 12]
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import urllib.request

ROOT = "/home/l/Desktop/AxiomTree/axiom_horizon"
ATTRACTOR_FILE = os.path.join(ROOT, ".kai_corpus_attractor.json")
BRIDGE_URL = "http://localhost:8765/v1/chat/completions"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Physics param ranges
PARAM_BOUNDS = {
    "novelty_scale": (0.0, 1.5),
    "vfe_tau_rate": (0.01, 0.5),
    "tau_min": (0.1, 1.0),
    "tau_max": (1.0, 5.0),
    "base_temperature": (0.1, 2.0),
    "top_p": (0.5, 1.0),
}

DEFAULT_PARAMS = {
    "novelty_scale": 0.3,
    "vfe_tau_rate": 0.1,
    "tau_min": 0.7,
    "tau_max": 1.5,
    "base_temperature": 1.0,
    "top_p": 0.9,
}

# Evaluation corpus — sample from attractor
N_EVAL_SAMPLES = 20
N_GENERATIONS = 20
N_POPULATION = 12
MUTATION_RATE = 0.15
PROMOTION_THRESHOLD = 0.5


def load_attractor() -> dict:
    with open(ATTRACTOR_FILE) as f:
        return json.load(f)


def embed_text(text: str) -> list:
    """Embed via Ollama nomic-embed-text."""
    payload = json.dumps({
        "model": "nomic-embed-text",
        "prompt": text[:8192],
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception as e:
        print(f"  [WARN] embed failed: {e}", file=sys.stderr)
        return []


def cosine_sim(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma < 1e-10 or mb < 1e-10:
        return 0.0
    return dot / (ma * mb)


def call_bridge(messages: list, params: dict, max_tokens: int = 5) -> dict:
    """Call the Kai bridge with given physics params. Returns response + timing."""
    body = {
        "model": "kai/tinyllama",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": params.get("base_temperature", 1.0),
        "top_p": params.get("top_p", 0.9),
        "kai_vfe": True,
        "kai_tau": 1.0,
        "kai_novelty_scale": params.get("novelty_scale", 0.3),
        "kai_vfe_tau_rate": params.get("vfe_tau_rate", 0.1),
        "kai_tau_min": params.get("tau_min", 0.7),
        "kai_tau_max": params.get("tau_max", 1.5),
        "kai_attractor_path": "/tmp/kai_data/attractor.bin",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BRIDGE_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - t0
        return {
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "elapsed": elapsed,
            "usage": result.get("usage", {}),
        }
    except Exception as e:
        return {"response": "", "elapsed": 999, "usage": {}, "error": str(e)}


def evaluate_params(params: dict, attractor: dict, eval_queries: list) -> dict:
    """Evaluate a set of physics params on corpus samples.
    
    Fitness metrics:
    - avg_vfe_accuracy: how well responses match expected corpus patterns
      (measured by cosine sim between response embedding and attractor entry)
    - avg_exploration: average novelty×curvature proxy from response diversity
    - avg_speed: tokens per second
    """
    total_vfe = 0.0
    total_novelty = 0.0
    total_speed = 0.0
    n = 0
    
    for query, expected_path, expected_text in eval_queries:
        result = call_bridge(
            [{"role": "user", "content": query}],
            params,
            max_tokens=10,
        )
        n += 1
        
        resp_text = result.get("response", "")
        elapsed = result.get("elapsed", 1.0)
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 1)
        
        # Speed
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0
        total_speed += tps
        
        # Novelty proxy: character diversity of response
        if resp_text:
            unique_ratio = len(set(resp_text.lower())) / max(len(resp_text), 1)
            total_novelty += min(1.0, unique_ratio * 2.0)  # scale up
        else:
            total_novelty += 0.5
        
        # VFE accuracy: how well does response align with expected corpus entry?
        if resp_text and expected_text:
            resp_emb = embed_text(resp_text[:200])
            exp_emb = embed_text(expected_text[:200])
            if resp_emb and exp_emb:
                sim = cosine_sim(resp_emb, exp_emb)
                # VFE = 1 - similarity (0 = perfect match, 1 = totally different)
                vfe = max(0.0, 1.0 - sim)
                total_vfe += vfe
            else:
                total_vfe += 0.5
        else:
            total_vfe += 0.5
    
    avg_vfe = total_vfe / max(n, 1)
    avg_novelty = total_novelty / max(n, 1)
    avg_speed = total_speed / max(n, 1)
    
    # Fitness: low VFE (good prediction) + high novelty (good exploration) + speed
    vfe_score = (1.0 - min(avg_vfe, 1.0)) * 0.4  # 40%
    explore_score = min(avg_novelty, 1.0) * 0.4    # 40%
    speed_score = min(avg_speed / 5.0, 1.0) * 0.2  # 20%
    fitness = vfe_score + explore_score + speed_score
    
    return {
        "fitness": fitness,
        "avg_vfe": avg_vfe,
        "avg_novelty": avg_novelty,
        "avg_speed": avg_speed,
        "vfe_score": vfe_score,
        "explore_score": explore_score,
        "speed_score": speed_score,
    }


def mutate_params(params: dict, rate: float = MUTATION_RATE) -> dict:
    """Mutate physics params by random perturbation."""
    new = copy.deepcopy(params)
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key not in new:
            continue
        if random.random() < rate:
            # Gaussian perturbation, 20% of range
            range_size = hi - lo
            sigma = range_size * 0.2
            new[key] += random.gauss(0, sigma)
            new[key] = max(lo, min(hi, new[key]))
    return new


def crossover_params(p1: dict, p2: dict) -> dict:
    """Uniform crossover between two parent param sets."""
    child = {}
    for key in p1:
        if key in p2:
            child[key] = random.choice([p1[key], p2[key]])
        else:
            child[key] = p1[key]
    return child


def sample_eval_queries(attractor: dict, n: int = N_EVAL_SAMPLES) -> list:
    """Sample evaluation queries from the corpus attractor."""
    entries = attractor.get("entries", {})
    keys = list(entries.keys())
    
    # Stratify: pick from different parts of the tree
    queries = []
    
    # Priority to root docs and key modules
    priority_prefixes = ["__root__", "rust/kai-fusion", "sources/llama.cpp", 
                         "sources/ollama", "kai", "llvm-project/mlir"]
    
    for prefix in priority_prefixes:
        matching = [k for k in keys if k.startswith(prefix)]
        if matching:
            pick = random.choice(matching)
            entry = entries[pick]
            text = entry.get("text", "")[:300]
            if text:
                # Generate a natural query from the entry
                path_parts = pick.split("/")
                topic = path_parts[-1] if path_parts[-1] != pick else pick
                queries.append((
                    f"Tell me about {topic} in the project",
                    pick,
                    text,
                ))
    
    # Fill remaining with random samples
    remaining = n - len(queries)
    if remaining > 0:
        random.shuffle(keys)
        for k in keys[:remaining * 2]:
            if len(queries) >= n:
                break
            entry = entries[k]
            text = entry.get("text", "")[:200]
            if text and len(text) > 20:
                path_parts = k.split("/")
                topic = path_parts[-1] if path_parts[-1] != k else k
                queries.append((
                    f"What is {topic} in the codebase?",
                    k,
                    text,
                ))
    
    return queries[:n]


def run_generation(population: list, attractor: dict, eval_queries: list,
                   gen: int) -> list:
    """Evaluate all candidates in the population."""
    print(f"\n=== Generation {gen} ===", file=sys.stderr)
    print(f"  Population: {len(population)} candidates", file=sys.stderr)
    
    results = []
    for i, params in enumerate(population):
        print(f"  Evaluating candidate {i+1}/{len(population)}...", file=sys.stderr)
        score = evaluate_params(params, attractor, eval_queries)
        results.append((score["fitness"], params, score))
        print(f"    fitness={score['fitness']:.4f} "
              f"vfe={score['avg_vfe']:.3f} "
              f"novelty={score['avg_novelty']:.3f} "
              f"speed={score['avg_speed']:.1f}t/s",
              file=sys.stderr)
    
    # Sort by fitness
    results.sort(key=lambda x: -x[0])
    return results


def main():
    parser = argparse.ArgumentParser(description="Darwin evolution over corpus")
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--population", type=int, default=N_POPULATION)
    parser.add_argument("--samples", type=int, default=N_EVAL_SAMPLES)
    parser.add_argument("--resume", type=str, help="Resume from saved state file")
    args = parser.parse_args()
    
    print("Loading corpus attractor...", file=sys.stderr)
    attractor = load_attractor()
    print(f"  {attractor['total_entries']} entries", file=sys.stderr)
    
    print("Sampling evaluation queries...", file=sys.stderr)
    eval_queries = sample_eval_queries(attractor, args.samples)
    print(f"  {len(eval_queries)} queries", file=sys.stderr)
    
    # Initialize or resume population
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            state = json.load(f)
        population = state.get("population", [])
        start_gen = state.get("generation", 0)
        best_fitness = state.get("best_fitness", 0.0)
        print(f"Resumed from generation {start_gen}, best fitness {best_fitness:.4f}", file=sys.stderr)
    else:
        # Seed with default params, mutate to create population
        population = []
        for i in range(args.population):
            if i == 0:
                params = dict(DEFAULT_PARAMS)
            else:
                params = mutate_params(DEFAULT_PARAMS, rate=0.3)
            population.append(params)
        start_gen = 0
        best_fitness = 0.0
    
    best_params = dict(DEFAULT_PARAMS)
    history = []
    
    for gen in range(start_gen, args.generations):
        results = run_generation(population, attractor, eval_queries, gen)
        
        # Promote top candidates
        top_fitness = results[0][0]
        if top_fitness > best_fitness:
            best_fitness = top_fitness
            best_params = results[0][1]
        
        # Record history
        gen_record = {
            "generation": gen,
            "best_fitness": best_fitness,
            "top_3": [(r[0], r[1]) for r in results[:3]],
        }
        history.append(gen_record)
        
        print(f"\n  Best fitness: {best_fitness:.4f}", file=sys.stderr)
        print(f"  Best params: {json.dumps(best_params, indent=2)}", file=sys.stderr)
        
        # Create next generation
        next_pop = []
        
        # Elitism: keep top 2
        for r in results[:2]:
            next_pop.append(r[1])
        
        # Fill rest with mutations of top 4
        top_parents = [r[1] for r in results[:4]]
        while len(next_pop) < args.population:
            p1 = random.choice(top_parents)
            if random.random() < 0.7:
                # Mutate
                child = mutate_params(p1)
            else:
                # Crossover
                p2 = random.choice(top_parents)
                child = crossover_params(p1, p2)
                child = mutate_params(child, rate=0.05)
            next_pop.append(child)
        
        population = next_pop
        
        # Save state
        state = {
            "generation": gen + 1,
            "best_fitness": best_fitness,
            "best_params": best_params,
            "population": population,
            "history": history[-10:],  # keep last 10
        }
        save_path = os.path.join(ROOT, ".kai_darwin_state.json")
        with open(save_path, "w") as f:
            json.dump(state, f, indent=1)
        print(f"  Saved state to {save_path}", file=sys.stderr)
    
    print("\n" + "=" * 60, file=sys.stderr)
    print("Darwin evolution complete!", file=sys.stderr)
    print(f"Final best fitness: {best_fitness:.4f}", file=sys.stderr)
    print(f"Best params:", file=sys.stderr)
    for k, v in best_params.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\nTo apply these params, update kai_bridge.py or add to opencode.json", file=sys.stderr)


if __name__ == "__main__":
    main()
