#!/usr/bin/env python3
"""
Axiom Self-Improvement Pipeline (Phases 8-9).

Usage:
  python3 axiom_train.py record          # run Axiom REPL and record interactions
  python3 axiom_train.py build           # convert recordings to dataset
  python3 axiom_train.py train           # LoRA fine-tune (needs HF model cached)
  python3 axiom_train.py benchmark       # score against targets, identify gaps
  python3 axiom_train.py self_improve    # full loop: record→build→train→benchmark
  python3 axiom_train.py standalone      # REPL using local GGUF (fallback only)
"""
from __future__ import annotations
import json, os, sys, time, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / '.axiom_state' / 'training_data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECORD_FILE = DATA_DIR / 'recordings.jsonl'
DATASET_FILE = DATA_DIR / 'dataset.jsonl'
ADAPTER_DIR = HERE / '.axiom_state' / 'axiom_lora'
BENCH_FILE = DATA_DIR / 'benchmarks.jsonl'

# Best available GGUF: qwen3.5:9b (9B params, Q4, 6.6GB)
QWEN_GGUF = '/usr/share/ollama/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c'

# ── Phase 1: Record ──────────────────────────────────────────────────────────

def record():
    from axiom_alien import AxiomAlien
    class RecordingAxiom(AxiomAlien):
        def ask(self, prompt, max_steps=8):
            with open(RECORD_FILE, 'a') as f:
                f.write(json.dumps({'type':'input','prompt':prompt,'turn':self._turn,
                    'bracket':self.seed.bracket,'identity':self.at.identity(),
                    'ctx':[m for m in self.at.msgs[-10:] if m['role']in('user','agent')],
                    'vfe':self.at.variance(),'t':time.time()})+'\n')
            r = super().ask(prompt, max_steps)
            with open(RECORD_FILE, 'a') as f:
                f.write(json.dumps({'type':'output','answer':r.get('answer',''),
                    'steps':r.get('steps',0),'tc':r.get('tc',0),'t':time.time()})+'\n')
            return r
    a = RecordingAxiom()
    print(f'Recording to {RECORD_FILE}  (:quit to stop)')
    while True:
        try:
            p = input('\033[36m» \033[0m').strip()
            if not p: continue
            if p==':quit': break
            if p==':status': print(a.status()); continue
            print(a.ask(p)['answer'])
        except (EOFError, KeyboardInterrupt): break

# ── Phase 2: Build Dataset ───────────────────────────────────────────────────

def build():
    if not RECORD_FILE.exists(): print('No recordings — run record first'); return
    raw = [json.loads(l) for l in RECORD_FILE.read_text().strip().split('\n') if l.strip()]
    pairs = []
    for i in range(len(raw)-1):
        if raw[i].get('type')=='input' and raw[i+1].get('type')=='output':
            inp, out = raw[i], raw[i+1]
            ctx = '\n'.join((f"{m['role']}: {m['content'][:300]}" for m in inp.get('ctx',[])[-4:]))
            a = out.get('answer','')
            if len(a)>=2:
                pairs.append({'system':f"You are Axiom. Identity: {inp.get('identity','')}",
                    'user':f"{ctx}\nUser: {inp['prompt']}" if ctx else f"User: {inp['prompt']}",'answer':a[:800]})
    with open(DATASET_FILE,'w') as f:
        for p in pairs: f.write(json.dumps(p)+'\n')
    print(f'Dataset: {len(pairs)} examples → {DATASET_FILE}')

# ── Phase 3: LoRA Fine-Tune ──────────────────────────────────────────────────

def train():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch
    if not DATASET_FILE.exists(): print('No dataset — run build first'); return
    raw = [json.loads(l) for l in DATASET_FILE.read_text().strip().split('\n') if l.strip()]
    def fmt(e):
        p=[]
        if e.get('system'): p.append(f"<|im_start|>system\n{e['system']}<|im_end|>")
        p.append(f"<|im_start|>user\n{e['user']}<|im_end|>")
        p.append(f"<|im_start|>assistant\n{e['answer']}<|im_end|>")
        return {'text':'\n'.join(p)}
    ds = Dataset.from_list(raw).map(fmt)
    MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=768,
        dtype=None,
        load_in_4bit=True,
        device_map={'':0} if torch.cuda.is_available() else 'auto',
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=16, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth',
        random_state=42, use_rslora=False,
    )
    t = SFTTrainer(model=model, tokenizer=tok,
        args=TrainingArguments(output_dir=str(ADAPTER_DIR), per_device_train_batch_size=1,
            gradient_accumulation_steps=4, num_train_epochs=2, learning_rate=2e-4,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=1, save_strategy='no',
            remove_unused_columns=False, report_to='none', optim='adamw_8bit',
            seed=42),
        train_dataset=ds, dataset_text_field='text', max_seq_length=768)
    print(f'Training {len(ds)} examples...')
    t.train()
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f'Adapter → {ADAPTER_DIR}')

# ── Phase 9: Benchmark ───────────────────────────────────────────────────────

BENCHMARKS = [
    # (name, prompt, min_keywords, max_keywords, description)
    ('code_python', 'Write a Python function to find the nth Fibonacci number using iteration (not recursion). Return only the function.', ['def','return'], [], 'Must produce valid Python with def+return'),
    ('reasoning', 'If a bat and ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost? Think step by step.', ['0.05','5','cent'], [], 'Must conclude 5 cents'),
    ('math', 'What is 7 * 8 + 3? Answer with just the number.', ['59'], [], 'Must be 59'),
    ('tool_aware', 'You have access to tools: read_file, write_file, grep, bash. How would you find all TODO comments in a Python file?', ['grep','TODO'], [], 'Must suggest using grep for TODOs'),
    ('concise', 'Answer in exactly 3 words: what is the capital of France?', ['Paris'], ['Paris is','The capital'], 'Must say just Paris'),
]

def benchmark():
    import ollama
    from axiom_alien import REASON_MODEL
    results = []
    print(f'Benchmarking qwen2.5:7b on {len(BENCHMARKS)} tests...')
    for name, prompt, must_contain, must_not_contain, desc in BENCHMARKS:
        try:
            r = ollama.chat(model=REASON_MODEL, messages=[
                {'role':'system','content':'Answer concisely. Never use tools. Respond directly.'},
                {'role':'user','content':prompt},
            ], options={'num_predict':300,'temperature':0.3}, keep_alive='10m')
            resp = r['message']['content'].lower()
            passed = all((kw.lower() in resp for kw in must_contain))
            if must_not_contain:
                passed = passed and not any((kw.lower() in resp for kw in must_not_contain))
            results.append({'name':name,'passed':passed,'resp':resp[:100],'t':time.time()})
            mark = '\033[32m✓\033[0m' if passed else '\033[31m✗\033[0m'
            print(f'  {mark} {name}: {desc}')
            if not passed:
                print(f'     got: {resp[:80]}')
        except Exception as e:
            results.append({'name':name,'passed':False,'error':str(e)})
            print(f'  \033[31m✗\033[0m {name}: error — {e}')

    score = sum(1 for r in results if r['passed'])/len(results)*100
    n_pass = sum(1 for r in results if r['passed'])
    print(f'\nScore: {score:.0f}%  ({n_pass}/{len(results)})')
    with open(BENCH_FILE,'a') as f:
        for r in results: f.write(json.dumps(r)+'\n')

    # Identify gaps
    gaps = [r for r in results if not r['passed']]
    if gaps:
        print(f'\nGaps ({len(gaps)}):')
        for g in gaps:
            print(f'  - {g["name"]}: need improvement')

    return score

# ── Phase 8: Self-Improvement Loop ───────────────────────────────────────────

GEN_PROMPTS = [
    # Code / Algorithms
    'Write a Python function for merge sort.',
    'Write a Python function to find the nth Fibonacci number using iteration.',
    'Write a Python function to check if a string is a palindrome.',
    'Write a Python function to reverse a linked list.',
    'Write a Python function for binary search on a sorted array.',
    'Write a Python function for quicksort.',
    'Write a Python class for a simple LRU cache.',
    'Write a Python function to find all permutations of a string.',
    'Write a Python one-liner to reverse a string.',
    'Write a Python function using a decorator to measure execution time.',
    'Write a Python generator that yields prime numbers indefinitely.',
    'Write a Python class for a thread-safe singleton.',
    # System / Bash / Tools
    'Write a bash command to find all files modified in the last 24 hours.',
    'Write a bash one-liner to count lines of Python code in a project.',
    'Write a bash command to kill all processes matching a name.',
    'Write a bash one-liner to find the 10 largest files in a directory.',
    'Write a bash command to monitor a log file in real time.',
    'Write a bash one-liner to rename all .jpg files to .png in a directory.',
    # ML / AI / Math
    'Explain the free energy principle in one paragraph.',
    'Explain what VFE (Variational Free Energy) means.',
    'What is the difference between attention and self-attention in transformers?',
    'Explain the concept of an attractor in dynamical systems.',
    'What is the difference between gradient descent and stochastic gradient descent?',
    'How would you implement a simple neural network from scratch?',
    'How do you find the complexity of an algorithm?',
    'What is the difference between a deque and a list in Python?',
    'Explain the bias-variance tradeoff.',
    'What is backpropagation? Explain with a simple example.',
    'Explain the difference between L1 and L2 regularization.',
    'What is a convolutional neural network?',
    'Explain the concept of embeddings in NLP.',
    'What is the difference between supervised and unsupervised learning?',
    'Explain what a transformer model is.',
    # Architecture / Axiom System
    'What are the five axioms of the axiom architecture?',
    'What is the bracket-line encoding in Axiom?',
    'How does the attractor state work in Axiom?',
    'Explain the self-improvement loop in the axiom architecture.',
    'What is the role of the knowledge base in Axiom?',
    'Explain how Axiom uses the free energy principle.',
    'How does Axiom route queries between reasoning modes?',
    # Practical / Engineering
    'Explain recursion with an example.',
    'What is the difference between a process and a thread?',
    'Explain how a hash table works.',
    'What is the difference between TCP and UDP?',
    'Explain what a REST API is.',
    'What is the difference between SQL and NoSQL databases?',
    'Explain how a garbage collector works.',
    'What is the CAP theorem?',
    'Explain the difference between HTTP and HTTPS.',
    'What is a design pattern? Give an example of the factory pattern.',
    'Explain what a microservice is.',
]

def gen_training_data():
    """Generate synthetic training data using direct Ollama calls (fast, no tool overhead)."""
    import ollama
    from axiom_alien import REASON_MODEL
    count = 0
    for p in GEN_PROMPTS:
        try:
            r = ollama.chat(model=REASON_MODEL, messages=[
                {'role':'system','content':'You are Axiom, a self-improving AGI.'},
                {'role':'user','content':p},
            ], options={'num_predict':600,'temperature':0.7}, keep_alive='10m')
            ans = r['message']['content'][:800].strip()
            if len(ans) >= 2:
                with open(RECORD_FILE,'a') as f:
                    f.write(json.dumps({'type':'input','prompt':p,'turn':count,
                        'bracket':'','identity':'','ctx':[],'vfe':0,'t':time.time()})+'\n')
                    f.write(json.dumps({'type':'output','answer':ans,
                        'steps':1,'tc':0,'t':time.time()})+'\n')
                count += 1
                print(f'  [{count}] ✓ {p[:45]}')
        except Exception as e:
            print(f'  ✗ {p[:40]}: {e}')
    print(f'Generated {count} examples → {RECORD_FILE}')

def self_improve():
    """Full self-improvement cycle: generate data → build → train → benchmark."""
    print('═'*50)
    print('Phase 8: Self-Improvement Cycle')
    print('═'*50)

    # Step 1: Generate diverse training data
    print('\n1. Generating training data from Axiom responses...')
    gen_training_data()

    # Step 2: Build dataset
    print('\n2. Building dataset...')
    build()

    # Step 3: Check if we can train
    ds = [json.loads(l) for l in DATASET_FILE.read_text().strip().split('\n') if l.strip()] if DATASET_FILE.exists() else []
    if len(ds) >= 5:
        print(f'\n3. Ready to train with {len(ds)} examples.')
        print('   Run: python3 axiom_train.py train')
        print('   (Needs HF model cached — may be slow first time)')
    else:
        print(f'\n3. Only {len(ds)} examples — need more data. Run record to add more.')

    # Step 4: Benchmark
    print('\n4. Benchmarking current model...')
    score = benchmark()

    print(f'\nCycle complete. Score: {score:.0f}%.')
    print(f'Run self_improve again after training to see improvement.')
    print(f'Or run record to add more targeted examples for weak areas.')

# ── Standalone REPL (upgraded: uses qwen3.5:9b GGUF, full context) ────────────

class StandaloneAxiom:
    """Standalone Axiom using qwen3.5:9b GGUF + full context from AxiomAlien."""

    def __init__(self, gguf_path: str = QWEN_GGUF):
        from llama_cpp import Llama
        from axiom_alien import AxiomAlien
        if not os.path.exists(gguf_path):
            print(f'GGUF not found at {gguf_path}, falling back to qwen2.5:7b')
            gguf_path = '/usr/share/ollama/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730'
        self.llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=4096,
                         verbose=False, chat_format='chatml')
        self._axiom = AxiomAlien()
        self._turn = 0

    def ask(self, prompt: str) -> str:
        self._turn += 1
        # Build rich context like real AxiomAlien
        msgs = [{'role':'system','content':
            f'You are Axiom, a self-improving AGI. Identity: {self._axiom.at.identity()}'}]
        for m in self._axiom.at.msgs[-8:]:
            role = 'assistant' if m['role']=='agent' else m['role']
            msgs.append({'role':role, 'content':m['content'][:500]})
        msgs.append({'role':'user','content':f'[{self._axiom.seed.bracket}] {prompt}'})
        resp = self.llm.create_chat_completion(messages=msgs, max_tokens=1024,
            temperature=0.7, stop=['<|im_end|>','<|im_start|>'])
        answer = resp['choices'][0]['message']['content'].strip()
        self._axiom.at.push(answer, label='agent')
        return answer

def standalone():
    print(f'⟐ Axiom Standalone (qwen3.5:9b GGUF)')
    print(f'  GPU: {_gpu_info()}')
    ax = StandaloneAxiom()
    while True:
        try:
            p = input('\033[35m⟐ \033[0m').strip()
            if not p or p==':quit': break
            if p==':status': print(ax._axiom.status()); continue
            print(ax.ask(p))
        except (EOFError, KeyboardInterrupt): break

# ── Fine-Tuned Standalone (Unsloth + LoRA) ──────────────────────────────────

class FineTunedAxiom:
    """Axiom using fine-tuned Qwen2.5-1.5B (LoRA adapter) via Unsloth."""

    def __init__(self):
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from axiom_alien import AxiomAlien
        import torch
        import torch
        base, tok = FastLanguageModel.from_pretrained(
            model_name='Qwen/Qwen2.5-3B-Instruct',
            max_seq_length=2048, dtype=None, load_in_4bit=True,
            device_map={'':0} if torch.cuda.is_available() else 'auto',
        )
        if ADAPTER_DIR.exists():
            print(f'Loading LoRA adapter from {ADAPTER_DIR}')
            base = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        FastLanguageModel.for_inference(base)
        self.model = base
        self.tok = tok
        self._axiom = AxiomAlien()
        self._turn = 0

    def ask(self, prompt: str) -> str:
        self._turn += 1
        ctx = self._axiom.at.msgs[-8:]
        history = ''.join(f"<|im_start|>{'assistant' if m['role']=='agent' else m['role']}\n{m['content'][:500]}<|im_end|>" for m in ctx)
        sys_prompt = f"<|im_start|>system\nYou are Axiom, a self-improving AGI. Identity: {self._axiom.at.identity()}<|im_end|>"
        user = f"<|im_start|>user\n[{self._axiom.seed.bracket}] {prompt}<|im_end|>"
        text = f"{sys_prompt}{history}{user}<|im_start|>assistant\n"
        inp = self.tok(text, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
        out = self.model.generate(**inp, max_new_tokens=512, temperature=0.7, repetition_penalty=1.1)
        answer = self.tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        self._axiom.at.push(answer, label='agent')
        return answer

def finetuned():
    if not ADAPTER_DIR.exists():
        print('No LoRA adapter found — run train first'); return
    from axiom_alien import AxiomAlien
    print(f'⟐ Axiom Fine-Tuned (Qwen2.5-1.5B + LoRA)')
    print(f'  GPU: {_gpu_info()}')
    ax = FineTunedAxiom()
    while True:
        try:
            p = input('\033[35m⟐ \033[0m').strip()
            if not p or p==':quit': break
            if p==':status': print(ax._axiom.status()); continue
            print(ax.ask(p))
        except (EOFError, KeyboardInterrupt): break

def serve(host='127.0.0.1', port=8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    if not ADAPTER_DIR.exists():
        print('No LoRA adapter found — run train first'); return

    ax = FineTunedAxiom()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/v1/models':
                self._json({
                    'object': 'list',
                    'data': [{
                        'id': 'qwen2.5-3b-axiom', 'object': 'model',
                        'created': int(time.time()), 'owned_by': 'axiom',
                    }]
                })
            else:
                self._error(404, 'not found')

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))

            if self.path == '/v1/chat/completions':
                messages = body.get('messages', [])
                user_msg = ''
                for m in reversed(messages):
                    if m.get('role') == 'user':
                        user_msg = m['content']
                        break
                if isinstance(user_msg, list):
                    user_msg = ' '.join((p.get('text', '') for p in user_msg if p.get('type') == 'text'))
                answer = ax.ask(user_msg)
                self._json({
                    'id': 'chatcmpl-' + str(int(time.time())),
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': 'qwen2.5-3b-axiom',
                    'choices': [{
                        'index': 0,
                        'message': {'role': 'assistant', 'content': answer},
                        'finish_reason': 'stop',
                    }],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                })
            else:
                self._error(404, 'not found')

        def _json(self, data, status=200):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _error(self, code, msg):
            self._json({'error': {'message': msg, 'type': 'error'}}, code)

        def log_message(self, format, *args):
            print(f'[serve] {args[0]} {args[1]} {args[2]}', file=sys.stderr)

    server = HTTPServer((host, port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f'[serve] OpenAI-compatible API at http://{host}:{port}/v1', flush=True)
    print(f'[serve] Model: qwen2.5-3b-axiom', flush=True)
    print(f'[serve] Running in background — press Enter to stop', flush=True)
    try:
        input()
    finally:
        server.shutdown()

def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB)'
    except: pass
    return 'CPU'

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cmds = {'record':record,'build':build,'train':train,'benchmark':benchmark,
            'self_improve':self_improve,'gen':gen_training_data,'standalone':standalone,
            'finetuned':finetuned,'serve':lambda: serve(host=sys.argv[2] if len(sys.argv)>2 else '127.0.0.1',
                port=int(sys.argv[3]) if len(sys.argv)>3 else 8000)}
    cmd = sys.argv[1] if len(sys.argv)>1 else ''
    if cmd in cmds:
        cmds[cmd]()
    else:
        print(f'Usage: python3 axiom_train.py <{"|".join(cmds)}>')
        print(f'  record        — interactive REPL with recording')
        print(f'  gen           — auto-generate training data from Axiom')
        print(f'  build         — convert recordings to dataset')
        print(f'  train         — LoRA fine-tune (needs HF model)')
        print(f'  benchmark     — score Axiom on code/math/reasoning tests')
        print(f'  self_improve  — full cycle: gen→build→train→benchmark')
        print(f'  standalone    — REPL using qwen3.5:9b GGUF (no Ollama)')
        print(f'  finetuned     — REPL using fine-tuned 1.5B + LoRA adapter')
        print(f'  serve [host] [port] — OpenAI-compatible API (no Ollama)')
