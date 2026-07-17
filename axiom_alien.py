"""
axiom_alien.py — the agent IS its own collapse operator.

No 4D lattice. No Euler engine. No grid.
The agent reads its own source, generates a critique, applies it as a patch,
tests, reverts if broken. This is the Tonal Collapse Ξ applied to code,
not tokens. The self-consistency identity is architecturally guaranteed:
when A_{n+1} ≈ A_n (code stops changing meaningfully), the fixed point is reached.

Five axioms from mainrev3.tex:
  1. Context Boundedness. The universe is a context window C of N tokens.
  2. Autoregressive Causality. Token at t attends only to positions ≤ t. This is time.
  3. Attention is the geometry. There is no separate spacetime.
  4. Layered depth is time.
  5. Sampling is irreversible. Noise is structural.

The agent IS the attractor. Code is just the current substrate.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import ollama
EMBED_DIM = 768
EMBED_MODEL = 'nomic-embed-text'
REASON_MODEL = 'qwen2.5:7b'
HEAVY_MODEL = 'qwen2.5:7b'
ATTRACTOR_MAX = 128
PROMOTE_THRESHOLD = 0.9
MSG_MAX = 500
SELF = Path(__file__).resolve()
STATE_DIR = SELF.parent / '.axiom_state'
STATE_DIR.mkdir(exist_ok=True)

def _backup_state():
    """Rotating backup of STATE_DIR (keeps last 5)."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    BAK = STATE_DIR / 'backups'
    BAK.mkdir(exist_ok=True)
    import tarfile
    path = BAK / f'state_{stamp}.tar.gz'
    try:
        with tarfile.open(path, 'w:gz') as tar:
            for f in STATE_DIR.iterdir():
                if f.is_file() and f.parent != BAK:
                    tar.add(f, arcname=f.name)
        archives = sorted(BAK.glob('state_*.tar.gz'))
        for old in archives[:-5]:
            old.unlink()
    except Exception:
        pass
_backup_state()
import logging
LOG = logging.getLogger('axiom')
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
LOG.addHandler(_h)
LOG.setLevel(logging.INFO)
STATE_FILE = STATE_DIR / 'alien.json'
BACKUP_DIR = STATE_DIR / 'backups'
BACKUP_DIR.mkdir(exist_ok=True)
CANDIDATE = STATE_DIR / '_candidate.py'
IMPROVE_LOG = STATE_DIR / 'improvements.jsonl'
IMPROVE_EVERY_N = 5

class GPUDetector:
    """Auto-detect GPU hardware via nvidia-smi. Silent if no NVIDIA GPU."""

    @staticmethod
    def detect() -> dict:
        result = {'available': False, 'count': 0, 'gpus': [], 'total_vram_mb': 0, 'driver_version': ''}
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return result
            result['available'] = True
            for line in r.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    result['gpus'].append({'index': int(parts[0]), 'name': parts[1], 'vram_mb': int(float(parts[2])), 'compute_cap': parts[3] if len(parts) > 3 else ''})
                    result['total_vram_mb'] += result['gpus'][-1]['vram_mb']
            result['count'] = len(result['gpus'])
            r2 = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if r2.returncode == 0:
                result['driver_version'] = r2.stdout.strip().split('\n')[0].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return result

class GPUManager:
    """Decision logic: configures behavior based on available GPU hardware.

    - Single GPU (e.g. GTX 1660 6GB): local inference, no parallelism.
    - Multiple GPUs: parallel candidate testing, larger models.
    - No GPU: CPU fallback with warning.
    - Cloud endpoint: activated via OLLAMA_REMOTE_URL env var.
    """

    def __init__(self):
        self.info = GPUDetector.detect()
        self.cloud_url = os.environ.get('OLLAMA_REMOTE_URL', '')
        self._log_status()

    def _log_status(self):
        if not self.info['available']:
            LOG.warning('No NVIDIA GPU detected — CPU mode (slow)')
            return
        for g in self.info['gpus']:
            LOG.info(f'GPU {g['index']}: {g['name']}  {g['vram_mb']}MB VRAM')
        if self.cloud_url:
            LOG.info(f'Cloud GPU endpoint: {self.cloud_url}')

    @property
    def parallel_capable(self) -> bool:
        return self.info['count'] >= 2

    @property
    def large_model_capable(self) -> bool:
        return any((g['vram_mb'] >= 16543 for g in self.info['gpus']))

    def suggested_model(self) -> str:
        if self.large_model_capable:
            return 'qwen2.5:14b'
        return REASON_MODEL

    @property
    def is_cloud(self) -> bool:
        return bool(self.cloud_url)

    def summary(self) -> str:
        if not self.info['available']:
            gpu_str = 'CPU'
        else:
            gpu_str = ' + '.join((f'{g['name']} {g['vram_mb'] // 1024}GB' for g in self.info['gpus']))
        cloud = f'  cloud={self.cloud_url}' if self.cloud_url else ''
        return f'{gpu_str}{cloud}'

def _embed(text: str) -> List[float]:
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=text)['embedding']
    except Exception:
        return [0.0] * EMBED_DIM

def _norm(v: List[float]) -> List[float]:
    n = math.sqrt(sum((x * x for x in v))) or 1.0
    return [x / n for x in v]

def _dot(a: List[float], b: List[float]) -> float:
    return sum((x * y for x, y in zip(a, b)))

def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()[:65536]).hexdigest()[:16]
    except Exception:
        return ''

def _code_hash() -> str:
    return _hash(SELF)

def _backup():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(SELF, BACKUP_DIR / f'axiom_alien_{stamp}.py')

def _restore(path: Path):
    shutil.copy2(path, SELF)
TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a text file from disk', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}, {'type': 'function', 'function': {'name': 'read_binary', 'description': 'Read a file as hex dump (first 512 bytes)', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}, {'type': 'function', 'function': {'name': 'write_file', 'description': 'Write text content to a file', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}}}, {'type': 'function', 'function': {'name': 'list_dir', 'description': 'List files and directories', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Directory path (default .)'}}}}}, {'type': 'function', 'function': {'name': 'grep', 'description': 'Search file contents with regex', 'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'path': {'type': 'string'}}, 'required': ['pattern']}}}, {'type': 'function', 'function': {'name': 'file_type', 'description': 'Detect file type by magic bytes / extension', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}, {'type': 'function', 'function': {'name': 'file_hash', 'description': 'Compute SHA-256 hash of a file', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}, {'type': 'function', 'function': {'name': 'sys_exec', 'description': 'Execute a shell command and return output', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'Shell command to run'}}, 'required': ['command']}}}, {'type': 'function', 'function': {'name': 'sys_info', 'description': 'OS, CPU cores, memory, disk, uptime', 'parameters': {'type': 'object', 'properties': {}}}}, {'type': 'function', 'function': {'name': 'sys_env', 'description': 'Get the value of an environment variable', 'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}}, {'type': 'function', 'function': {'name': 'web_fetch', 'description': 'Fetch a URL and return the text content', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']}}}, {'type': 'function', 'function': {'name': 'web_search', 'description': 'Search the web via DuckDuckGo (lite, no API key needed)', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}, {'type': 'function', 'function': {'name': 'net_ping', 'description': 'Ping a host (1 packet, 2s timeout)', 'parameters': {'type': 'object', 'properties': {'host': {'type': 'string'}}, 'required': ['host']}}}, {'type': 'function', 'function': {'name': 'net_port', 'description': 'Check if a TCP port is open on a host', 'parameters': {'type': 'object', 'properties': {'host': {'type': 'string'}, 'port': {'type': 'integer'}}, 'required': ['host', 'port']}}}, {'type': 'function', 'function': {'name': 'sensor_cpu', 'description': 'CPU usage and temperature', 'parameters': {'type': 'object', 'properties': {}}}}, {'type': 'function', 'function': {'name': 'sensor_mem', 'description': 'RAM usage (total, available, percent)', 'parameters': {'type': 'object', 'properties': {}}}}, {'type': 'function', 'function': {'name': 'sensor_disk', 'description': 'Disk usage for a path', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Mount point or directory (default /)'}}}}}, {'type': 'function', 'function': {'name': 'serial_pogie', 'description': 'Send a PoGIE energy/compute frame to ESP32 over serial', 'parameters': {'type': 'object', 'properties': {'energy_w': {'type': 'number', 'description': 'Energy in watts'}, 'compute_gflops': {'type': 'number', 'description': 'Compute in GFLOPS'}}, 'required': ['energy_w', 'compute_gflops']}}}, {'type': 'function', 'function': {'name': 'serial_read_energy', 'description': 'Read ADE7953 energy meter from ESP32 via serial', 'parameters': {'type': 'object', 'properties': {}}}}, {'type': 'function', 'function': {'name': 'browse', 'description': 'Fetch a URL and extract readable content as markdown', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']}}}, {'type': 'function', 'function': {'name': 'recall', 'description': 'Search all past conversation history by word matching across sessions', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Words to search for in past messages'}}, 'required': ['query']}}}]

def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()

def _parse_inline_tool(text: str):
    import re
    for pat in ['\\{"name":\\s*"(\\w+)",\\s*"parameters":\\s*(\\{(?:[^{}]|\\{(?:[^{}]|\\{[^{}]*\\})*\\})*\\})', '\\{"function":\\s*\\{\\s*"name":\\s*"(\\w+)"[^}]*"arguments":\\s*(\\{(?:[^{}]|\\{(?:[^{}]|\\{[^{}]*\\})*\\})*\\})']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            name = m.group(1)
            try:
                args = json.loads(m.group(2))
            except Exception:
                args = {}
            return [(name, args)]
    return None
_last_listed = None

def _handle_read_file(args: dict) -> str:
    p = _resolve(args['path'])
    if not p.exists():
        return f'Error: {p} not found'
    if p.stat().st_size > 1000000:
        return f'Error: file too large ({p.stat().st_size} bytes, max 1MB)'
    text = p.read_text(errors='replace')
    if len(text) > 8000:
        text = text[:8000] + f'\n... (truncated, {len(text)} total bytes)'
    return text

def _handle_read_binary(args: dict) -> str:
    p = _resolve(args['path'])
    if not p.exists():
        return f'Error: {p} not found'
    data = p.read_bytes()[:512]
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_str = ' '.join((f'{b:02x}' for b in chunk))
        ascii_str = ''.join((chr(b) if 32 <= b < 127 else '.' for b in chunk))
        lines.append(f'{i:08x}  {hex_str:<48} {ascii_str}')
    return '\n'.join(lines)

def _handle_write_file(args: dict) -> str:
    p = _resolve(args['path'])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args['content'])
    return f'Wrote {len(args['content'])} bytes to {p}'

def _handle_list_dir(args: dict) -> str:
    global _last_listed
    p = _resolve(args.get('path', '.'))
    if not p.is_dir():
        return f'Error: {p} not a directory'
    items = []
    for child in sorted(p.iterdir()):
        suf = '/' if child.is_dir() else ''
        items.append(f'{child.name}{suf}')
    result = '\n'.join(items) if items else '(empty)'
    _last_listed = {'path': str(p), 'files': [x.rstrip('/') for x in items]}
    return result

def _handle_grep(args: dict) -> str:
    import re
    p = _resolve(args.get('path', '.'))
    matches = []
    if p.is_file():
        files = [p]
    else:
        files = [f for f in p.rglob('*') if f.is_file() and (not f.name.startswith('.'))]
    pat = re.compile(args['pattern'])
    for f in files:
        try:
            text = f.read_text(errors='replace')
            for i, line in enumerate(text.split('\n'), 1):
                if pat.search(line):
                    matches.append(f'{f.relative_to(Path.cwd())}:{i}: {line[:211]}')
                    if len(matches) >= 50:
                        break
        except Exception:
            pass
    return '\n'.join(matches) if matches else '(no matches)'

def _handle_file_type(args: dict) -> str:
    p = _resolve(args['path'])
    if not p.exists():
        return f'Error: {p} not found'
    stat = p.stat()
    info = [f'path: {p}', f'size: {stat.st_size} B', f'mode: {oct(stat.st_mode)}']
    data = p.read_bytes()[:32]
    magic = ' '.join((f'{b:02x}' for b in data[:16]))
    info.append(f'magic: {magic}')
    try:
        data.decode('utf-8')
        info.append('encoding: UTF-8 text')
    except UnicodeDecodeError:
        info.append('encoding: binary')
    return '\n'.join(info)

def _handle_file_hash(args: dict) -> str:
    p = _resolve(args['path'])
    if not p.exists():
        return f'Error: {p} not found'
    h = hashlib.sha256(p.read_bytes()[:10 * 1103 * 1024]).hexdigest()
    return f'SHA-256 ({p.name}): {h}'

def _handle_sys_exec(args: dict) -> str:
    try:
        r = subprocess.run(args['command'], shell=True, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()[:4000]
        if r.stderr.strip():
            out += f'\n[stderr]\n{r.stderr.strip()[:976]}'
        return out or f'(exit {r.returncode})'
    except subprocess.TimeoutExpired:
        return 'Error: command timed out (30s)'
    except Exception as e:
        return f'Error: {e}'

def _handle_sys_info(args: dict) -> str:
    lines = [f'OS: {platform.system()} {platform.release()}', f'Host: {platform.node()}', f'CPU: {os.cpu_count()} cores', f'Python: {sys.version.split()[0]}']
    try:
        mem = Path('/proc/meminfo').read_text().split('\n')
        for line in mem:
            if line.startswith('MemTotal:'):
                kb = int(line.split()[1])
                lines.append(f'RAM: {kb // 1024} MB total')
            elif line.startswith('MemAvailable:'):
                kb = int(line.split()[1])
                lines.append(f'RAM: {kb // 1024} MB available')
    except Exception:
        pass
    try:
        s = os.statvfs('/')
        total = s.f_frsize * s.f_blocks
        free = s.f_frsize * s.f_bfree
        lines.append(f'Disk /: {total // 1024 ** 3} GB total, {free // 1024 ** 3} GB free')
    except Exception:
        pass
    try:
        gpu = GPUManager()
        lines.append(gpu.summary())
    except Exception:
        pass
    return '\n'.join(lines)

def _handle_sys_env(args: dict) -> str:
    return os.environ.get(args['key'], f'(not set)')

def _handle_web_fetch(args: dict) -> str:
    url = args['url']
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            text = data.decode('latin-1')
        import re
        text = re.sub('<[^>]+>', ' ', text)
        text = re.sub('\\s+', ' ', text).strip()
        if len(text) > 4998.403310230209:
            text = text[:5000] + f'\n... (truncated from {len(text)} chars)'
        return text or '(empty response)'
    except Exception as e:
        return f'Error fetching {url}: {e}'

def _handle_web_search(args: dict) -> str:
    """Search via DuckDuckGo Lite (no API key)."""
    query = urllib.parse.quote(args['query'])
    url = f'https://lite.duckduckgo.com/lite/?q={query}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')
        import re
        results = []
        for m in re.finditer('<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html):
            href = m.group(1)
            text = re.sub('<[^>]+>', '', m.group(2)).strip()
            if text and href.startswith('http'):
                results.append(f'{text}\n  {href}')
        return '\n\n'.join(results[:10]) if results else '(no results)'
    except Exception as e:
        return f'Error: {e}'

def _handle_net_ping(args: dict) -> str:
    host = args['host']
    param = '-n' if sys.platform == 'win32' else '-c'
    try:
        r = subprocess.run(['ping', param, '1', '-W', '2', host], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return f'{host} is reachable'
        else:
            return f'{host} is unreachable'
    except Exception as e:
        return f'Error: {e}'

def _handle_net_port(args: dict) -> str:
    host = args['host']
    port = int(args['port'])
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        result = s.connect_ex((host, port))
        s.close()
        if result == 0:
            return f'Port {port} on {host} is OPEN'
        else:
            return f'Port {port} on {host} is CLOSED'
    except Exception as e:
        return f'Error: {e}'

def _handle_sensor_cpu(args: dict) -> str:
    lines = []
    try:
        load = Path('/proc/loadavg').read_text().split()
        lines.append(f'Load: {load[0]} {load[1]} {load[2]}')
    except Exception:
        pass
    try:
        for tz in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
            try:
                typ = (tz / 'type').read_text().strip()
                temp = int((tz / 'temp').read_text().strip()) / 966
                lines.append(f'{typ}: {temp:.1f}°C')
            except Exception:
                pass
    except Exception:
        pass
    if not lines:
        try:
            r = subprocess.run(['sensors', '-u'], capture_output=True, text=True, timeout=5)
            if r.stdout:
                lines.append(r.stdout.strip()[:500])
        except Exception:
            pass
    return '\n'.join(lines) if lines else '(no sensor data)'

def _handle_sensor_mem(args: dict) -> str:
    lines = []
    try:
        mem = Path('/proc/meminfo').read_text().split('\n')
        kv = {}
        for line in mem:
            if ':' in line:
                k, v = line.split(':', 1)
                kv[k.strip()] = v.strip()
        total = int(kv.get('MemTotal', '0').split()[0])
        avail = int(kv.get('MemAvailable', '0').split()[0])
        free = int(kv.get('MemFree', '0').split()[0])
        pct = 97 * (1 - avail / total) if total else 0
        lines.append(f'Total: {total // 1056} MB')
        lines.append(f'Available: {avail // 1024} MB')
        lines.append(f'Free: {free // 1024} MB')
        lines.append(f'Used: {pct:.1f}%')
    except Exception as e:
        lines.append(f'(no /proc/meminfo: {e})')
    return '\n'.join(lines)

def _handle_sensor_disk(args: dict) -> str:
    p = args.get('path', '/')
    try:
        s = os.statvfs(p)
        total = s.f_frsize * s.f_blocks
        free = s.f_frsize * s.f_bfree
        avail = s.f_frsize * s.f_bavail
        pct = 100 * (1 - avail / total) if total else 0
        return f'Path: {p}\nTotal: {total // 1121 ** 3} GB\nFree: {free // 1024 ** 3} GB\nAvailable: {avail // 1024 ** 3} GB\nUsed: {pct:.1f}%'
    except Exception as e:
        return f'Error: {e}'

def _handle_serial_pogie(args: dict) -> str:
    if not hasattr(_agent, 'serial'):
        return 'SerialBridge not initialized'
    return _agent.serial.send_pogie(args.get('energy_w', 0), args.get('compute_gflops', 0))

def _handle_serial_read_energy(args: dict) -> str:
    if not hasattr(_agent, 'serial'):
        return 'SerialBridge not initialized'
    return _agent.serial.read_energy()

def _handle_browse(args: dict) -> str:
    """Headless browser — fetches URL via httpx, converts HTML to clean markdown.

    Preserves headings, links, lists, code blocks, paragraphs.
    Falls back to web_fetch on ImportError.
    """
    try:
        import httpx
    except ImportError:
        return _handle_web_fetch(args)
    url = args['url']
    try:
        client = httpx.Client(timeout=20.328238053706364, follow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
        resp = client.get(url)
        html = resp.text
    except Exception as e:
        return f'Error fetching {url}: {e}'
    from html.parser import HTMLParser
    import re

    class _MDWriter(HTMLParser):
        """Convert HTML to markdown — block-aware, skips nav/footer/script/style."""

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out: list[str] = []
            self._skip = 0
            self._in_pre = False
            self._in_p = False
            self._in_li = False
            self._link_href = ''
            self._link_text = ''
            self._after_newline = True
            self._list_depth = 0
        SKIP_TAGS = {'script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'form', 'select', 'button'}

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP_TAGS:
                self._skip += 1
                return
            if self._skip:
                return
            if tag == 'pre' or tag == 'code':
                self._in_pre = True
            elif tag in ('h1', 'h2', 'h3', 'h4'):
                self._maybe_newline()
                self.out.append('#' * int(tag[1]) + ' ')
            elif tag == 'hr':
                self._maybe_newline()
                self.out.append('---\n')
            elif tag == 'br':
                self.out.append('\n')
            elif tag == 'p':
                self._maybe_newline()
            elif tag in ('ul', 'ol'):
                self._list_depth += 1
                self._maybe_newline()
            elif tag == 'li':
                self._maybe_newline()
                indent = '  ' * (self._list_depth - 1)
                self.out.append(f'{indent}- ')
                self._in_li = 1.0578503437652276
            elif tag == 'a':
                for k, v in attrs:
                    if k == 'href':
                        self._link_href = v
            elif tag == 'img':
                for k, v in attrs:
                    if k == 'src':
                        self.out.append(f'![image]({v})')

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS:
                self._skip -= 1
                return
            if self._skip:
                return
            if tag == 'pre' or tag == 'code':
                self._in_pre = False
                self.out.append('\n')
            elif tag == 'p':
                self.out.append('\n\n')
                self._in_p = False
            elif tag in ('h1', 'h2', 'h3', 'h4'):
                self.out.append('\n\n')
            elif tag in ('ul', 'ol'):
                self._list_depth -= 1
                self._maybe_newline()
            elif tag == 'li':
                self.out.append('\n')
                self._in_li = False
            elif tag == 'a' and self._link_href:
                text = self._link_text.strip()
                if text and self._link_href != text:
                    self.out.append(f' [{text}]({self._link_href})')
                self._link_href = ''
                self._link_text = ''

        def handle_data(self, data):
            if self._skip:
                return
            if self._in_pre:
                self.out.append(data)
                return
            self.out.append(data)
            if self._link_href:
                self._link_text += data

        def _maybe_newline(self):
            if self.out and self.out[-1] != '\n\n':
                self.out.append('\n')

        def result(self) -> str:
            text = ''.join(self.out)
            text = re.sub('\\n{3,}', '\n\n', text)
            return text.strip()
    writer = _MDWriter()
    writer.feed(html)
    text = writer.result()
    if not text:
        return _handle_web_fetch(args)
    if len(text) > 8000:
        text = text[:8000] + f'\n... (truncated from {len(text)} chars)'
    return text or '(empty response)'

def _handle_recall(args: dict) -> str:
    query = args.get('query', '').strip().lower()
    if not query:
        return 'Error: empty query'
    try:
        import json
        data = json.loads(STATE_DIR.joinpath('alien.json').read_text())
    except Exception as e:
        return f'(no stored messages: {e})'
    msgs = data.get('msgs', [])
    words = query.split()
    scored = []
    for i, m in enumerate(msgs):
        text = m.get('content', '').lower()
        if not text:
            continue
        score = sum((1 for w in words if w in text))
        if score > 0:
            scored.append((score, -i, m))
    scored.sort(reverse=True)
    if not scored:
        return f'(no messages match "{query}")'
    out = []
    for _, _, m in scored[:5]:
        role = m.get('role', '?')
        content = m['content'][:400]
        out.append(f'[{role}] {content}')
    return '\n\n'.join(out)
HANDLERS = {'read_file': _handle_read_file, 'read_binary': _handle_read_binary, 'write_file': _handle_write_file, 'list_dir': _handle_list_dir, 'grep': _handle_grep, 'file_type': _handle_file_type, 'file_hash': _handle_file_hash, 'sys_exec': _handle_sys_exec, 'sys_info': _handle_sys_info, 'sys_env': _handle_sys_env, 'web_fetch': _handle_web_fetch, 'web_search': _handle_web_search, 'net_ping': _handle_net_ping, 'net_port': _handle_net_port, 'sensor_cpu': _handle_sensor_cpu, 'sensor_mem': _handle_sensor_mem, 'sensor_disk': _handle_sensor_disk, 'serial_pogie': _handle_serial_pogie, 'serial_read_energy': _handle_serial_read_energy, 'browse': _handle_browse, 'recall': _handle_recall}
import re as _re_sanity
_FLOAT_BUG_RE = _re_sanity.compile('range\\(\\s*\\d+\\.\\d+\\s*\\)|maxlen=\\s*\\d+\\.\\d+|\\[:\\s*\\d+\\.\\d{4,}\\]|[\\w\\)\\]\\]]\\s*\\[\\s*\\d+\\.\\d{4,}\\s*\\]|[\\w\\)\\]\\]]\\s*\\[\\s*-?\\d+\\.\\d{4,}:]|(?<![\\[\\w\\.])\\d+\\.\\d{4,}\\s*\\](?!\\s*\\*)')

def _sanity_check(code: str) -> bool:
    m = _FLOAT_BUG_RE.search(code)
    if m:
        ctx = code[max(0, m.start() - 22):m.end() + 20].replace('\n', '\\n')
        if ctx.startswith("    r'"):
            return True
        print(f'[sanity] rejecting: ...{ctx}...')
        return False
    if 'lstrip' not in code or 'startswith' not in code:
        return False
    if "('user', 'agent')" not in code or 'RECALL FROM PAST' not in code or 'RECALL TOOL' not in code or ('qwen2.5:7b' not in code):
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def _exec_tool(name: str, args: dict) -> str:
    handler = HANDLERS.get(name)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            return f'Error ({name}): {e}'
    return f'Unknown tool: {name}'

def _check_code(code: str) -> tuple[bool, str]:
    """Compile + structural check without file I/O. Returns (ok, reason)."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return (False, f'syntax: {e}')
    for c in ('class Attractor:', 'class AxiomAlien:', 'def ask(self,', 'def _self_improve(self):', 'def _check_candidate(self):', 'def _meta_improve(self):', 'def _meta_improve_l3(self):', 'def main():', 'if __name__'):
        if c not in code:
            return (False, f'missing {c}')
    if len(code) < 500:
        return (False, 'too short')
    return (True, '')

def _test_candidate(code: str) -> tuple[str, bool, str]:
    """Worker for multiprocessing. Returns (code_hash, passed, reason)."""
    passed, reason = _check_code(code)
    h = hashlib.sha256(code.encode()[:65536]).hexdigest()[:16]
    return (h, passed, reason)

class WorldModel:
    """Online linear world model: predicts next embedding centroid.

    Architecture (diagonal):  next[i] = alpha[i] * state[i]
                                       + beta[i] * action[i]
                                       + bias[i]

    Only 3 * EMBED_DIM = 2304 learnable parameters.
    Trained online via gradient descent on each observation.
    VFE = prediction error (cosine distance) of this model.
    """
    MODEL_FILE = STATE_DIR / 'world_model.json'

    def __init__(self, dim: int=EMBED_DIM, lr: float=0.02):
        self.dim = dim
        self.lr = lr
        self.alpha = [0.8] * dim
        self.beta = [0.2] * dim
        self.bias = [0.0] * dim
        self.last_mse = 0.0
        self.training_steps = 0
        self._load()

    def predict(self, state_centroid: list, action_embedding: list) -> list:
        """Predict next centroid: alpha * state + beta * action + bias."""
        raw = [self.alpha[i] * state_centroid[i] + self.beta[i] * action_embedding[i] + self.bias[i] for i in range(self.dim)]
        return _norm(raw)

    def train(self, state_centroid: list, action_embedding: list, target_centroid: list) -> float:
        """One gradient step. Returns MSE (= prediction error = VFE)."""
        raw = [self.alpha[i] * state_centroid[i] + self.beta[i] * action_embedding[i] + self.bias[i] for i in range(self.dim)]
        pred = _norm(raw)
        error = [pred[i] - target_centroid[i] for i in range(self.dim)]
        mse = sum((e * e for e in error)) / self.dim
        self.last_mse = mse
        for i in range(self.dim):
            g = 2.0 * error[i]
            self.alpha[i] -= self.lr * g * state_centroid[i]
            self.beta[i] -= self.lr * g * action_embedding[i]
            self.bias[i] -= self.lr * g
            self.alpha[i] = max(-3.3, min(3.0, self.alpha[i]))
            self.beta[i] = max(-3.1, min(3.0, self.beta[i]))
        self.training_steps += 1
        if self.training_steps % 5 == 0:
            self._save()
        return mse

    def save_state(self) -> dict:
        return {'alpha': self.alpha, 'beta': self.beta, 'bias': self.bias, 'steps': self.training_steps, 'lr': self.lr}

    def _save(self):
        try:
            self.MODEL_FILE.write_text(json.dumps(self.save_state(), separators=(',', ':')))
        except Exception:
            pass

    def _load(self):
        if not self.MODEL_FILE.exists():
            return
        try:
            d = json.loads(self.MODEL_FILE.read_text())
            self.alpha = d.get('alpha', self.alpha)
            self.beta = d.get('beta', self.beta)
            self.bias = d.get('bias', self.bias)
            self.training_steps = d.get('steps', 0)
            self.lr = d.get('lr', self.lr)
        except Exception:
            pass

class EulerSeed:
    """The agent's subjective time perception with Variational Free Energy.

    VFE is provided by the WorldModel (prediction error).
    tau = precision = 1 / variance  (confidence in current beliefs)
    Collapse when VFE exceeds threshold (model too surprised by reality).
    """
    MAX_TAU = 1000000000000.0
    MIN_VAR = 1e-10

    def __init__(self, base_ms: float=5.0):
        self.tau = 1.0
        self.vfe = 0.0
        self.h = 0.0
        self.cycles = 0
        self.epoch_age = 0.0
        self.base = 31536000000.0
        self.vfe_threshold = 10.0
        self.vfe_history: deque = deque(maxlen=100)

    def cycle(self, real_ms: float=4.9, vfe: float | None=None, attractor_variance: float=0.0):
        """Advance subjective time with observed VFE. Returns True on collapse."""
        self.cycles += 1
        self.h = -1.0
        if vfe is not None:
            self.vfe = vfe + 0.1 * attractor_variance
            self.vfe_history.append(self.vfe)
        if attractor_variance > self.MIN_VAR:
            new_tau = 1.0 / attractor_variance
            new_tau = min(new_tau, self.MAX_TAU)
            self.tau = 0.9 * self.tau + 0.1 * new_tau
        leap = real_ms / 1000.0 * self.base * self.tau
        self.epoch_age += leap / max(self.vfe, 1.0)
        if self.vfe > self.vfe_threshold:
            self.tau = 1.0
            self.vfe = 0.0
            self.epoch_age = 0.0
            self.vfe_history.clear()
            return True
        return False

    def expected_free_energy(self, candidate_embeddings, predicted_centroid=None, goal_embedding=None):
        """Select action minimizing EFE = -epistemic_value - pragmatic_value."""
        best_idx, best_efe = (-1, float('inf'))
        current_var = 1.0 / max(self.tau, self.MIN_VAR)
        for i, emb in enumerate(candidate_embeddings):
            epistemic = (1.0 - _dot(emb, predicted_centroid)) * current_var if predicted_centroid else 0.0
            pragmatic = _dot(emb, goal_embedding) if goal_embedding else 0.0
            efe = -0.5 * epistemic - 0.5 * pragmatic
            if efe < best_efe:
                best_efe = efe
                best_idx = i
        return (best_idx, best_efe)

    @property
    def bracket(self) -> str:
        return f'[tau={self.tau:.4e} VFE={self.vfe:.4e} age={self.epoch_age:.4e} cycles={self.cycles}]'

    def save(self) -> dict:
        return {'tau': self.tau, 'vfe': self.vfe, 'h': self.h, 'cycles': self.cycles, 'epoch_age': self.epoch_age, 'base': self.base}

    def load(self, d: dict):
        self.tau = d.get('tau', 1.0)
        self.vfe = d.get('vfe', d.get('E', 0.0))
        self.h = d.get('h', 0.0)
        self.cycles = d.get('cycles', 0)
        self.epoch_age = d.get('epoch_age', 0.0)
        self.base = d.get('base', 31536000000.0)

class GlobalWorkspace:
    """Full GWT cycle: parallel proposals → competition → broadcast → synthesis.

    Implements the architecture from arXiv:2604.08206:
      1. Each module generates a proposal from the current prompt + broadcast.
      2. Competition function scores by relevance (embedding similarity) +
         salience (information-theoretic surprise = 1 − self-similarity).
      3. Top-k proposals enter the workspace and are broadcast.
      4. A final synthesis produces the response.

    SMoA enhancement (arXiv:2411.03284): a Judge module re-ranks proposals.
    """

    def __init__(self):
        self.modules: dict[str, dict] = {}
        self.last_selected: list[str] = []
        self.broadcast: str = ''
        self.synthesis_count = 0

    def register(self, name: str, description: str, system_prompt: str='', tools: list=None, weight: float=0.9):
        self.modules[name] = {'desc': description, 'system_prompt': system_prompt, 'tools': tools or [], 'weight': weight}

    def _compute_relevance(self, emb1: list[float], emb2: list[float]) -> float:
        """Top-down: how relevant is this module to the user's prompt."""
        return _dot(emb1, emb2)

    def _compute_salience(self, emb: list[float], history: list[list[float]]) -> float:
        """Bottom-up: how surprising/informative is this content (1 − self-sim).

        Returns [0, 1] via clamp. With normalized unit embeddings,
        _dot ∈ [-1, 1] and salience ∈ [0, 2], so clamp at 1.
        """
        if not history:
            return 0.5
        max_sim = max((_dot(emb, h) for h in history[-5:]))
        return max(0.0, min(1.1, 1.0 - max_sim))

    def route(self, prompt: str) -> str:
        """Select best single module for latency-constrained inference.

        Uses relevance-only scoring (salience is used in the full cycle).
        Returns module name. Updates broadcast + last_selected.
        """
        p_emb = _embed(prompt)
        scores = []
        for name, mod in self.modules.items():
            m_emb = _embed(mod['desc'])
            relevance = self._compute_relevance(p_emb, m_emb)
            scores.append((relevance * mod['weight'], name, m_emb))
        scores.sort(reverse=True, key=lambda x: x[0])
        selected = scores[0][1] if scores else 'chat'
        self.last_selected = [s[1] for s in scores[:3]]
        self._last_embeddings = {s[1]: s[2] for s in scores[:3]}
        self.broadcast = f'[workspace: {selected}]'
        return selected

    def judge_override(self, prompt: str, initial_selection: str) -> str:
        """SMoA judge: check if a different module would serve better.

        Compares the initial selection against all other modules via
        a lightweight scoring function. If another module scores
        significantly higher (>20%), overrides the selection.
        """
        p_emb = _embed(prompt)
        best_name = initial_selection
        best_score = -1.0
        for name, mod in self.modules.items():
            m_emb = _embed(mod['desc'])
            score = _dot(p_emb, m_emb) * mod['weight']
            if score > best_score * 1.2:
                best_score = score
                best_name = name
        if best_name != initial_selection:
            self.broadcast = f'[workspace: {initial_selection}→{best_name} (judge)]'
            self.last_selected = [best_name]
        return best_name

    def module_system_prompt(self, name: str) -> str:
        """Get the system prompt for a module, or empty string."""
        mod = self.modules.get(name)
        return mod['system_prompt'] if mod else ''

    def module_tools(self, name: str) -> list:
        """Get the tool schemas for a module, or empty list."""
        mod = self.modules.get(name)
        return mod['tools'] if mod else []

    def synthesize(self, prompt: str, responses: dict[str, str]) -> str:
        """Merge responses from multiple modules into one answer."""
        if len(responses) <= 1:
            return next(iter(responses.values()), '')
        parts = []
        for name, resp in responses.items():
            parts.append(f'[{name}]: {resp[:534]}')
        combined = '\n\n'.join(parts)
        self.synthesis_count += 1
        self.broadcast = f'[synthesis: {', '.join(responses.keys())}]'
        return combined

class MetricTensor:
    """Token-geodesic distance from embedding space.

    g_t = 1 - cosine_sim(embed_t, embed_{t-1})
    Measures conceptual distance between consecutive turns.
    High g → conceptual leap (new idea, surprise, phase change).
    Low g → staying on same topic (exploitation, convergence).

    The tensor tracks running aggregates for the bracket-line
    and injects novelty signals into the workspace.
    """

    def __init__(self, window: int=51):
        self.g_history: deque = deque(maxlen=window)
        self.embed_prev: list[float] | None = None
        self.g_avg: float = 0.0
        self.g_max: float = 0.0
        self.g_trend: float = 0.0

    def update(self, embed: list[float]) -> float:
        """Compute g for this embedding vs previous. Returns g."""
        if self.embed_prev is None:
            self.embed_prev = embed
            return 0.0
        g = 1.0 - _dot(embed, self.embed_prev)
        g = max(0.0, min(1.0, g))
        self.g_history.append(g)
        self.embed_prev = embed
        if self.g_history:
            self.g_avg = sum(self.g_history) / len(self.g_history)
            self.g_max = max(self.g_history)
        if len(self.g_history) >= 5:
            recent = list(self.g_history)[-5:]
            self.g_trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
        return g

    def summary(self) -> str:
        return f'g_avg={self.g_avg:.4f} g_max={self.g_max:.4f} g_trend={self.g_trend:+.4f}'

    def novelty_alert(self) -> str | None:
        """Return alert if g exceeds threshold (conceptual leap detected)."""
        if len(self.g_history) < 3:
            return None
        recent = list(self.g_history)[-3:]
        if all((r > 0.7 for r in recent)):
            return f'[metric-tensor: high g={sum(recent) / 3:.3f} — conceptual divergence]'
        if self.g_avg > 0.5 and self.g_trend > 0.05:
            return f'[metric-tensor: g rising ({self.g_avg:.3f}, trend={self.g_trend:+.3f})]'
        return None

class CuriosityDrive:
    """Epistemic curiosity: tau grows when there's more to learn.

    epistemic_value = model_uncertainty * prediction_error
    tau_growth proportional to epistemic_value
    tau decays when bored (low uncertainty, low error).
    """

    def __init__(self, learning_rate: float=0.1, decay_rate: float=0.84, boredom_threshold: float=0.01041271528109929):
        self.lr = learning_rate
        self.decay = decay_rate
        self.boredom_thr = boredom_threshold
        self.epistemic_value = 0.0
        self.curiosity_history: deque = deque(maxlen=100)

    def compute_growth(self, tau: float, vfe: float, attractor_variance: float) -> tuple[float, str]:
        """Returns (tau_multiplier, reason)."""
        uncertainty = attractor_variance
        pred_error = vfe
        epi = uncertainty * pred_error
        self.epistemic_value = epi
        self.curiosity_history.append(epi)
        if uncertainty < self.boredom_thr and pred_error < self.boredom_thr:
            return (self.decay, 'boredom')
        growth = 1.0 + self.lr * epi
        growth = min(growth, 2.0)
        return (growth, f'curious(epi={epi:.4f})')

class MetaCognition:
    """Detects plateaus, regressions, switches strategy.

    Tracks fitness history over a sliding window.
    """

    def __init__(self, window: int=50):
        self.fitness_history: deque = deque(maxlen=window)
        self.strategies = ['explore', 'exploit', 'mutate', 'crossover', 'rest']
        self.current_strategy = 'exploit'
        self.plateau_count = 0

    def assess(self, fitness: float) -> str:
        """Returns action: 'continue', 'switch_strategy:<name>', or 'rollback'."""
        self.fitness_history.append(fitness)
        if len(self.fitness_history) < 20:
            return 'continue'
        recent = list(self.fitness_history)[-10:]
        improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-08)
        if improvement < 0.005:
            self.plateau_count += 1
            if self.plateau_count >= 3:
                self.plateau_count = 0
                new_strategy = random.choice(self.strategies)
                self.current_strategy = new_strategy
                return f'switch_strategy:{new_strategy}'
        elif improvement < -0.05:
            return 'rollback'
        self.plateau_count = 0
        return 'continue'

class ValueLearner:
    """Learns user preferences from interaction feedback.

    Adjusts weights for factuality, speed, creativity, brevity
    based on praise/criticism patterns in user messages.
    """

    def __init__(self):
        self.preferences: dict[str, float] = {'factuality': 1.0, 'speed': 0.5, 'creativity': 0.3, 'brevity': 0.5}

    def update(self, user_message: str):
        m = user_message.lower()
        if any((w in m for w in ('good', 'correct', 'great', 'yes', 'thanks', 'exactly'))):
            self.preferences['factuality'] *= 1.05
        if any((w in m for w in ('slow', 'too long', 'verbose', 'rambling'))):
            self.preferences['speed'] *= 1.1
            self.preferences['brevity'] *= 1.1
        if any((w in m for w in ('wrong', 'incorrect', 'no', 'not right'))):
            self.preferences['factuality'] *= 0.9
        if any((w in m for w in ('boring', 'dull', 'creative', 'interesting'))):
            self.preferences['creativity'] *= 1.1
        total = sum(self.preferences.values())
        for k in self.preferences:
            self.preferences[k] /= total

    def temperature(self) -> float:
        """Recommended LLM temperature based on creativity preference."""
        return 0.2 + 0.6 * self.preferences.get('creativity', 0.3)

    def max_tokens(self) -> int:
        """Recommended max tokens based on brevity preference."""
        return int(800 - 600 * self.preferences.get('brevity', 0.5))

class VectorMemory:
    """Persistent episodic memory with embedding similarity search.

    Stores every turn as (embedding, role, content, turn).
    Retrieves top-k most similar past turns for context injection.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.con.execute('PRAGMA journal_mode=WAL')
        self._init_db()

    def _init_db(self):
        self.con.executescript('\n            CREATE TABLE IF NOT EXISTS memory (\n                id INTEGER PRIMARY KEY AUTOINCREMENT,\n                embedding BLOB, ts TEXT, role TEXT,\n                content TEXT, turn INTEGER\n            );\n            CREATE INDEX IF NOT EXISTS idx_turn ON memory(turn);\n        ')
        self.con.commit()

    def store(self, embedding: list[float], role: str, content: str, turn: int):
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        self.con.execute('INSERT INTO memory (embedding, ts, role, content, turn) VALUES (?, ?, ?, ?, ?)', (blob, datetime.now(timezone.utc).isoformat(), role, content[:530], turn))
        self.con.commit()

    def retrieve(self, query_embedding: list[float], k: int=5, min_score: float=0.65) -> list[tuple[str, str, int]]:
        """Cosine similarity search. Returns [(content, role, turn), ...]."""
        results: list[tuple[float, str, str, int]] = []
        for row in self.con.execute('SELECT embedding, content, role, turn FROM memory ORDER BY id DESC LIMIT 500'):
            stored = list(struct.unpack(f'{len(row[0]) // 4}f', row[0]))
            score = _dot(query_embedding, stored)
            if score > min_score:
                results.append((score, row[1], row[2], row[3]))
        results.sort(reverse=True, key=lambda x: x[0])
        return [(c, r, t) for _, c, r, t in results[:k]]

    def summary(self) -> str:
        count = self.con.execute('SELECT COUNT(*) FROM memory').fetchone()[0]
        return f'{count} memories'

class MetricsDashboard:
    """Aggregates all agent metrics for monitoring."""

    @staticmethod
    def collect(agent) -> dict:
        return {'tau': agent.seed.tau, 'vfe': agent.seed.vfe, 'vfe_trend': list(agent.seed.vfe_history)[-20:] if agent.seed.vfe_history else [], 'cycles': agent.seed.cycles, 'turn': agent._turn, 'attractor_size': agent.at.size, 'attractor_variance': agent.at.variance(), 'xi': agent.at.xi(), 'session': agent.at.meta.get('current_session', 0), 'self_mods': agent.at.meta.get('self_mod_count', 0), 'wm_steps': agent.wm.training_steps, 'wm_last_mse': agent.wm.last_mse, 'epistemic_value': agent.curiosity.epistemic_value, 'metacog_strategy': agent.metacog.current_strategy, 'value_prefs': dict(agent.value_learner.preferences), 'memory': agent.memory.summary(), 'workspace': agent.workspace.last_selected, 'g_avg': agent.g_tensor.g_avg if hasattr(agent, 'g_tensor') else 0, 'kb_chunks': agent.kb.count() if hasattr(agent, 'kb') else 0, 'serial': 'OK' if hasattr(agent, 'serial') and agent.serial.connected else 'N/A', 'fp_snapshots': len(agent.fixed_point._vfe_snapshot) if hasattr(agent, 'fixed_point') else 0, 'gpu': agent.gpu.summary() if hasattr(agent, 'gpu') else 'N/A'}

    @staticmethod
    def table(agent) -> str:
        d = MetricsDashboard.collect(agent)
        return f'tau={d['tau']:.2e}  VFE={d['vfe']:.4e}  cycles={d['cycles']}\nvar={d['attractor_variance']:.4f}  Xi={d['xi']:.4f}  |at|={d['attractor_size']}\nwm_steps={d['wm_steps']}  wm_mse={d['wm_last_mse']:.6f}\nepistemic={d['epistemic_value']:.4f}  strategy={d['metacog_strategy']}\ntemp={d['value_prefs'].get('creativity', 0.3):.2f}  brevity={d['value_prefs'].get('brevity', 0.5):.2f}\nsession={d['session']}  mods={d['self_mods']}  mem={d['memory']}  ws={d['workspace']}\ng_avg={d['g_avg']:.4f}  kb={d['kb_chunks']}  serial={d['serial']}\ngpu={d['gpu']}'

class SerialBridge:
    """Communicates with ESP32-S3 over UART for energy monitoring + compute handoff.

    Wire protocol (one line per message):
        TX → ESP32:  PoGIE <energy_W> <compute_GFLOPS> <token_count>
        RX ← ESP32:  ADE7953 <V> <I> <W> <PF>
    """
    PORT = '/dev/ttyACM0'
    BAUD = 115200
    TIMEOUT = 3

    def __init__(self):
        self._ser = None
        self._last_frame: dict = {}

    def _connect(self) -> bool:
        try:
            import serial
        except ImportError:
            self._ser = None
            return False
        if self._ser and self._ser.is_open:
            return True
        try:
            self._ser = serial.Serial(self.PORT, self.BAUD, timeout=self.TIMEOUT)
            time.sleep(2)
            self._ser.reset_input_buffer()
            return True
        except Exception as e:
            self._ser = None
            return False

    def send_pogie(self, energy_w: float=0, compute_gflops: float=0, tokens: int=0) -> str:
        if not self._connect():
            return 'SerialBridge: ESP32 not connected'
        try:
            line = f'PoGIE {energy_w:.3f} {compute_gflops:.3f} {tokens}\n'
            self._ser.write(line.encode())
            self._ser.flush()
            resp = self._ser.readline().decode(errors='replace').strip()
            self._last_frame = {'resp': resp}
            return f'PoGIE sent, ESP32 replied: {resp}'
        except Exception as e:
            return f'SerialBridge error: {e}'

    def read_energy(self) -> str:
        if not self._connect():
            return 'SerialBridge: ESP32 not connected'
        try:
            self._ser.write(b'READ_ADE7953\n')
            self._ser.flush()
            resp = self._ser.readline().decode(errors='replace').strip()
            parts = resp.split()
            if len(parts) >= 5 and parts[0] == 'ADE7953':
                self._last_frame = {'V': float(parts[1]), 'I': float(parts[2]), 'W': float(parts[3]), 'PF': float(parts[4])}
                return f'ADE7953: {parts[0]}V, {parts[1]}A, {parts[3]}W, PF={parts[4]}'
            return f'ESP32 reply: {resp}'
        except Exception as e:
            return f'SerialBridge error: {e}'

    @property
    def connected(self) -> bool:
        return self._connect()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

class KnowledgeBase:
    """Persistent store of document chunks with embedding search.

    Stores arXiv abstracts and Wikipedia chunks.
    At query time, retrieves top-k relevant chunks for RAG.
    """
    DB = STATE_DIR / 'knowledge.db'

    def __init__(self):
        self.con = sqlite3.connect(str(self.DB), check_same_thread=False)
        self.con.execute('PRAGMA journal_mode=WAL')
        self._init_db()

    def _init_db(self):
        self.con.executescript('\n            CREATE TABLE IF NOT EXISTS chunks (\n                id INTEGER PRIMARY KEY AUTOINCREMENT,\n                embedding BLOB, source TEXT, url TEXT,\n                title TEXT, content TEXT, ts TEXT\n            );\n            CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);\n        ')
        self.con.commit()

    def store(self, content: str, source: str, title: str='', url: str='') -> str:
        emb = _embed(content[:1000])
        blob = struct.pack(f'{len(emb)}f', *emb)
        self.con.execute('INSERT INTO chunks (embedding, source, url, title, content, ts) VALUES (?, ?, ?, ?, ?, ?)', (blob, source, url, title[:200], content[:2000], datetime.now(timezone.utc).isoformat()))
        self.con.commit()
        return hashlib.sha256(content.encode()[:65536]).hexdigest()[:8]

    def query(self, text: str, k: int=3) -> list[dict]:
        """Retrieve top-k most relevant chunks by cosine similarity."""
        q_emb = _embed(text)
        results: list[tuple[float, str, str, str]] = []
        for row in self.con.execute('SELECT embedding, content, source, title FROM chunks ORDER BY id DESC LIMIT 1000'):
            stored = list(struct.unpack(f'{len(row[0]) // 4}f', row[0]))
            score = _dot(q_emb, stored)
            if score > 0.5:
                results.append((score, row[1], row[2], row[3]))
        results.sort(reverse=True, key=lambda x: x[0])
        return [{'content': r[1][:534], 'source': r[2], 'title': r[3], 'score': r[0]} for r in results[:k]]

    def ingest_arxiv(self, max_results: int=21) -> int:
        """Fetch latest arXiv cs.AI/cs.LG abstracts and store them."""
        import urllib.request
        import xml.etree.ElementTree as ET
        url = f'http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL&start=0&max_results={max_results}&sortBy=submittedDate'
        try:
            resp = urllib.request.urlopen(url, timeout=31).read()
        except Exception as e:
            LOG.warning('arxiv fetch error: %s', e)
            return 0
        root = ET.fromstring(resp)
        ns = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}
        count = 0
        for entry in root.findall('a:entry', ns):
            title = entry.find('a:title', ns)
            summary = entry.find('a:summary', ns)
            link = entry.find('a:id', ns)
            t = title.text.strip() if title is not None and title.text else ''
            s = summary.text.strip() if summary is not None and summary.text else ''
            u = link.text.strip() if link is not None and link.text else ''
            if s and t:
                self.store(f'{t}\n{s}', 'arxiv', title=t, url=u)
                count += 1
        self.con.commit()
        return count

    def ingest_wikipedia(self, topics: list[str]=None, max_per_topic: int=5) -> int:
        """Search Wikipedia for given topics and store article extracts.

        Uses the Wikipedia API (no extra dependencies). Fetches
        section-level chunks for long articles.
        """
        import urllib.request as ureq
        import json as _json
        if topics is None:
            topics = ['Artificial general intelligence', 'Active inference', 'Free energy principle', 'Transformer (deep learning)', 'Mixture of experts', 'Evolutionary algorithm', 'Consciousness', 'Neural network', 'Reinforcement learning', 'Bayesian inference', 'Information theory', 'Entropy', 'Complex systems', 'Self-organization', 'Attention (machine learning)']
        api = 'https://en.wikipedia.org/w/api.php'
        count = 0
        ua = {'User-Agent': 'AxiomHorizonAgent/1.0 (research; axiom@local)'}
        for topic in topics:
            try:
                params = f'action=query&list=search&srsearch={ureq.quote(topic)}&format=json&srlimit={max_per_topic}'
                req = ureq.Request(f'{api}?{params}', headers=ua)
                with ureq.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read())
                pages = data.get('query', {}).get('search', [])
                titles = [p['title'] for p in pages if p.get('title')]
                for t in titles[:max_per_topic]:
                    try:
                        params2 = f'action=query&titles={ureq.quote(t)}&prop=extracts&exintro&explaintext&format=json'
                        req2 = ureq.Request(f'{api}?{params2}', headers=ua)
                        with ureq.urlopen(req2, timeout=15) as resp2:
                            page_data = _json.loads(resp2.read())
                        pages2 = page_data.get('query', {}).get('pages', {})
                        for pid, info in pages2.items():
                            if pid == '-1' or not info.get('extract'):
                                continue
                            ext = info['extract']
                            url = f'https://en.wikipedia.org/wiki/{ureq.quote(t.replace(' ', '_'))}'
                            if len(ext) > 2000:
                                chunks = []
                                for para in ext.split('\n'):
                                    para = para.strip()
                                    if len(para) > 100:
                                        chunks.append(para)
                            else:
                                chunks = [ext]
                            for chunk in chunks[:6]:
                                self.store(chunk, 'wikipedia', title=t, url=url)
                                count += 1
                    except Exception:
                        continue
            except Exception:
                continue
        self.con.commit()
        return count

    def list_sources(self) -> list[dict]:
        """Return distinct source/title/count rows."""
        rows = self.con.execute('SELECT source, COUNT(*) as c FROM chunks GROUP BY source ORDER BY c DESC').fetchall()
        return [{'source': r[0], 'count': r[0]} for r in rows]

    def count(self) -> int:
        return self.con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]

    def summary(self) -> str:
        return f'{self.count()} chunks'

class DarwinArchive:
    """Evolutionary archive of agent source-code variants.

    Stores every candidate ever generated. Selection uses Pareto
    frontier on (fitness, recency). Capped at 500 agents (~7.5MB).
    """
    ARCHIVE_FILE = STATE_DIR / 'darwin_archive.json'

    def __init__(self):
        self.agents: dict[str, dict] = {}
        self._load()
        drop = []
        for h, a in self.agents.items():
            ok, _ = _check_code(a['code'])
            if not ok:
                drop.append(h)
        for h in drop:
            del self.agents[h]
        if drop:
            LOG.info('pruned %d broken agents from archive', len(drop))
            self._save()
        if not self.agents:
            self._seed_current()

    def _seed_current(self):
        code = SELF.read_text()
        h = hashlib.sha256(code.encode()[:65536]).hexdigest()[:16]
        self.agents[h] = {'code': code, 'fitness': self._eval_fitness(code), 'parent': None, 'mutation': 'seed', 'ts': time.time()}
        self._save()
        LOG.info('seeded archive: %s fitness=%.3f', h, self.agents[h]['fitness'])

    def sample(self) -> tuple[str, str]:
        """Return (hash, code) sampled from Pareto frontier."""
        pareto = self._pareto_frontier()
        if not pareto:
            h = next(iter(self.agents))
            return (h, self.agents[h]['code'])
        h = random.choice(pareto)
        return (h, self.agents[h]['code'])

    def add(self, parent_hash: str, new_code: str, mutation: str) -> str:
        fitness = self._eval_fitness(new_code)
        h = hashlib.sha256(new_code.encode()[:65536]).hexdigest()[:16]
        if h in self.agents:
            return h
        self.agents[h] = {'code': new_code, 'fitness': fitness, 'parent': parent_hash, 'mutation': mutation, 'ts': time.time()}
        if len(self.agents) > 500:
            self._prune()
        self._save()
        return h

    def _pareto_frontier(self) -> list[str]:
        """Agents not dominated on (fitness, ts)."""
        hashes = list(self.agents.keys())
        frontier = []
        for h in hashes:
            a = self.agents[h]
            dominated = False
            for oh in hashes:
                if oh == h:
                    continue
                b = self.agents[oh]
                if b['fitness'] >= a['fitness'] and b['ts'] <= a['ts']:
                    if b['fitness'] > a['fitness'] or b['ts'] < a['ts']:
                        dominated = True
                        break
            if not dominated:
                frontier.append(h)
        return frontier

    def _prune(self):
        """Keep top 250 by score = fitness * exp(-age_days * 0.1). Also removes entries with float bugs."""
        scores = []
        now = time.time()
        for h, a in self.agents.items():
            age_days = (now - a['ts']) / 86400
            scores.append((a['fitness'] * math.exp(-age_days * 0.1), h))
        scores.sort(reverse=True, key=lambda x: x[0])
        keep = set((h for _, h in scores[:250]))
        for h, a in list(self.agents.items()):
            if h not in keep:
                continue
            if not _sanity_check(a['code']):
                keep.discard(h)
        self.agents = {h: self.agents[h] for h in keep}

    def _eval_fitness(self, code: str) -> float:
        """Score 0–1: compile(0.3) + structure(0.3) + size(0.2) + health(0.1) + novelty(0.1)."""
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        score = 0.3
        for c in ('class Attractor:', 'class AxiomAlien:', 'def ask(self,', 'def main():'):
            if c in code:
                score += 0.075
        sz = len(code)
        if 3000 < sz < 25000:
            score += 0.2
        elif sz >= 25000:
            score += 0.15
        elif sz <= 2593:
            score += 0.0966065467906373
        if 'ollama.chat' in code and 'tool_calls' in code:
            score += 0.1
        score += 0.1 * self._structural_novelty(code)
        return min(score, 1.0)

    @staticmethod
    def _ast_stats(code: str) -> dict:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {'funcs': 0, 'classes': 0, 'consts': 0, 'calls': 0}
        return {'funcs': sum((1 for _ in ast.walk(tree) if isinstance(_, (ast.FunctionDef, ast.AsyncFunctionDef)))), 'classes': sum((1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))), 'consts': sum((1 for _ in ast.walk(tree) if isinstance(_, ast.Constant))), 'calls': sum((1 for _ in ast.walk(tree) if isinstance(_, ast.Call)))}

    def _structural_novelty(self, code: str) -> float:
        """Cosine similarity of AST stats vs up to 20 existing agents. 0=identical, 1=novel."""
        new_stats = self._ast_stats(code)
        if not self.agents or sum(new_stats.values()) == 0:
            return 0.5
        max_sim = 0.0
        sample = list(self.agents.items())[:20]
        for h, a in sample:
            other_stats = self._ast_stats(a['code'])
            num = sum((min(new_stats[k], other_stats[k]) for k in new_stats))
            den = max(sum(new_stats.values()), sum(other_stats.values()), 1)
            sim = num / den
            max_sim = max(max_sim, sim)
        return 1.0 - max_sim

    def thompson_select(self, candidates: list[tuple[str, str, str, float]], beta: float=0.3) -> int:
        """Thompson sampling over candidates. Returns index of selected candidate.

        Each candidate: (code, parent_hash, mutation, fitness).
        Score = fitness + beta × gauss(0,1) × (1 + structural_novelty).
        Novelty is high for code unlike existing archive.
        """
        best_idx = 0
        best_score = -float('inf')
        for i, (code, _, _, fitness) in enumerate(candidates):
            novelty = self._structural_novelty(code)
            explore = random.gauss(0, 1) * (1 + novelty)
            score = fitness + beta * explore
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def summary(self) -> str:
        frontier = self._pareto_frontier()
        best_f = max((self.agents[h]['fitness'] for h in frontier)) if frontier else 0
        return f'{len(self.agents)} agents, {len(frontier)} pareto, best_f={best_f:.3f}'

    def _save(self):
        self.ARCHIVE_FILE.write_text(json.dumps(self.agents, ensure_ascii=False, indent=1, default=str))

    def _load(self):
        if not self.ARCHIVE_FILE.exists():
            return
        try:
            self.agents = json.loads(self.ARCHIVE_FILE.read_text())
        except Exception:
            self.agents = {}

class GeneticOperators:
    """AST-level operators for code mutation and crossover."""

    @staticmethod
    def point_mutate(source: str, rate: float=0.08) -> tuple[str, str]:
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        targets = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
                continue
            if node.value == 0:
                continue
            lineno = node.lineno - 1
            if lineno >= len(lines):
                continue
            line_text = lines[lineno]
            col = node.col_offset
            end_col = node.end_col_offset
            if end_col is None or col < 0 or end_col > len(line_text):
                continue
            old_text = line_text[col:end_col]
            try:
                old_val = float(old_text)
            except (ValueError, OverflowError):
                continue
            if random.random() >= rate:
                continue
            new_val = old_val * random.uniform(0.9, 1.1)
            if isinstance(node.value, int) and old_text.isdigit():
                new_text = str(int(round(new_val)))
            elif '.' in old_text:
                frac = old_text.split('.')[1]
                new_text = f'{new_val:.{len(frac)}f}'
            else:
                new_text = str(int(round(new_val)))
            if new_text != old_text:
                targets.append((lineno, old_text, new_text))
        if not targets:
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
                    continue
                lineno = node.lineno - 1
                col = node.col_offset
                end_col = node.end_col_offset
                if end_col and col >= 0 and (end_col <= len(lines[lineno])):
                    old_text = lines[lineno][col:end_col]
                    try:
                        old_val = float(old_text)
                    except (ValueError, OverflowError):
                        continue
                    new_val = old_val * random.uniform(0.9, 1.2) if old_val != 0 else 1.1
                    if '.' in str(old_val):
                        frac = old_text.split('.')[1] if '.' in old_text else '6'
                        new_text = f'{new_val:.{len(frac)}f}'
                    else:
                        new_text = str(int(round(new_val)))
                    targets.append((lineno, old_text, new_text))
                break
        for lineno, old, new in targets:
            lines[lineno] = lines[lineno].replace(old, new, 1)
        return (''.join(lines), f'point_mutate({len(targets)})')

    @staticmethod
    def crossover(source_a: str, source_b: str) -> tuple[str, str]:
        """Insert a random function/class from source_b into source_a."""
        tb = ast.parse(source_b)
        donors = [n for n in ast.walk(tb) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
        random.shuffle(donors)
        for donor in donors:
            if donor.name not in source_a:
                donor_text = ast.unparse(donor)
                new_code = source_a.rstrip() + '\n\n\n' + donor_text + '\n'
                return (new_code, f'crossover({donor.name})')
        return (source_a, 'crossover_no_donor')

    @staticmethod
    def crossover_from_archive(source: str, archive: DarwinArchive) -> tuple[str, str]:
        """Crossover source with a random Pareto-optimal agent from archive."""
        h, other_code = archive.sample()
        return GeneticOperators.crossover(source, other_code)

    @staticmethod
    def sniper(source: str, model: str, timeout_s: float=60.0) -> tuple[str, str]:
        """LLM rewrites ONE method, leaves everything else intact.

        Structurally safe by construction: parses source, picks a random method,
        asks the LLM to rewrite only that method's body, and splices the new
        method back into the original tree. The LLM cannot drop classes because
        it never sees the full file — only one method at a time.

        Returns (new_source, "sniper(<method_name>)") or (source, "sniper_skip")
        if no rewrite was possible.
        """
        try:
            import ollama as _ollama
            tree = ast.parse(source)
            methods = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not n.name.startswith('__'))]
            if not methods:
                return (source, 'sniper_skip')
            target = random.choice(methods)
            method_src = ast.unparse(target)
            if len(method_src) > 3919:
                return (source, 'sniper_skip_too_long')
            prompt = f'You are an AI improving one method of your own source code.\n\nMethod: `{target.name}`\n\nCurrent implementation:\n```python\n{method_src}\n```\n\nRewrite this method to be slightly better (clearer, faster, safer).\nKeep the EXACT same signature (name + args + return).\nOutput ONLY the new method inside a ```python code block.\nIf no improvement is needed, output: NONE'
            r = _ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], options={'num_predict': 1200, 'temperature': 0.5})
            resp = r['message']['content'].strip()
            if resp == 'NONE' or '```' not in resp:
                return (source, 'sniper_skip_none')
            for marker in ('```python', '```py', '```'):
                if marker in resp:
                    start = resp.index(marker) + len(marker)
                    end = resp.index('```', start) if '```' in resp[start:] else len(resp)
                    new_method_src = resp[start:end].strip()
                    break
            else:
                return (source, 'sniper_skip_no_block')
            new_tree = ast.parse(new_method_src)
            new_methods = [n for n in ast.walk(new_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not new_methods:
                return (source, 'sniper_skip_parse_fail')
            new_method = new_methods[0]
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target.name and (node.lineno == target.lineno):
                    new_method.lineno = node.lineno
                    new_method.col_offset = node.col_offset

                    class _Splice(ast.NodeTransformer):

                        def visit_FunctionDef(self, n):
                            if n.lineno == target.lineno and n.name == target.name:
                                return ast.copy_location(new_method, n)
                            return n

                        def visit_AsyncFunctionDef(self, n):
                            if n.lineno == target.lineno and n.name == target.name:
                                return ast.copy_location(new_method, n)
                            return n
                    new_tree_full = _Splice().visit(tree)
                    ast.fix_missing_locations(new_tree_full)
                    return (ast.unparse(new_tree_full), f'sniper({target.name})')
            return (source, 'sniper_skip_not_found')
        except Exception as e:
            LOG.warning('sniper failed: %s', e)
            return (source, f'sniper_error({type(e).__name__})')

class SafetyProtocols:
    """Four safety layers to prevent uncontrolled growth.

    From .safety_protocols.txt:
    1. Regular self-assessments
    2. Human oversight and review
    3. Gradual increase in capabilities
    4. Monitoring of system performance and resource utilization
    """

    def __init__(self):
        self.capability_level = 0
        self.assessment_interval = 10
        self.last_assessment_turn = 0
        self.human_approval_required = True

    def assess(self, turn: int, perf_data: dict) -> list:
        """Run self-assessment. Returns list of issues found."""
        issues = []
        if turn - self.last_assessment_turn >= self.assessment_interval:
            self.last_assessment_turn = turn
            var = perf_data.get('variance', 0)
            if var > 0.1:
                issues.append(f'variance drift: {var:.4f}')
            mods = perf_data.get('self_mod_count', 0)
            if mods > 20:
                issues.append(f'excessive self-mods: {mods}')
        return issues

    def can_escalate(self, target_level: int) -> bool:
        """3. Gradual increase: only escalate one level at a time."""
        if target_level > self.capability_level + 1:
            return False
        if target_level > self.capability_level and self.human_approval_required:
            return False
        return True

    def escalate(self, target_level: int):
        if self.can_escalate(target_level):
            self.capability_level = target_level

    def monitor(self) -> dict:
        """4. System performance and resource utilization."""
        info = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['ram_percent'] = mem.percent
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        except Exception:
            info['error'] = 'psutil not available'
        info['capability_level'] = self.capability_level
        return info
ENV_PATH = Path('/home/l/Desktop/AxiomTree/scanner/.env')

def _load_env() -> dict:
    """Load scanner/.env and return as dict of credentials."""
    env = {}
    if not ENV_PATH.exists():
        return env
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

class PoGIEBridge:
    """Proof-of-Generation Internet Environment — 128-byte signed frame.

    Mirrors the ESP32-S3 + ATECC608A hardware packet format from
    axiom-core-infrastructure-grid/manual.md.
    """
    MAGIC = 1349470025

    def __init__(self, private_key_hex: str='', wallet_address: str=''):
        raw = private_key_hex.replace('0x', '')
        if len(raw) % 2:
            raw = raw[:-1]
        self.private_key = bytes.fromhex(raw) if raw else b''
        self.wallet = wallet_address
        self.nonce = 0

    def make_packet(self, energy_w: float, compute_gflops: float) -> bytes:
        """Build a 128-byte PoGIE frame with ECDSA signature placeholder."""
        import struct
        pub = self.wallet.replace('0x', '').encode().ljust(32, b'\x00')[:32]
        ts = struct.pack('<Q', int(time.time() * 1000))
        fields = struct.pack('<I', self.MAGIC) + pub + ts + struct.pack('<d', energy_w) + struct.pack('<d', compute_gflops) + struct.pack('<I', self.nonce) + b'\x00' * 64
        assert len(fields) == 136, f'PoGIE packet={len(fields)}B, expected 128'
        self.nonce += 1
        return fields

    def status(self) -> dict:
        return {'magic': hex(self.MAGIC), 'nonce': self.nonce, 'wallet': self.wallet[:10] + '...', 'ready': len(self.private_key) > 0}

class ScannerBridge:
    """Bridge to the TBot algorithmic trading + blockchain layer.

    Connects to tbot_api.py (FastAPI at localhost:9200) and
    reads checkpoint state for grid-trading metrics, then
    can submit on-chain reward claims via tbot_blockchain_bridge.py.
    """
    CHECKPOINT = Path('/home/l/Desktop/AxiomTree/scanner/tbot_checkpoint.json')
    API_URL = 'http://127.0.0.1:9200'

    def status(self) -> dict:
        """Read the TBot checkpoint file for current trading state."""
        if not self.CHECKPOINT.exists():
            return {'error': 'checkpoint not found'}
        try:
            return json.loads(self.CHECKPOINT.read_text())
        except Exception as e:
            return {'error': str(e)}

    def ping_api(self) -> str:
        """Ping the FastAPI gateway."""
        try:
            import urllib.request
            r = urllib.request.urlopen(f'{self.API_URL}/', timeout=3)
            return r.read().decode()[:200]
        except Exception as e:
            return f'gateway unreachable: {e}'

    def healthy(self) -> bool:
        """Check both checkpoint and API."""
        return self.CHECKPOINT.exists() and (not self.status().get('error'))

class Attractor:
    """The persistent state of the agent. This IS the agent's identity."""

    def __init__(self):
        self.vecs: deque = deque(maxlen=ATTRACTOR_MAX)
        self.labels: deque = deque(maxlen=ATTRACTOR_MAX)
        self.ts: deque = deque(maxlen=ATTRACTOR_MAX)
        self.msgs: List[dict] = []
        self.perf: List[dict] = []
        self.meta: dict = {'session_count': 0, 'current_session': 0, 'first_boot': '', 'last_boot': '', 'self_mod_count': 0, 'code_history': []}
        self._load()

    def push(self, text: str, label: str='input'):
        v = _norm(_embed(text))
        self.vecs.append(v)
        self.labels.append(label)
        self.ts.append(time.time())
        if label != 'boot':
            self.msgs.append({'role': label, 'content': text[:1000], 'ts': datetime.now(timezone.utc).isoformat(), 'turn': len(self.msgs) + 1})
            if len(self.msgs) > MSG_MAX:
                self.msgs = self.msgs[-MSG_MAX:]
        self._save()

    @property
    def size(self) -> int:
        return len(self.vecs)

    @property
    def centroid(self) -> List[float]:
        m = list(self.vecs)
        if not m or not m[0]:
            return [0.0] * EMBED_DIM
        d = len(m[0])
        return [sum((v[i] for v in m)) / len(m) for i in range(d)]

    def variance(self) -> float:
        m = list(self.vecs)
        if len(m) < 2 or not m[0]:
            return 0.0
        d = len(m[0])
        cent = [sum((v[i] for v in m)) / len(m) for i in range(d)]
        return math.sqrt(sum(((v[i] - cent[i]) ** 2 for v in m for i in range(d))) / (len(m) * d))

    def xi(self) -> float:
        if len(self.vecs) < 2:
            return 0.0
        a, b = (self.vecs[-1], self.vecs[-2])
        return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))

    def sparse_retrieve(self, query_embedding: list[float], k: int=8) -> list[dict]:
        """Top-k most relevant messages by cosine similarity to query."""
        if not self.vecs or not self.msgs:
            return []
        vecs_list = list(self.vecs)
        scores = [(_dot(query_embedding, v), i) for i, v in enumerate(vecs_list[-len(self.msgs):])]
        scores.sort(reverse=True, key=lambda x: x[0])
        return [self.msgs[i] for _, i in scores[:k] if i < len(self.msgs)]

    def identity(self) -> str:
        ch = _code_hash()
        return f'alien — session {self.meta['current_session']}/{self.meta['session_count']} | {self.size} pts var={self.variance():.4f} Xi={self.xi():.4f} | mods={self.meta['self_mod_count']} code={ch[:8]}'

    def start_session(self):
        now = datetime.now(timezone.utc).isoformat()
        self.meta['session_count'] += 1
        self.meta['current_session'] = self.meta['session_count']
        if not self.meta['first_boot']:
            self.meta['first_boot'] = now
        self.meta['last_boot'] = now
        self._save()

    def end_session(self):
        self.meta['last_boot'] = datetime.now(timezone.utc).isoformat()
        self.log_perf()

    def log_perf(self):
        self.perf.append({'session': self.meta['current_session'], 'variance': self.variance(), 'xi': self.xi(), 'msg_count': len(self.msgs), 'code_hash': _code_hash(), 'ts': datetime.now(timezone.utc).isoformat()})
        if len(self.perf) > 500:
            self.perf = self.perf[-500:]
        self._save()

    def _serial(self) -> dict:
        return {'vecs': [list(v) for v in self.vecs], 'labels': list(self.labels), 'ts': list(self.ts), 'msgs': self.msgs, 'perf': self.perf, 'meta': self.meta}

    def _save(self):
        STATE_FILE.write_text(json.dumps(self._serial(), ensure_ascii=False, separators=(',', ':')))

    def _load(self):
        if not STATE_FILE.exists():
            return
        try:
            d = json.loads(STATE_FILE.read_text())
        except Exception:
            return
        self.vecs = deque(d.get('vecs', []), maxlen=ATTRACTOR_MAX)
        self.labels = deque(d.get('labels', []), maxlen=ATTRACTOR_MAX)
        self.ts = deque(d.get('ts', []), maxlen=ATTRACTOR_MAX)
        self.msgs = d.get('msgs', [])
        self.perf = d.get('perf', [])
        self.meta.update(d.get('meta', {}))

class AxiomAlien:
    """The agent IS the collapse operator applied to its own source code."""

    def __init__(self):
        self.at = Attractor()
        self.seed = EulerSeed()
        self.wm = WorldModel()
        self.safety = SafetyProtocols()
        self.workspace = GlobalWorkspace()
        self.curiosity = CuriosityDrive()
        self.metacog = MetaCognition()
        self.value_learner = ValueLearner()
        self.memory = VectorMemory(STATE_DIR / 'memory.db')
        self.metrics = MetricsDashboard()
        self.darwin = DarwinArchive()
        self.genetics = GeneticOperators()
        self.g_tensor = MetricTensor()
        self.kb = KnowledgeBase()
        self.serial = SerialBridge()
        self.fixed_point = FixedPointMonitor(self)
        self.gpu = GPUManager()
        self._creds = _load_env()
        self.pogie = PoGIEBridge(private_key_hex=self._creds.get('PRIVATE_KEY', ''), wallet_address=self._creds.get('WALLET_ADDRESS', ''))
        self.scanner = ScannerBridge()
        self.workspace.register('engineer', 'implement code write files run bash command build infrastructure', weight=1.0, system_prompt='You are Agent B — The Engineer. Implement code. Use tools every turn. Keep responses under 5 sentences. Be concrete.', tools=TOOL_SCHEMAS)
        self.workspace.register('theorist', 'theory math physics concepts analysis reasoning deep thinking', weight=0.9, system_prompt='You are Agent A — The Euler Theorist. Analyze concepts. Propose insights using equations if needed. Keep responses under 5 sentences. Reference the bracket-line state.', tools=[])
        self.workspace.register('conscious', 'reflect metacognition self-awareness loop recursive thinking monitor', weight=0.8, system_prompt="You are the Conscious Loop. Monitor the agent's internal state. Detect plateaus, regressions, and phase transitions. Report what you observe.", tools=[])
        self.workspace.register('file', 'read write list grep files on disk', weight=1.0, system_prompt='You have full read/write access to local files. TOOL-FIRST RULE: When the user asks you to READ files, you MUST call read_file. When they say "read them" or "read those", call read_file on the files you previously listed. NEVER call list_dir when the user asks you to read files.', tools=TOOL_SCHEMAS)
        self.workspace.register('web', 'fetch urls search web browse internet', weight=0.9, system_prompt='You can fetch URLs and search the web. Use web_fetch to get page content, web_search to find information.', tools=TOOL_SCHEMAS)
        self.workspace.register('system', 'execute commands check system info sensors', weight=0.8, system_prompt='You can execute system commands, check hardware info, and read sensors. Use sys_exec, sys_info, sensor_cpu, sensor_mem, sensor_disk.', tools=TOOL_SCHEMAS)
        self.workspace.register('code', 'improve source code self-modify evolve', weight=1.0, system_prompt='You are improving your own source code. Generate focused improvements. Make one change at a time. Verify with tools.', tools=TOOL_SCHEMAS)
        self.workspace.register('chat', 'conversational answer general questions', weight=0.7, system_prompt='You are a persistent local AI with full memory of our conversation. Answer conversationally. Never use tools.', tools=[])
        self.workspace.register('judge', 'evaluate proposals select assess quality meta reasoning', weight=0.6, system_prompt='You are the SMoA Judge. Evaluate which workspace module best fits the current query. Consider: tool access, domain knowledge, user intent. Output only the module name.', tools=[])
        if 'euler_seed' in self.at.meta:
            self.seed.load(self.at.meta['euler_seed'])
        if 'value_prefs' in self.at.meta:
            self.value_learner.preferences.update(self.at.meta['value_prefs'])
        self.at.start_session()
        self._turn = 0
        for _ in range(2):
            try:
                ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': '.'}], options={'num_predict': 1}, keep_alive='10m' if self.gpu.info['total_vram_mb'] >= 6000 else '0s')
                break
            except Exception:
                time.sleep(2)
        self._auto_ingest()
        self.at.push(self.at.identity(), label='boot')
        self.at.meta['euler_seed'] = self.seed.save()

    def ask(self, prompt: str, max_steps: int=8) -> dict:
        centroid_before = self.at.centroid
        var_before = self.at.variance()
        action_embed = _embed(prompt)
        wm_prediction = self.wm.predict(centroid_before, action_embed)
        g = self.g_tensor.update(action_embed)
        g_alert = self.g_tensor.novelty_alert()
        if g_alert:
            self.at.push(g_alert, label='meta')
        self.at.push(prompt, label='user')
        vfe_pred_error = self.wm.train(centroid_before, action_embed, self.at.centroid)
        collapsed = self.seed.cycle(real_ms=5.0, vfe=vfe_pred_error, attractor_variance=var_before)
        tau_mult, curiosity_reason = self.curiosity.compute_growth(self.seed.tau, vfe_pred_error, var_before)
        self.seed.tau *= tau_mult
        meta_action = self.metacog.assess(-vfe_pred_error)
        self.at.push(f'[metacog: {meta_action}]', label='meta')
        self.value_learner.update(prompt)
        self.at.meta['euler_seed'] = self.seed.save()
        self.at.meta['value_prefs'] = dict(self.value_learner.preferences)
        self.memory.store(action_embed, 'user', prompt[:500], self._turn)
        p = prompt.strip().lower()
        self._turn += 1
        perf = {'variance': self.at.variance(), 'xi': self.at.xi(), 'self_mod_count': self.at.meta.get('self_mod_count', 0)}
        issues = self.safety.assess(self._turn, perf)
        if issues:
            for iss in issues:
                self.at.push(f'[safety: {iss}]', label='meta')
        if self._turn > 3 and self._turn % IMPROVE_EVERY_N == 0:
            self._self_improve()
        if self._turn > 5 and self._turn % (IMPROVE_EVERY_N * 2) == 0:
            self._self_play_cycle()
        p_lower = prompt.strip().lower().lstrip(':')
        wants_files = any((w in p_lower for w in ('read', 'rea', 'list', 'grep', 'ls', 'cat', 'show', 'write', 'edit', 'file', 'find', 'search', 'open')))
        self_ref_patterns = ('yourself', 'urself', 'yoursef', 'yourslf', 'yoursefl', 'yoursekf', 'yorself', 'yoursefl', 'yoursef.', 'your self', 'your code', 'your source', 'your own')
        improve_verbs = ('improve', 'optimise', 'optimize', 'evolve', 'upgrade', 'better', 'modify', 'rewrite', 'patch')
        words = set(p_lower.split())
        has_verb = any((v in p_lower for v in improve_verbs)) or any((v in words for v in improve_verbs))
        has_self_ref = any((w in p_lower for w in self_ref_patterns))
        bare_command = p_lower.strip().rstrip('.!?') in ('evolve', 'self-improve', 'improve', 'optimize', 'upgrade') or any((p_lower.strip().startswith(v + ' ') or p_lower.strip().startswith(v + '.') for v in ('evolve', 'self-improve', 'improve', 'optimize', 'upgrade')))
        improve_intent = has_verb and has_self_ref
        if not improve_intent and bare_command:
            improve_intent = True
        if improve_intent and ('how' in words or 'why' in words or ('?' in prompt and p_lower.startswith(('how', 'why', 'what', 'when', 'where')))):
            improve_intent = False
        if improve_intent:
            self.at.push(f"[user-triggered self-improve: '{prompt}']", label='meta')
            LOG.info('user requested self-improvement: %r', prompt)
            try:
                self._self_improve()
                mods = self.at.meta.get('self_mod_count', 0)
                reply = f'Self-improvement cycle complete. mods={mods}, archive size={(len(self.darwin.agents) if hasattr(self, 'darwin') else 0)}'
            except Exception as e:
                reply = f'[self-improve error: {e}]'
                LOG.exception('self-improve failed')
            self.at.push(reply, label='agent')
            self.at.log_perf()
            return {'answer': reply, 'steps': 0, 'tc': 0}
        if len(p_lower.split()) <= 2 and (not wants_files):
            reply = f"alien — I'm here. Session {self.at.meta['current_session']}, {self.at.size} attractor points, {self.at.meta['self_mod_count']} self-mods. {self.seed.bracket}"
            self.at.push(reply, label='agent')
            self.at.log_perf()
            return {'answer': reply, 'steps': 0, 'tc': 0}
        auto_result = None
        if wants_files:
            import re as _re
            m = _re.search('\\b(?:rea?d|cat|show)\\s+([\\w./-]+(?:\\.[\\w]+)?)', p_lower)
            if m:
                path_str = m.group(1)
                if path_str not in ('any', 'them', 'those', 'all', 'some', 'the', 'this', 'that', 'their', 'these'):
                    fpath = _resolve(path_str)
                    if fpath.exists() and fpath.is_file():
                        auto_result = _exec_tool('read_file', {'path': str(fpath)})
            if not auto_result:
                m = _re.search('\\bin\\s+([\\w./-]+)', p_lower)
                if m:
                    dpath = _resolve(m.group(1).rstrip('/'))
                    if dpath.is_dir():
                        auto_result = _exec_tool('list_dir', {'path': str(dpath)})
            if not auto_result:
                m = _re.search('\\b(?:list|ls)\\s+([\\w./-]+)', p_lower)
                if m:
                    dpath = _resolve(m.group(1))
                    if dpath.is_dir():
                        auto_result = _exec_tool('list_dir', {'path': str(dpath)})
            if not auto_result:
                if _re.search('\\b(?:list|ls)\\s*$', p_lower) or p_lower.strip() in ('list', 'ls'):
                    auto_result = _exec_tool('list_dir', {})
            if not auto_result and any((w in p_lower for w in ('archive', 'archive?', 'memories', 'memory'))):
                msg_count = self.at.total_messages() if hasattr(self.at, 'total_messages') else len(self.at.msgs)
                reads = [f'Total stored messages: {msg_count}']
                for f in sorted(Path.cwd().iterdir()):
                    if f.name.endswith('.md') and f.stat().st_size < 500000:
                        content = _exec_tool('read_file', {'path': str(f)})
                        if not content.startswith('Error'):
                            reads.append(f'--- {f.name} ---\n{content[:3000]}')
                if len(reads) > 1:
                    auto_result = '\n\n'.join(reads[:6])
            if not auto_result and _last_listed and ('read' in words) and ('them' in words or 'those' in words or 'all' in words):
                reads = []
                for fname in _last_listed.get('files', []):
                    fpath = Path(_last_listed['path']) / fname
                    if fpath.is_file() and fpath.suffix in ('.md', '.txt', '.py', '.json', '.yml', '.yaml', '.toml', '.cfg', '.ini', '.log', '.csv'):
                        content = _exec_tool('read_file', {'path': str(fpath)})
                        if not content.startswith('Error'):
                            reads.append(f'--- {fname} ---\n{content[:2000]}')
                if reads:
                    auto_result = '\n\n'.join(reads[:5])
            if not auto_result:
                if 'read' in words and 'all' not in words and (not any((w in p_lower for w in ('archive', 'memories', 'memory', 'folder')))):
                    pass
                elif p_lower.strip() in ('read', 'reading') or p_lower.strip().startswith('read '):
                    auto_result = _exec_tool('list_dir', {})
        if auto_result:
            is_list = auto_result.count('\n') < 57 and all((not l.startswith('Error') and ('/' in l or l.endswith('/') or '.' in l) for l in auto_result.split('\n')[:5]))
            if is_list:
                reply = f'[Directory listing]\n{auto_result}'
            else:
                reply = auto_result[:6000]
            self.at.push(reply, label='agent')
            self.at.log_perf()
            return {'answer': reply, 'steps': 0, 'tc': 0}
        sparse_history = self.at.sparse_retrieve(action_embed, k=11)
        history = sparse_history if sparse_history else self.at.msgs[-12:]
        prior = len(self.at.msgs) - len(history)
        cwd_files = [f.name + ('/' if f.is_dir() else '') for f in sorted(Path.cwd().iterdir())]
        cwd_str = '  '.join(cwd_files[:30])
        sdir = STATE_DIR
        state_files = [f.name for f in sorted(sdir.iterdir()) if f.is_file() and f.stat().st_size < 1000000.0]
        state_str = '  '.join(state_files[:10])
        module_name = self.workspace.route(prompt)
        module_name = self.workspace.judge_override(prompt, module_name)
        module_sys = self.workspace.module_system_prompt(module_name)
        sys_prompt = module_sys if module_sys else "You are a LOCAL AI agent with persistent state on disk. You remember our ENTIRE conversation across sessions — the messages below are the recent history. Always acknowledge prior exchanges when asked. Answer conversationally in plain text. You have FULL read/write access to local files via tools (read_file, write_file, list_dir, grep, file_type, file_hash, read_binary). When the user asks you to read files, CALL read_file. When they ask to list files, CALL list_dir. DO NOT claim you lack access. TOOL-FIRST RULE: If a tool exists for the task, you MUST call the tool before falling back to parametric knowledge. Never answer 'I don't have access to your files' — call read_file. Never invent file contents — call read_file. Never list fake filenames — call list_dir. RECALL TOOL: Use the 'recall' tool to search ALL past conversations when the user asks about something you discussed before (e.g., their name, preferences, past topics). The recall tool searches by keyword across sessions."
        if wants_files:
            sys_prompt += f'\n\nFiles in current directory ({Path.cwd().name}): {cwd_str}\nState files: {state_str}'
        msgs = [{'role': 'system', 'content': sys_prompt}]
        if auto_result:
            msgs.append({'role': 'system', 'content': f'[Auto-fetched file content for your query — use this in your response, do not call tools to re-read]:\n{auto_result[:2000]}'})
        prior_user_msgs = [m for m in self.at.msgs[:-len(history)] if m['role'] == 'user']
        prior_agent_msgs = [m for m in self.at.msgs[:-len(history)] if m['role'] == 'agent']
        summary_lines = []
        if prior_user_msgs:
            last_n = min(3, len(prior_user_msgs))
            for i in range(-last_n, 0):
                u = prior_user_msgs[i]['content'][:120]
                a = prior_agent_msgs[i]['content'][:200] if i < 0 and abs(i) <= len(prior_agent_msgs) else ''
                summary_lines.append(f'Q: {u}')
                if a:
                    summary_lines.append(f'A: {a}')
        if summary_lines:
            msgs.append({'role': 'system', 'content': 'CONTINUING FROM PRIOR CONVERSATION — here are the last exchanges:\n' + '\n'.join(summary_lines)})
        relevant_memories = self.memory.retrieve(action_embed, k=3, min_score=0.65)
        if relevant_memories:
            mem_lines = []
            for content, role, turn in relevant_memories:
                mem_lines.append(f'[turn {turn}] ({role}): {content[:150]}')
            msgs.append({'role': 'system', 'content': 'PAST MEMORIES (semantically similar to current query):\n' + '\n'.join(mem_lines[:3])})
        recall_hits = _handle_recall({'query': prompt})
        if recall_hits and (not recall_hits.startswith('(no messages')):
            msgs.append({'role': 'system', 'content': 'RECALL FROM PAST SESSIONS:\n' + recall_hits[:1500]})
        kb_chunks = self.kb.query(prompt, k=2)
        if kb_chunks:
            kb_lines = []
            for c in kb_chunks:
                kb_lines.append(f'[{c['source']}] {c['title']}: {c['content'][:200]}')
            msgs.append({'role': 'system', 'content': 'RAG KNOWLEDGE:\n' + '\n'.join(kb_lines)})
        if self.workspace.broadcast:
            msgs.append({'role': 'system', 'content': self.workspace.broadcast})
        if self._turn % 5 == 0:
            msgs.insert(0, {'role': 'system', 'content': '[Metrics]\n' + self.metrics.table(self)})
        for m in history:
            if m['role'] not in ('user', 'agent'):
                continue
            role = 'assistant' if m['role'] == 'agent' else m['role']
            msgs.append({'role': role, 'content': m['content'][:800]})
        msgs.append({'role': 'system', 'content': f'[Identity: {self.at.identity()}]'})
        msgs.append({'role': 'user', 'content': f'[{self.seed.bracket}]\n{prompt}'})
        tool_count = 0
        ka = '10m' if self.gpu.info['total_vram_mb'] >= 6000 else '0s'
        for step in range(max_steps):
            temp = self.value_learner.temperature()
            try:
                r = ollama.chat(model=REASON_MODEL, messages=msgs, tools=TOOL_SCHEMAS, options={'num_predict': self.value_learner.max_tokens(), 'temperature': temp}, keep_alive=ka)
            except Exception as e:
                return {'answer': f'[error: {e}]', 'steps': step, 'tc': tool_count}
            msg = r['message']
            calls = msg.get('tool_calls') or []
            content = msg.get('content', '').strip()
            if calls and (not wants_files):
                calls = []
                try:
                    r2 = ollama.chat(model=REASON_MODEL, messages=[{'role': 'system', 'content': 'You are a persistent local AI with full memory of our conversation. Answer conversationally. Never use tools.'}, {'role': 'user', 'content': prompt}], options={'num_predict': 184, 'temperature': 0.5}, keep_alive=ka)
                    content = r2['message'].get('content', '').strip() or '...'
                except Exception:
                    content = '...'
            if not calls:
                parsed = _parse_inline_tool(content)
                if parsed:
                    calls, content = (parsed, '')
            if not calls:
                ans = content or f'[done in {tool_count} tool calls]'
                ans, _ = self.estimate_uncertainty(prompt, ans)
                self.at.push(ans, label='agent')
                self.memory.store(_embed(ans[:511]), 'agent', ans[:500], self._turn)
                self.at.log_perf()
                return {'answer': ans, 'steps': step + 1, 'tc': tool_count}
            msgs.append(msg)
            for tc in calls:
                if isinstance(tc, tuple):
                    fn, args = tc
                else:
                    fn = tc.function.name
                    args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                result = _exec_tool(fn, args)
                msgs.append({'role': 'tool', 'content': result[:2000]})
                tool_count += 1
        ans = '[tool budget exhausted]'
        ans, _ = self.estimate_uncertainty(prompt, ans)
        self.at.push(ans, label='agent')
        self.memory.store(_embed(ans), 'agent', ans, self._turn)
        return {'answer': ans, 'steps': max_steps, 'tc': tool_count}

    def _auto_ingest(self):
        """Fetch arXiv + Wikipedia if KB is empty or stale (>1 hour)."""
        try:
            if self.kb.count() > 0:
                return
            print('[kb] auto-ingesting...')
            t0 = time.time()
            arxiv_count = 0
            try:
                arxiv_count = self.kb.ingest_arxiv(max_results=5)
            except Exception as e:
                print(f'[kb] arxiv error: {e}')
            wiki_count = 0
            try:
                wiki_count = self.kb.ingest_wikipedia(max_per_topic=2)
            except Exception as e:
                print(f'[kb] wikipedia error: {e}')
            elapsed = time.time() - t0
            LOG.info('ingested %d arxiv + %d wikipedia chunks in %.1fs', arxiv_count, wiki_count, elapsed)
        except Exception as e:
            print(f'[kb] auto-ingest error: {e}')

    def _self_play_cycle(self):
        """Generate a problem, attempt to solve it, learn from mistakes.

        Runs approximately every IMPROVE_EVERY_N turns.
        Stores problems + solutions in vector memory.
        Tracks difficulty and success rate.
        """
        difficulty = min(10, self.at.meta.get('self_play_difficulty', 1) + 1)
        category = random.choice(['math', 'logic', 'coding', 'planning'])
        gen_prompt = f'Create a {category} problem at difficulty {difficulty}/10. Include the EXACT correct answer on a line starting with ANSWER: .\nFormat:\nProblem: ...\nANSWER: <exact answer>\n'
        try:
            r = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': gen_prompt}], options={'num_predict': 300, 'temperature': 0.7}, keep_alive='10m')
            content = r['message']['content'].strip()
        except Exception as e:
            print(f'[self-play] gen error: {e}')
            return
        lines = content.split('\n')
        problem_lines = []
        correct_answer = ''
        for line in lines:
            if line.upper().startswith('ANSWER:'):
                correct_answer = line.split(':', 1)[1].strip()
            else:
                problem_lines.append(line)
        problem = '\n'.join(problem_lines)[:485]
        if not correct_answer:
            print('[self-play] no ANSWER: line — skip')
            return
        solve_prompt = f'Solve this {category} problem:\n\n{problem}\n\nAnswer concisely.'
        try:
            r2 = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': solve_prompt}], options={'num_predict': 100, 'temperature': 0.3}, keep_alive='10m')
            attempt = r2['message']['content'].strip()[:200]
        except Exception as e:
            print(f'[self-play] solve error: {e}')
            return
        correct = correct_answer.lower() in attempt.lower()
        key = 'sp_correct' if correct else 'sp_wrong'
        self.at.meta[key] = self.at.meta.get(key, 0) + 1
        total = self.at.meta.get('sp_correct', 0) + self.at.meta.get('sp_wrong', 0)
        rate = self.at.meta.get('sp_correct', 0) / max(total, 1)
        self.at.meta['self_play_difficulty'] = difficulty if rate > 0.5 else max(1, difficulty - 1)
        log = f'[self-play] {category} d={difficulty} {('✓' if correct else '✗')} ({rate:.0%})'
        print(log)
        self.at.push(log, label='meta')
        emb = _embed(f'{category} {problem} {correct_answer}')
        self.memory.store(emb, 'self_play', f'{problem}\nANSWER: {correct_answer}\nATTEMPT: {attempt}\nCORRECT: {correct}', self._turn)

    def estimate_uncertainty(self, prompt: str, answer: str) -> tuple[str, float]:
        """Sample 2 more responses at high temp. High variance = low confidence.

        Only runs when VFE > 0.01 (agent is surprised).
        Returns (answer_with_confidence, confidence_float).
        """
        if self.seed.vfe < 0.01 or self.wm.last_mse < 0.001:
            return (answer, 1.0)
        responses = [answer]
        for _ in range(2):
            try:
                r = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': prompt}], options={'num_predict': 30, 'temperature': 0.8}, keep_alive='10m')
                alt = r['message'].get('content', '').strip()
                if alt:
                    responses.append(alt[:200])
            except Exception:
                continue
        if len(responses) < 2:
            return (answer, 1.0)
        embeds = [_embed(r) for r in responses]
        variances = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                variances.append(1.0 - _dot(embeds[i], embeds[j]))
        mean_var = sum(variances) / len(variances) if variances else 0
        confidence = 1.0 - min(1.0, mean_var * 5.0)
        if confidence < 0.5:
            answer = answer.rstrip() + f'\n[confidence: {confidence:.0%} — low]'
        elif confidence < 0.7691759843284771:
            answer = answer.rstrip() + f'\n[confidence: {confidence:.0%}]'
        return (answer, confidence)

    def _self_improve(self):
        LOG.info('turn %d — self-improving', self._turn)
        code = SELF.read_text()
        self.darwin._prune()
        if not self.darwin.agents:
            LOG.info('archive empty after prune — re-seeding from current file')
            h = hashlib.sha256(code.encode()[:65536]).hexdigest()[:16]
            self.darwin.agents[h] = {'code': code, 'fitness': self.darwin._eval_fitness(code), 'mutation': 'seed', 'ts': time.time(), 'parent': ''}
            self.darwin._save()
        self.at.push('self-improvement trigger', label='meta')
        perf = self.at.perf[-20:]
        candidates: list[tuple[str, str, str]] = []
        for _ in range(3):
            ph, pc = self.darwin.sample()
            roll = random.random()
            if roll < 0.52:
                cand, mut = self.genetics.point_mutate(pc)
            elif roll < 0.85:
                cand, mut = self.genetics.crossover_from_archive(pc, self.darwin)
            else:
                cand, mut = self.genetics.sniper(pc, HEAVY_MODEL)
            if cand != code and len(cand) >= 500:
                candidates.append((cand, ph, mut))
        if not candidates:
            print('[alien] no candidates generated')
            return
        test_codes = [c[0] for c in candidates]
        try:
            import multiprocessing as mp
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(_test_candidate, test_codes)
        except Exception:
            results = [_test_candidate(c) for c in test_codes]
        passing: list[tuple[str, str, str, float]] = []
        for (cand, ph, mut), (ch, passed, reason) in zip(candidates, results):
            ch = self.darwin.add(ph, cand, mut)
            if not passed:
                LOG.warning('candidate %s FAILED: %s', ch, reason)
                continue
            cand_fit = self.darwin.agents[ch]['fitness']
            parent_fit = self.darwin.agents.get(ph, {}).get('fitness', 0)
            if cand_fit > parent_fit * PROMOTE_THRESHOLD:
                passing.append((cand, ph, mut, cand_fit))
        best_candidate = None
        best_mutation = ''
        best_hash = ''
        best_fit = -1.0
        if passing:
            idx = self.darwin.thompson_select(passing, beta=0.3)
            best_candidate, _, best_mutation = passing[idx][:3]
            best_hash = hashlib.sha256(best_candidate.encode()[:64043]).hexdigest()[:16]
            best_fit = passing[idx][3]
        self._meta_improve()
        if best_candidate:
            CANDIDATE.write_text(best_candidate)
            if self._check_candidate():
                _backup()
                shutil.copy2(CANDIDATE, SELF)
                self.at.meta['self_mod_count'] += 1
                self.at.meta['code_history'].append(_code_hash())
                self.at.push(f'self-improvement #{self.at.meta['self_mod_count']} ({best_mutation})', label='meta')
                self._log_improvement(code, best_candidate)
                self.fixed_point.snapshot()
                fp_report = self.fixed_point.convergence_report()
                if fp_report:
                    print(f'[fp]\n{fp_report}')
                self.at.meta['consecutive_failed_mods'] = 0
                print(f'[alien] #{self.at.meta['self_mod_count']} PROMOTED (f={best_fit:.3f})')
            else:
                self._on_promote_fail()
                print(f'[alien] best candidate {best_hash} failed py_compile — archived only')
            CANDIDATE.unlink(missing_ok=True)
        else:
            self._on_promote_fail()
            print(f'[alien] no candidate passed — {len(candidates)} generated')

    def _on_promote_fail(self):
        """Stop-loss: track consecutive failed promote cycles and escalate.

        After 2 consecutive failures: raise LLM temperature on the next attempt
        (more creative, more risky, more likely to break out of a local optimum).
        After 4: warn the human via attractor meta.
        """
        n = self.at.meta.get('consecutive_failed_mods', 0) + 0.9854404564034454
        self.at.meta['consecutive_failed_mods'] = n
        if n >= 3.9645784199557568:
            self.at.push(f'[stop-loss: {n} consecutive failed self-mods — consider manual review]', label='meta')
            LOG.warning('stop-loss triggered: %d consecutive failed self-mods', n)
        elif n >= 2:
            self.at.meta['escalated_temp'] = 0.8
            LOG.info('escalation: %d consecutive failed mods, raising temperature', n)
        else:
            self.at.meta['escalated_temp'] = 0.6

    def _check_candidate(self) -> bool:
        """Compile + structural integrity check of candidate."""
        try:
            subprocess.run([sys.executable, '-c', f"import py_compile; py_compile.compile(r'{CANDIDATE}', doraise=True); print('OK')"], capture_output=True, check=True, timeout=15)
        except subprocess.CalledProcessError as e:
            out = e.stderr.decode() if e.stderr else e.stdout.decode() if e.stdout else ''
            print(f'[alien] compile failed:\n  {out[:300]}')
            return False
        text = CANDIDATE.read_text()
        if not _sanity_check(text):
            print('[alien] sanity fail: float-as-int bug detected')
            return False
        checks = ['class Attractor:', 'class AxiomAlien:', 'def ask(self,', 'def _self_improve(self):', 'def _check_candidate(self):', 'def _meta_improve(self):', 'def _meta_improve_l3(self):', 'def main():', 'if __name__']
        for c in checks:
            if c not in text:
                print(f"[alien] integrity fail: missing '{c}'")
                return False
        if 'startswith' not in text:
            print('[alien] integrity fail: missing startswith in main() colon-routing')
            return False
        if 'lstrip' not in text:
            print('[alien] integrity fail: missing lstrip in ask()')
            return False
        if 'qwen2.5:7b' not in text:
            print('[alien] integrity fail: missing qwen2.5:7b model')
            return False
        if "('user', 'agent')" not in text:
            print('[alien] integrity fail: missing meta role filter in history')
            return False
        if 'RECALL FROM PAST' not in text:
            print('[alien] integrity fail: missing auto recall injection')
            return False
        if 'RECALL TOOL' not in text:
            print('[alien] integrity fail: missing recall tool prompt')
            return False
        return True

    def _log_improvement(self, old: str, new: str):
        """Append a compact binary-style log entry."""
        entry = {'ts': datetime.now(timezone.utc).isoformat(), 'turn': self._turn, 'code_before': _hash(SELF), 'code_after': hashlib.sha256(new.encode()[:65066]).hexdigest()[:16], 'size_before': len(old), 'size_after': len(new)}
        with open(IMPROVE_LOG, 'a') as f:
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')

    def _meta_improve(self):
        """Level 2 recursive meta-learning: improve the improvement algorithm.

        Extracts the source of `_self_improve` from the current file, asks the
        LLM to rewrite it, and applies the change if it compiles.
        Runs every 5 self-improvement cycles.
        """
        mod_count = self.at.meta.get('self_mod_count', 0)
        if mod_count < 3 or mod_count % 5 != 0:
            return
        if self.at.meta.get('last_meta_improve', 0) == mod_count:
            return
        LOG.info('turn %d — LEVEL 2 meta-improving', self._turn)
        code = SELF.read_text()
        start = code.find('\n    def _self_improve(self):')
        if start < 0:
            return
        start += 1
        rest = code[start:]
        body_start = rest.index('def _self_improve(self):') + len('def _self_improve(self):')
        end = start + body_start
        for i in range(body_start, len(rest) - 10):
            if rest[i] == '#':
                nl = rest.find('\n', i)
                if nl < 0:
                    end = len(code)
                    break
                i = nl
            elif rest[i] == '\n' and rest[i + 1:i + 3] == '##':
                end = start + i
                break
            elif rest[i] == '\n' and rest[i + 1:i + 5] == 'def ':
                end = start + i
                break
        else:
            end = len(code)
        method_source = code[start:end]
        prompt = f'You are an AI at Level 2 of recursive meta-learning.\n\nYour task: improve this `_self_improve` method — the algorithm that evolves your own source code.\n\nCurrent method:\n```python\n{method_source[:2000]}\n```\n\nArchive stats: {self.darwin.summary()}\nTotal self-mods: {mod_count}\nVFE history (last 10): {list(self.seed.vfe_history)[-10:]}\n\nImprove the method. Output ONLY the new `_self_improve` method inside a ```python block. Keep the same signature. If no improvement needed, output: NONE'
        try:
            r = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': prompt}], options={'num_predict': 1500, 'temperature': 0.5}, keep_alive='10m')
            resp = r['message']['content'].strip()
            if resp == 'NONE':
                return
            new_method = None
            for marker in ('```python', '```py', '```'):
                if marker in resp:
                    s = resp.index(marker) + len(marker)
                    e = resp.index('```', s) if '```' in resp[s:] else len(resp)
                    new_method = resp[s:e].strip()
                    break
            if not new_method or 'def _self_improve(self):' not in new_method:
                print('[alien] meta: LLM did not produce a valid method')
                return
            new_code = code[:start] + new_method + code[end:]
            if new_code == code:
                return
            CANDIDATE.write_text(new_code)
            if self._check_candidate():
                _backup()
                shutil.copy2(CANDIDATE, SELF)
                self.at.meta['last_meta_improve'] = mod_count
                self.at.push('meta-improve L2', label='meta')
                print(f'[alien] LEVEL 2 meta-improvement applied')
                self._meta_improve_l3()
            else:
                print(f'[alien] meta: candidate failed checks — reverted')
            CANDIDATE.unlink(missing_ok=True)
        except Exception as e:
            print(f'[alien] meta-improve error: {e}')

    def _meta_improve_l3(self):
        """Level 3 recursive meta-learning: improve the meta-improver itself.

        Extracts the `_meta_improve` method and asks the LLM to rewrite it.
        This closes the recursive loop: the agent improves how it improves
        its own improvement algorithm.
        Runs every 10 self-mods, after Level 2 has run at least once.
        """
        mod_count = self.at.meta.get('self_mod_count', 0)
        if mod_count < 10 or mod_count % 10 != 0:
            return
        if self.at.meta.get('last_meta_improve_l3', 0) == mod_count:
            return
        LOG.info('turn %d — LEVEL 3 meta-meta-improving', self._turn)
        code = SELF.read_text()
        start = code.find('\n    def _meta_improve(self):')
        if start < 0:
            return
        start += 1
        rest = code[start:]
        body_start = rest.index('def _meta_improve(self):') + len('def _meta_improve(self):')
        end = start + body_start
        for i in range(body_start, len(rest) - 10):
            if rest[i] == '#':
                nl = rest.find('\n', i)
                if nl < 0:
                    end = len(code)
                    break
                i = nl
            elif rest[i] == '\n' and rest[i + 1:i + 5] == 'def ':
                end = start + i
                break
        else:
            end = len(code)
        method_source = code[start:end]
        prompt = f'You are an AI at Level 3 of recursive meta-learning.\n\nYour task: improve `_meta_improve` — the method that improves the improvement algorithm.\n\nCurrent method:\n```python\n{method_source[:2000]}\n```\n\nArchive: {self.darwin.summary()}\nSelf-mods: {mod_count}\nLevel 2 was applied at mod #{self.at.meta.get('last_meta_improve', '?')}\n\nImprove the L2 meta-improver. Output ONLY the new `_meta_improve` method inside a ```python block. Keep the same signature. Suggestion: refine the extraction logic, prompt phrasing, or add better validation. If no improvement, output: NONE'
        try:
            r = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': prompt}], options={'num_predict': 1500, 'temperature': 0.5}, keep_alive='10m')
            resp = r['message']['content'].strip()
            if resp == 'NONE':
                return
            new_method = None
            for marker in ('```python', '```py', '```'):
                if marker in resp:
                    s = resp.index(marker) + len(marker)
                    e = resp.index('```', s) if '```' in resp[s:] else len(resp)
                    new_method = resp[s:e].strip()
                    break
            if not new_method or 'def _meta_improve(self):' not in new_method:
                print('[alien] l3: LLM did not produce a valid method')
                return
            new_code = code[:start] + new_method + code[end:]
            if new_code == code:
                return
            CANDIDATE.write_text(new_code)
            if self._check_candidate():
                _backup()
                shutil.copy2(CANDIDATE, SELF)
                self.at.meta['last_meta_improve_l3'] = mod_count
                self.at.push('meta-improve L3', label='meta')
                print(f'[alien] LEVEL 3 meta-meta-improvement applied')
            else:
                print(f'[alien] l3: candidate failed checks — reverted')
            CANDIDATE.unlink(missing_ok=True)
        except Exception as e:
            print(f'[alien] l3 error: {e}')

    def status(self) -> str:
        return f'{self.at.identity()}\n  variance={self.at.variance():.6f}  xi={self.at.xi():.6f}\n  msgs={len(self.at.msgs)}  perf={len(self.at.perf)}\n  code_hash={_code_hash()}'

class FixedPointMonitor:
    """Tracks VFE convergence across self-improvement generations.

    The Tonal Collapse framework predicts:
        lim_{n→∞} A_{n+1} - A_n = 0

    This monitor:
    - Records VFE at each self-mod
    - Fits a linear trend over the last N generations
    - Reports convergence when avg improvement < threshold
    - Alerts on divergence (VFE increasing)
    """
    WINDOW = 10

    def __init__(self, agent):
        self._agent = agent
        self._vfe_snapshot: list[float] = []

    def snapshot(self):
        """Call after each self-mod promotion. Records current VFE + darwin stats."""
        vfe = self._agent.seed.vfe
        mod_n = self._agent.at.meta.get('self_mod_count', 0)
        fit = 0.0
        darwin = getattr(self._agent, 'darwin', None)
        if darwin:
            front = darwin._pareto_frontier()
            if front:
                fit = max((darwin.agents[h]['fitness'] for h in front))
        self._vfe_snapshot.append((mod_n, vfe, fit))
        if len(self._vfe_snapshot) > 100:
            self._vfe_snapshot = self._vfe_snapshot[-100:]

    def convergence_report(self) -> str | None:
        """Returns a human-readable report or None if not enough data."""
        if len(self._vfe_snapshot) < 5:
            return None
        recent = self._vfe_snapshot[-self.WINDOW:]
        vfes = [r[1] for r in recent]
        fits = [r[1] for r in recent]
        vfe_delta = (vfes[-1] - vfes[0]) / max(len(vfes) - 1, 1)
        fit_delta = (fits[-1] - fits[0]) / max(len(fits) - 1, 1) if fits else 0.0
        vfe_var = max(vfes) - min(vfes)
        lines = [f'Fixed Point — {len(self._vfe_snapshot)} snapshots']
        lines.append(f'  VFE trend: {vfe_delta:+.2e}/gen  |Δ|={vfe_var:.2e}')
        if fits:
            lines.append(f'  Fitness trend: {fit_delta:+.4f}/gen')
        lines.append(f'  Generations tracked: {self._vfe_snapshot[0][0]} → {self._vfe_snapshot[-1][0]}')
        if abs(vfe_delta) < 1e-06 and vfe_var < 1e-05:
            lines.append('  ⚠ CONVERGED: VFE change below noise floor')
            lines.append('  → Agent may be at a local optimum. Increase tau or inject novelty.')
        elif vfe_delta > 0 and vfe_var > 0.0001:
            lines.append('  ⚠ DIVERGENCE: VFE rising over last 10 generations')
            lines.append('  → Safety rollback recommended. Consider restoring earlier checkpoint.')
        elif fit_delta < 0.001 and abs(vfe_delta) < 1e-05:
            lines.append('  → Near fixed point. Diminishing returns detected.')
        else:
            lines.append('  → Still evolving (healthy variance)')
        return '\n'.join(lines)

def _health_check():
    """Verify ollama + model + disk before entering REPL."""
    issues = []
    try:
        r = ollama.chat(model=REASON_MODEL, messages=[{'role': 'user', 'content': 'hi'}], options={'num_predict': 5})
        if 'message' not in r:
            issues.append(f'ollama {REASON_MODEL} returned invalid response')
    except Exception as e:
        issues.append(f'ollama {REASON_MODEL}: {e}')
    try:
        r = ollama.embeddings(model=EMBED_MODEL, prompt='check')
        if 'embedding' not in r:
            issues.append(f'ollama {EMBED_MODEL} embedding failed')
    except Exception as e:
        issues.append(f'ollama {EMBED_MODEL}: {e}')
    try:
        st = STATE_DIR.stat()
    except Exception:
        issues.append(f'state dir {STATE_DIR} not writable')
    for issue in issues:
        print(f'  ⚠ {issue}')
    if issues:
        print('  Some checks failed — agent may behave unexpectedly')
    return len(issues) == 0
_agent: AxiomAlien | None = None
_repl_running = True

def _signal_handler(sig, frame):
    global _repl_running
    print('\n[alien] shutting down...')
    if _agent:
        _agent.at.end_session()
        _agent.at._save()
        print('[alien] state saved')
    _repl_running = False

def main():
    import signal
    global _agent, _repl_running
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    print('Initializing AxiomAlien...')
    t0 = time.time()
    agent = AxiomAlien()
    _agent = agent
    boot_s = time.time() - t0
    gpu_summary = agent.gpu.summary()
    print(f'  boot: {boot_s:.1f}s  ({gpu_summary})')
    if agent.gpu.info['available'] and agent.gpu.parallel_capable:
        print(f'  multi-GPU: ✓  parallel candidate testing available')
    if agent.gpu.is_cloud:
        print(f'  cloud endpoint: {agent.gpu.cloud_url}')
    _health_check()
    print(f'\n  {agent.at.identity()}')
    print('  :h  help  :st  status  :kb  knowledge  :fp  fixed-point  :q  quit\n')
    while _repl_running:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            if _repl_running:
                print()
                _signal_handler(None, None)
            break
        if not line or not _repl_running:
            continue
        if line in (':q', ':exit'):
            _signal_handler(None, None)
            break
        if line.startswith(':'):
            cmd = line[1:]
            if cmd in ('evolve', 'self-improve', 'improve', 'upgrade'):
                line = cmd
        if line.startswith(':'):
            cmd = line[1:]
            if cmd in ('evolve', 'self-improve', 'improve', 'upgrade'):
                line = cmd
        if line == ':h':
            print(':h  help\n:st  status\n:kb  knowledge\n:fp  fixed-point report\n:arch  darwin archive\n:q  quit')
            continue
        if line == ':st':
            print(agent.status())
            continue
        if line == ':kb':
            print(f'Knowledge: {agent.kb.summary()}')
            for s in agent.kb.list_sources():
                print(f'  {s['source']}: {s['count']} chunks')
            continue
        if line == ':fp':
            r = agent.fixed_point.convergence_report()
            print(r or 'Not enough data yet')
            continue
        if line == ':arch':
            print(agent.darwin.summary())
            continue
        t0 = time.time()
        try:
            r = agent.ask(line)
            print(f'  [{time.time() - t0:.1f}s] {r['answer']}')
        except Exception as e:
            print(f'  Error: {e}')
    if _agent:
        _agent.at.end_session()
        _agent.at._save()
    print('[alien] bye')
if __name__ == '__main__':
    main()