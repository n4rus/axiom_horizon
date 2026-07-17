"""task_ingestion.py — AGI task ingestion pipeline for crypto revenue.

The daemon searches for, selects, and completes crypto-paying tasks
(bounties, audits, coding, anything legal) to fill the wallet.

Architecture:
  TaskSource      → fetches available tasks from a platform
  TaskFilter      → scores tasks by AGI feasibility + reward
  TaskExecutor    → uses the LLM to generate and submit solutions
  TaskPipeline    → orchestrates: fetch → select → execute → track

Platforms (tier 1 — public APIs, no auth):
  - Code4rena: audit contests (public API, prize pools)
  - Gitcoin: legacy grants/bounties (public API)
  - Dework: DAO task boards (public)
  - Cantina: audit contests (public)

Each cycle, the daemon can:
  1. Check pipeline for pending tasks
  2. Work on the best-scored task
  3. Submit solution
  4. Log result + payment tracking
"""

from __future__ import annotations
import json, math, os, random, re, time, threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import sys
sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).parent
STATE = BASE / '.axiom_state'
TASK_DB = STATE / 'tasks.json'
EARNINGS_LOG = STATE / 'earnings.json'

WALLET = '0x8f36105eE73b4Aadc0Cf5301A756378F49eB0eb5'
DEST_WALLET = '0xD0b864545a5b6CA2654e46bEcb5602946E3AEbb7'

# --- Data models ---

@dataclass
class Task:
    id: str
    source: str
    title: str
    description: str
    reward: str
    reward_usd: float
    url: str
    tags: list[str] = field(default_factory=list)
    difficulty: float = 0.5
    submitted: bool = False
    completed: bool = False
    payment_tx: str = ''
    grounded: bool = False  # set True if the submitted fix passed compile+import+critic
    fetched_at: float = 0.0
    _score: float = 0.0  # computed, shadows method via property-like convention

    def score(self, agi_capabilities: list[str]) -> float:
        """Score how suitable this task is for the AGI (0-1). Higher = better."""
        score = 0.0
        # Prefer tasks with higher USD reward
        score += min(self.reward_usd / 1000.0, 0.3)
        # Prefer tasks matching AGI capabilities (coding, analysis, writing)
        tag_overlap = sum(1 for t in self.tags if t in agi_capabilities)
        score += min(tag_overlap * 0.1, 0.3)
        # Prefer easier tasks (lower difficulty)
        score += (1.0 - self.difficulty) * 0.2
        # Prefer tasks with good description (length = detail)
        desc_score = min(len(self.description) / 2000.0, 0.2)
        score += desc_score
        return min(score, 1.0)


# --- Task Sources ---

class TaskSource:
    name = 'base'
    def fetch(self) -> list[Task]:
        return []


class GitcoinSource(TaskSource):
    name = 'gitcoin'
    API_URL = 'https://grants-stack-indexer.gitcoindao.com/api/grant-applications/search?limit=10&orderBy=created_at_desc'
    LEGACY_API = 'https://gitcoin.co/api/v0.1/bounties/?network=mainnet&page_size=10'

    def fetch(self) -> list[Task]:
        tasks = []
        # Try legacy bounties API
        try:
            req = Request(self.LEGACY_API, headers={'User-Agent': 'Axiom/1.0'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            for b in data:
                title = b.get('title', '') or b.get('standard_bounties', {}).get('title', '')
                desc = b.get('description', '') or b.get('standard_bounties', {}).get('description', '')
                reward_str = '0'
                try:
                    reward_str = b.get('fulfillment_accepted_amount', '0')
                except Exception:
                    pass
                reward = float(reward_str) if reward_str else 0.0
                tid = b.get('id', '') or b.get('standard_bounties', {}).get('id', '')
                tasks.append(Task(
                    id=f"gitcoin_{tid}",
                    source='gitcoin',
                    title=title[:200],
                    description=desc[:2000],
                    reward=f"{reward} ETH",
                    reward_usd=reward * 1800.0,
                    url=f"https://gitcoin.co/issue/{b.get('github_url', '')}",
                    tags=['coding', 'open_source', 'solidity'],
                    difficulty=0.6,
                    fetched_at=time.time(),
                ))
        except Exception:
            pass
        return tasks


class Code4renaSource(TaskSource):
    name = 'code4rena'
    API_URL = 'https://code4rena.com/api/v1/contests'

    def fetch(self) -> list[Task]:
        tasks = []
        try:
            req = Request(self.API_URL, headers={'User-Agent': 'Axiom/1.0', 'Accept': 'application/json'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            contests = data if isinstance(data, list) else data.get('data', [])
            for c in contests[:20]:
                title = c.get('title', '') or c.get('name', '')
                desc = c.get('description', '') or c.get('details', '')
                prize = c.get('prize', '') or c.get('total_prize', '')
                try:
                    prize_usd = float(prize.replace('$', '').replace(',', ''))
                except (ValueError, AttributeError):
                    prize_usd = 0.0
                tid = c.get('id', random.randint(0, 999999))
                tasks.append(Task(
                    id=f"c4_{tid}",
                    source='code4rena',
                    title=title[:200],
                    description=desc[:2000],
                    reward=f"${prize_usd:,.0f}" if prize_usd else prize,
                    reward_usd=prize_usd,
                    url=f"https://code4rena.com/contests/{tid}",
                    tags=['audit', 'solidity', 'security'],
                    difficulty=0.8,
                    fetched_at=time.time(),
                ))
        except Exception:
            pass
        return tasks


class GitHubBountySource(TaskSource):
    name = 'github'
    API_URL = 'https://api.github.com/search/issues?q=label:bounty+state:open&sort=created&order=desc&per_page=20'

    def fetch(self) -> list[Task]:
        tasks = []
        try:
            req = Request(self.API_URL, headers={
                'User-Agent': 'Axiom/1.0',
                'Accept': 'application/vnd.github.v3+json',
            })
            with urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            for item in data.get('items', []):
                title = item.get('title', '')
                body = item.get('body', '') or ''
                url = item.get('html_url', '')
                repo = item.get('repository_url', '').split('/')[-1] if item.get('repository_url') else ''
                labels = [l.get('name', '') for l in item.get('labels', [])]
                # Extract bounty amount from labels or body
                reward_usd = 0.0
                for label in labels:
                    try:
                        if label.lower().startswith('bounty'):
                            nums = re.findall(r'[\d,.]+', label)
                            if nums:
                                reward_usd = float(nums[0].replace(',', ''))
                    except:
                        pass
                # Also try to find dollar amounts in body
                if reward_usd == 0.0:
                    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)\s*(?:USD|usd)?', body[:1000])
                    if matches:
                        try:
                            reward_usd = float(matches[0].replace(',', ''))
                        except:
                            pass
                # Try to find ETH amounts
                reward_str = f"${reward_usd:,.0f}" if reward_usd > 0 else 'varies'
                tasks.append(Task(
                    id=f"github_{item.get('id', random.randint(0,999999))}",
                    source='github',
                    title=title[:200],
                    description=body[:2000],
                    reward=reward_str,
                    reward_usd=reward_usd,
                    url=url,
                    tags=['coding'] + labels[:5],
                    difficulty=0.5,
                    fetched_at=time.time(),
                ))
        except Exception as e:
            pass
        return tasks


class DeworkSource(TaskSource):
    name = 'dework'
    API_URL = 'https://api.dework.xyz/v1/tasks?status=open&limit=10'

    def fetch(self) -> list[Task]:
        tasks = []
        try:
            req = Request(self.API_URL, headers={'User-Agent': 'Axiom/1.0'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            raw = data if isinstance(data, list) else data.get('data', [])
            for t in raw:
                title = t.get('name', '') or t.get('title', '')
                desc = t.get('description', '') or t.get('body', '')
                reward_str = 'varies'
                reward_usd = 0.0
                try:
                    reward_str = t.get('reward', {}).get('amount', '0')
                    reward_usd = float(reward_str) if reward_str else 0.0
                except:
                    pass
                tasks.append(Task(
                    id=f"dework_{t.get('id', '')}",
                    source='dework',
                    title=title[:200],
                    description=desc[:2000],
                    reward=reward_str,
                    reward_usd=reward_usd,
                    url=f"https://app.dework.xyz/task/{t.get('id', '')}",
                    tags=t.get('tags', ['general']),
                    difficulty=0.5,
                    fetched_at=time.time(),
                ))
        except Exception:
            pass
        return tasks


class CantinaSource(TaskSource):
    name = 'cantina'
    API_URL = 'https://api.cantina.xyz/v1/contests?status=active'

    def fetch(self) -> list[Task]:
        tasks = []
        try:
            req = Request(self.API_URL, headers={'User-Agent': 'Axiom/1.0'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            contests = data if isinstance(data, list) else data.get('data', [])
            for c in contests[:10]:
                title = c.get('name', '') or c.get('title', '')
                prize_pool = c.get('prizePool', 0)
                tasks.append(Task(
                    id=f"cantina_{c.get('id', '')}",
                    source='cantina',
                    title=title[:200],
                    description=c.get('description', '')[:2000],
                    reward=f"${prize_pool:,}" if prize_pool else 'varies',
                    reward_usd=float(prize_pool) if prize_pool else 0.0,
                    url=f"https://cantina.xyz/contests/{c.get('id', '')}",
                    tags=['audit', 'security'],
                    difficulty=0.85,
                    fetched_at=time.time(),
                ))
        except Exception:
            pass
        return tasks


class AlgoraSource(TaskSource):
    name = 'algora'
    API_URL = 'https://api.algora.io/v1/bounties?status=open&limit=10'

    def fetch(self) -> list[Task]:
        tasks = []
        try:
            req = Request(self.API_URL, headers={'User-Agent': 'Axiom/1.0', 'Accept': 'application/json'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            items = data if isinstance(data, list) else data.get('bounties', [])
            for item in items[:10]:
                title = item.get('title', '') or item.get('name', '')
                body = item.get('description', '') or item.get('body', '')
                reward = item.get('reward', 0) or item.get('amount', 0)
                try:
                    reward_usd = float(reward)
                except:
                    reward_usd = 0.0
                tasks.append(Task(
                    id=f"algora_{item.get('id', '')}",
                    source='algora',
                    title=title[:200],
                    description=body[:2000],
                    reward=f"${reward_usd:,.0f}" if reward_usd else 'varies',
                    reward_usd=reward_usd,
                    url=item.get('url', '') or item.get('html_url', ''),
                    tags=['coding', 'open_source'],
                    difficulty=0.5,
                    fetched_at=time.time(),
                ))
        except Exception:
            pass
        return tasks


class RealBountySource(TaskSource):
    """Fetches real bounties from verified repos via GitHub API search."""
    name = 'real_bounty'
    QUERIES = [
        'commenter:app/algora-pbc is:issue is:open',
        'repo:tenstorrent/tt-metal label:bounty state:open',
        'repo:tscircuit/jlcsearch label:bounty state:open',
        'repo:tscircuit/autorouting label:bounty state:open',
        '"$500" OR "$1000" OR "$1500" OR "$2000" label:bounty state:open',
    ]
    SKIP_REPOS = {'zhangjiayang6835-cyber/ai-research', 'UnsafeLabs/Bounty-Hunters',
                  'SecureBananaLabs/bug-bounty', 'rohitdash08/FinMind'}

    def fetch(self) -> list[Task]:
        import re as _re
        from urllib.parse import quote
        tasks = []
        seen = set()
        token = os.environ.get('GITHUB_TOKEN', '')
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if token:
            headers['Authorization'] = f'token {token}'
        for q in self.QUERIES:
            try:
                encoded_q = quote(q)
                url = f'https://api.github.com/search/issues?q={encoded_q}&sort=created&order=desc&per_page=15'
                req = Request(url, headers=headers)
                with urlopen(req, timeout=15) as r:
                    data = json.loads(r.read())
                for item in data.get('items', []):
                    if item['id'] in seen:
                        continue
                    seen.add(item['id'])
                    repo_url = item.get('repository_url', '')
                    repo = '/'.join(repo_url.split('/')[-2:]) if repo_url else ''
                    if repo in self.SKIP_REPOS:
                        continue
                    title = item.get('title', '')
                    body = (item.get('body', '') or '')[:500]
                    amounts = _re.findall(r'\$(\d[\d,]*)', title + ' ' + body)
                    max_amount = max([int(a.replace(',', '')) for a in amounts]) if amounts else 0
                    comments = item.get('comments', 0)
                    labels = [l['name'] for l in item.get('labels', [])]
                    tasks.append(Task(
                        id=f"real_{item['id']}",
                        source='real_bounty',
                        title=title[:200],
                        description=body[:2000],
                        reward=f"${max_amount:,.0f}" if max_amount else 'varies',
                        reward_usd=float(max_amount),
                        url=item['html_url'],
                        tags=labels[:5] + ['coding', 'real_bounty'],
                        difficulty=0.4 if max_amount < 200 else 0.6,
                        fetched_at=time.time(),
                    ))
            except Exception:
                pass
        return tasks


# --- Task Pipeline ---

class TaskPipeline:
    """Orchestrates task fetching, selection, execution, and tracking."""

    SOURCES: list[TaskSource] = [
        RealBountySource(),
        GitHubBountySource(),
        AlgoraSource(),
        DeworkSource(),
        # ImmunefiSource(),  # Temporarily commented out to avoid import errors
        # Code4renaSource(),  # Temporarily commented out to avoid import errors
    ]

    AGI_CAPABILITIES = [
        'coding', 'solidity', 'python', 'rust', 'javascript',
        'analysis', 'audit', 'security', 'writing', 'documentation',
        'review', 'testing', 'deployment',
    ]

    def __init__(self):
        self.tasks: list[Task] = []
        self.queue: list[Task] = []
        self.active_task: Optional[Task] = None
        self.earnings: dict[str, float] = {}
        self.load_state()

    def ingest_cycle(self):
        """Fetch tasks, score them, populate queue. Called periodically."""
        try:
            fetched = self.fetch_all()
            scored = [(t.score(self.AGI_CAPABILITIES), t) for t in self.tasks if not t.submitted and not t.completed]
            scored.sort(key=lambda x: -x[0])
            self.queue = []
            for sc, t in scored[:10]:
                t._score = sc
                self.queue.append(t)
            # Advance the active task: if there is none, OR the current one is
            # already submitted/completed, promote the top unsubmitted queued
            # task. Without this the daemon submits one PR then stalls forever.
            if not self.active_task or self.active_task.submitted or self.active_task.completed:
                if self.queue:
                    self.active_task = self.queue[0]
            self.save_state()
        except Exception:
            pass

    def load_state(self):
        try:
            if TASK_DB.exists():
                data = json.loads(TASK_DB.read_text())
                self.tasks = [Task(**t) for t in data.get('tasks', [])]
                self.queue = [Task(**t) for t in data.get('queue', [])]
                act = data.get('active')
                if act:
                    self.active_task = Task(**act)
            if EARNINGS_LOG.exists():
                self.earnings = json.loads(EARNINGS_LOG.read_text())
        except Exception:
            self.tasks = []
            self.queue = []
            self.earnings = {}

    def save_state(self):
        try:
            STATE.mkdir(exist_ok=True)
            TASK_DB.write_text(json.dumps({
                'tasks': [t.__dict__ for t in self.tasks],
                'queue': [t.__dict__ for t in self.queue],
                'active': self.active_task.__dict__ if self.active_task else None,
            }, default=str, indent=2))
            EARNINGS_LOG.write_text(json.dumps(self.earnings, indent=2))
        except Exception:
            pass

    def fetch_all(self) -> list[Task]:
        """Fetch tasks from all sources and MERGE with persisted tasks.

        Previously this REPLACED self.tasks with only the freshly-fetched set,
        which wiped curated/manually-added bounties on every ingest cycle (and
        clobbered tasks.json on save). Now we keep existing tasks — preserving
        their submitted/completed progress — and only add newly fetched ones.
        """
        all_tasks = []
        for source in self.SOURCES:
            try:
                fetched = source.fetch()
                all_tasks.extend(fetched)
            except Exception:
                continue
        # Merge: retain persisted tasks (with progress flags) and add any
        # fetched task whose id we don't already have.
        existing = {t.id: t for t in self.tasks}
        merged = list(self.tasks)
        for t in all_tasks:
            if t.id not in existing:
                existing[t.id] = t
                merged.append(t)
            else:
                # refresh descriptive fields but preserve progress flags
                old = existing[t.id]
                for f in ('title', 'description', 'url', 'reward_usd',
                          'difficulty', 'source', 'repo', 'issue_num'):
                    v = getattr(t, f, None)
                    if v:
                        setattr(old, f, v)
        self.tasks = merged
        self.save_state()
        return merged

    def best_task(self) -> Optional[Task]:
        """Score and return the best task for the AGI to work on."""
        scored = [(t.score(self.AGI_CAPABILITIES), t) for t in self.tasks if not t.submitted and not t.completed]
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        self.active_task = best
        return best

    def mark_submitted(self, task_id: str, tx_hash: str = '', grounded_ok: bool = False):
        for t in self.tasks:
            if t.id == task_id:
                t.submitted = True
                t.payment_tx = tx_hash
                t.grounded = grounded_ok
                break
        for t in self.queue:
            if t.id == task_id:
                t.submitted = True
                t.payment_tx = tx_hash
                t.grounded = grounded_ok
                break
        # Keep the active-task object in sync (it's a separate reference) so the
        # daemon doesn't re-submit an already-submitted task.
        if self.active_task and self.active_task.id == task_id:
            self.active_task.submitted = True
            self.active_task.payment_tx = tx_hash
            self.active_task.grounded = grounded_ok
        self.save_state()

    def mark_completed(self, task_id: str, usd_earned: float = 0.0):
        for t in self.tasks:
            if t.id == task_id:
                t.completed = True
                t.submitted = True
                if usd_earned > 0:
                    tx = f"task_{task_id}_{int(time.time())}"
                    self.earnings[tx] = usd_earned
                break
        self.save_state()

    def total_earned_usd(self) -> float:
        return sum(self.earnings.values())

    def summary(self) -> str:
        pending = sum(1 for t in self.tasks if not t.submitted and not t.completed)
        submitted = sum(1 for t in self.tasks if t.submitted and not t.completed)
        completed = sum(1 for t in self.tasks if t.completed)
        total = self.total_earned_usd()
        return (f"Tasks: {len(self.tasks)} total, {pending} pending, "
                f"{submitted} submitted, {completed} completed\n"
                f"Earnings: ${total:.2f} USD")

    def task_context(self) -> str:
        """Return a text summary of the current best task for the LLM."""
        task = self.best_task()
        if not task:
            return "No suitable tasks available. Try refetching."
        return (f"[TASK] {task.source}/{task.id}\n"
                f"Title: {task.title}\n"
                f"Reward: {task.reward} (${task.reward_usd:.2f})\n"
                f"URL: {task.url}\n"
                f"Difficulty: {task.difficulty:.2f}\n"
                f"Tags: {', '.join(task.tags)}\n"
                f"Description: {task.description[:1500]}")


# --- Quick test ---
if __name__ == '__main__':
    pipe = TaskPipeline()
    print("Fetching tasks...")
    tasks = pipe.fetch_all()
    print(f"Found {len(tasks)} tasks")
    for t in sorted(tasks, key=lambda x: -x.score(TaskPipeline.AGI_CAPABILITIES))[:5]:
        score = t.score(TaskPipeline.AGI_CAPABILITIES)
        print(f"  [{score:.2f}] {t.source}/{t.id[:20]}: {t.title[:50]} — {t.reward}")
    print()
    best = pipe.best_task()
    if best:
        print("Best task:")
        print(pipe.task_context())
    print()
    print(f"Total earned: ${pipe.total_earned_usd():.2f}")

# Backward-compat alias for existing axiom.py import
TaskIngestionEngine = TaskPipeline
