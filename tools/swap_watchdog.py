#!/usr/bin/env python3
"""swap_watchdog.py — domain-prober-style periodic monitor for swap pressure.

Mirrors the domain-prober algorithm (domain_probe_resumer.py):
a background monitor thread sleeps on an interval, checks a resource
measurement, and acts when a threshold is crossed.

Here the resource is *swap usage*. We have no root (swapoff/swapon and
vm.drop_caches are unavailable), so the only rootless lever to actively
reclaim swap is to reap the largest non-protected memory holder — its
pages are freed, the kernel re-locates the rest from swap back into RAM,
and SwapUsed falls. A sync flushes dirty pages first so the reclaim is
not wasted on writeback.

Usage:
  python3 tools/swap_watchdog.py --interval 30 --act-gb 4 --warn-gb 2
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import signal

# Processes that must never be reaped by the watchdog.
PROTECTED = {
    "opencode",           # the agent itself
    "wiki_absorb",        # wiki ingestion (in-flight embedding)
    "kai_bridge",         # knowledge bridge / systemd service
    "kai-",               # kai daemon / bridge service
    "llama-server",       # ollama model host (embed + chat)
    "ollama",             # ollama runner
    "systemd",            # init + services
    "sshd",               # remote sessions
    "Xorg", "xorg",       # display server
    "caja", "mate",       # desktop shell
    "watchdog",           # ourselves
}

# Desktop / non-essential GUI processes that are safe reap targets.
REAPABLE_HINTS = ("firefox", "chrom", "thunderbird", "libreoffice", "gimp",
                  "kate", "gedit", "vlc", "mpv", "steam", "discord", "zoom")


def read_swap_usage_gb():
    """Return (used_gb, total_gb) parsed from /proc/meminfo."""
    used = total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) / 1024 / 1024  # kB -> GiB
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) / 1024 / 1024
                    used = total - free
                if total and used:
                    break
    except OSError:
        pass
    return used, total


def largest_process_to_reap(used_mb=0):
    """Return (pid, rss_mb, name) of the largest non-protected process.

    Prefers REAPABLE_HINTS (GUI apps) even when smaller; otherwise the
    largest process that is not protected.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,rss,comm", "--sort=-rss"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
    except subprocess.SubprocessError:
        return None

    best = None
    best_hint = None
    for line in out[1:]:
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, rss_mb, name = parts[0], int(parts[1]), parts[2]
        if rss_mb < 100:  # ignore small fry, incl. this script
            continue
        base = name.rsplit("/", 1)[-1].lower()
        if any(p in base for p in PROTECTED):
            continue
        if any(h in base for h in REAPABLE_HINTS):
            if best_hint is None or rss_mb > best_hint[1]:
                best_hint = (pid, rss_mb, base)
            continue
        if best is None or rss_mb > best[1]:
            best = (pid, rss_mb, base)
    return best_hint or best


def free_swap_quietly():
    """Best-effort swap reclaim without root."""
    try:
        subprocess.run(["sync"], check=False)
    except OSError:
        pass
    target = largest_process_to_reap()
    if target is None:
        return None
    pid, rss_mb, name = target
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        return None
    # Give it a moment to exit; escalate if it lingers.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(int(pid), 0)  # still alive?
            time.sleep(0.5)
        except OSError:
            return (pid, name, "reaped")
    try:
        os.kill(int(pid), signal.SIGKILL)
    except OSError:
        pass
    return (pid, name, "killed")


def monitor(interval, act_gb, warn_gb, logf):
    """Domain-prober style loop: sleep, measure, act on threshold."""
    while True:
        used, total = read_swap_usage_gb()
        stamp = time.strftime("%H:%M:%S")
        if used >= act_gb:
            result = free_swap_quietly()
            if result:
                pid, name, how = result
                logf(f"[{stamp}] swap {used:.1f}/{total:.1f}G >= {act_gb}G "
                     f"-> reaped {name} (pid {pid}) {how}")
            else:
                logf(f"[{stamp}] swap {used:.1f}/{total:.1f}G >= {act_gb}G "
                     f"but nothing safe to reap")
        elif used >= warn_gb:
            logf(f"[{stamp}] swap {used:.1f}/{total:.1f}G >= {warn_gb}G (warn)")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Swap-pressure watchdog (domain-prober style).")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--act-gb", type=float, default=4.0,
                    help="reap when swap used >= this (GiB)")
    ap.add_argument("--warn-gb", type=float, default=2.0)
    ap.add_argument("--log", default="/tmp/swap_watchdog.log")
    args = ap.parse_args()

    logf = lambda msg: print(msg, file=sys.stderr, flush=True) or \
        open(args.log, "a").write(msg + "\n")

    logf(f"[swap_watchdog] start interval={args.interval}s act={args.act_gb}G "
         f"warn={args.warn_gb}G")
    try:
        monitor(args.interval, args.act_gb, args.warn_gb, logf)
    except KeyboardInterrupt:
        logf("[swap_watchdog] stopped")


if __name__ == "__main__":
    main()
