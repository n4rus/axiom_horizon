#!/usr/bin/env bash
# Watchdog for the full wiki absorb: alerts if process dies or GPU stalls.
# Usage: bash tools/wiki_watch.sh  (run periodically, e.g. cron or nohup loop)
LOG=/tmp/wiki_watch.log
ABS_LOG=/tmp/wiki_absorb_b256.log

stamp() { date '+%F %T'; }

if ! pgrep -f wiki_absorb_chunks.py > /dev/null; then
  echo "$(stamp) ALERT: wiki_absorb_chunks.py NOT running" >> "$LOG"
  exit 1
fi

# Detect GPU stall: sample twice, 30s apart
G1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -dc '0-9')
sleep 30
G2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -dc '0-9')
echo "$(stamp) gpu_sample g1=$G1 g2=$G2" >> "$LOG"

if [ "$G1" -eq 0 ] && [ "$G2" -eq 0 ]; then
  echo "$(stamp) ALERT: GPU idle 60s+ (possible embed stall)" >> "$LOG"
  tail -n 3 "$ABS_LOG" >> "$LOG"
fi