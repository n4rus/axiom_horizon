#!/usr/bin/env bash
# Watchdog supervisor loop (single instance). Start via:
#   setsid -f bash /home/l/Desktop/AxiomTree/axiom_horizon/tools/wiki_watch_loop.sh
while true; do
    bash /home/l/Desktop/AxiomTree/axiom_horizon/tools/wiki_watch.sh
    sleep 300
done