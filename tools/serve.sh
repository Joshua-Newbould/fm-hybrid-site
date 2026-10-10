#!/usr/bin/env bash
# Serve site/ on 8766 in tmux session fm-site.
cd "$(dirname "$0")/.." || exit 1
tmux has-session -t fm-site 2>/dev/null && { echo "fm-site already running"; exit 0; }
tmux new-session -d -s fm-site "python3 -m http.server 8766 --directory site"
echo "serving site/ at http://127.0.0.1:8766/"
