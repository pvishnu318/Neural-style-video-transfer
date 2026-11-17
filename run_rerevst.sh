#!/bin/bash
cd "$(dirname "$0")"
echo "=== RERE START $(date)" >> logs/rerevst.log
./.venv/Scripts/python.exe src/eval_external.py --split test --methods rerevst --out results/eval_test_external.json > logs/eval_rerevst.log 2>&1
echo "=== RERE DONE $(date)" >> logs/rerevst.log
