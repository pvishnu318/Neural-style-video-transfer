#!/bin/bash
cd "$(dirname "$0")"
echo "=== MCC START $(date)" >> logs/mccnet.log
./.venv/Scripts/python.exe src/eval_external.py --split test --methods mccnet --out results/eval_test_external_mccnet.json > logs/eval_mccnet.log 2>&1
echo "=== MCC DONE $(date)" >> logs/mccnet.log
