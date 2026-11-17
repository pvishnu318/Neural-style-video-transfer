#!/bin/bash
cd "$(dirname "$0")"
echo "=== BLEND EXTRA START $(date)" >> logs/blend_extra.log
./.venv/Scripts/python.exe src/evaluate.py --split test --methods B0+blend0.1 B0+blend0.2 B0+blend0.45 --out results/eval_test_blend_extra.json > logs/eval_test_blend_extra.log 2>&1
echo "=== BLEND EXTRA DONE $(date)" >> logs/blend_extra.log
