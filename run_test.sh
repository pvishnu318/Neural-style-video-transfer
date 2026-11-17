#!/bin/bash
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
echo "=== TEST START $(date)" >> logs/test.log
$PY src/evaluate.py --split test --timing --out results/eval_test.json --methods B0 B0+blend0.3 B0+blend0.6 \
  ckpt:control:checkpoints/control.pt ckpt:b1_lt150:checkpoints/b1_lt150.pt ckpt:b1_lt150_s1:checkpoints/b1_lt150_s1.pt \
  ckpt:b1_lt500:checkpoints/b1_lt500.pt ckpt:b1_lt1500:checkpoints/b1_lt1500.pt \
  attn:b2_lt150:checkpoints/b2_lt150.pt attn:b2_lt150_s1:checkpoints/b2_lt150_s1.pt > logs/eval_test.log 2>&1
echo "=== TEST DONE $(date)" >> logs/test.log
