#!/bin/bash
# Round-2 step 2: B2 with the lt selected by the Amendment-1 rule on val (lt=150), then val eval of B2.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
echo "=== TRAIN b2_lt150 $(date)" >> logs/queue2.log
$PY src/train.py --mode b2 --name b2_lt150 --lt 150 --steps 2000 > logs/train_b2_lt150.log 2>&1
echo "=== EVAL $(date)" >> logs/queue2.log
$PY src/evaluate.py --split val --methods attn:b2_lt150:checkpoints/b2_lt150.pt --out results/eval_val_b2.json > logs/eval_val_b2.log 2>&1
echo "=== QUEUE2 DONE $(date)" >> logs/queue2.log
