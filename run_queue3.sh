#!/bin/bash
# Seed-variance check (no tuning): B1 and B2 at lt=150 with seed 1, then val eval.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
echo "=== TRAIN b1_lt150_s1 $(date)" >> logs/queue3.log
$PY src/train.py --mode b1 --name b1_lt150_s1 --lt 150 --steps 2000 --seed 1 > logs/train_b1_lt150_s1.log 2>&1
echo "=== TRAIN b2_lt150_s1 $(date)" >> logs/queue3.log
$PY src/train.py --mode b2 --name b2_lt150_s1 --lt 150 --steps 2000 --seed 1 > logs/train_b2_lt150_s1.log 2>&1
echo "=== EVAL $(date)" >> logs/queue3.log
$PY src/evaluate.py --split val --methods ckpt:b1_lt150_s1:checkpoints/b1_lt150_s1.pt attn:b2_lt150_s1:checkpoints/b2_lt150_s1.pt --out results/eval_val_seed1.json > logs/eval_val_seed1.log 2>&1
echo "=== QUEUE3 DONE $(date)" >> logs/queue3.log
