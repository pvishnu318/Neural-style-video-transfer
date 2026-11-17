#!/bin/bash
# Resumable experiment queue: skips any run whose checkpoint already exists. Logs in logs/.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
run() { name=$1; shift
  if [ -f checkpoints/$name.pt ]; then echo "=== SKIP $name (exists)"; return; fi
  echo "=== TRAIN $name $(date)"; $PY src/train.py --name $name --steps 2000 "$@" > logs/train_$name.log 2>&1; }
run control --mode control
for lt in 150 500 1500; do run b1_lt$lt --mode b1 --lt $lt; done
echo "=== EVAL $(date)"
$PY src/evaluate.py --split val --out results/eval_val_b1sweep.json --methods ckpt:control:checkpoints/control.pt ckpt:b1_lt150:checkpoints/b1_lt150.pt ckpt:b1_lt500:checkpoints/b1_lt500.pt ckpt:b1_lt1500:checkpoints/b1_lt1500.pt > logs/eval_val_b1sweep.log 2>&1
echo "=== QUEUE DONE $(date)"
