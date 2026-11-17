#!/bin/bash
cd "$(dirname "$0")"
echo "=== OLD EVAL START $(date)" >> logs/old_eval.log
./.venv/Scripts/python.exe src/eval_old_models.py > logs/eval_old_models.log 2>&1
echo "=== OLD EVAL DONE $(date)" >> logs/old_eval.log
