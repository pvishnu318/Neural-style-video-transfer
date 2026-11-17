#!/bin/bash
# Periodic commit + push of the project repo (what .gitignore allows only). Bounded: stops after MAX_HOURS (default 8)
# or as soon as a file named STOP_AUTOPUSH exists in this folder. Output goes to logs/autopush.log (git-ignored).
cd "$(dirname "$0")"
END=$(( $(date +%s) + ${MAX_HOURS:-8} * 3600 ))
echo "=== autopush start $(date)" >> logs/autopush.log
while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP_AUTOPUSH ]; do
  git add -A
  if ! git diff --cached --quiet; then
    git commit -q -m "Auto-save $(date '+%Y-%m-%d %H:%M')" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" >> logs/autopush.log 2>&1
  fi
  GIT_TERMINAL_PROMPT=0 timeout 900 git push -q origin main >> logs/autopush.log 2>&1 && echo "$(date '+%H:%M') pushed $(git rev-parse --short HEAD)" >> logs/autopush.log || echo "$(date '+%H:%M') push FAILED" >> logs/autopush.log
  sleep 600
done
echo "=== autopush stop $(date)" >> logs/autopush.log
