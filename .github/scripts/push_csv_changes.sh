#!/usr/bin/env bash
# Push CSV changes with retry — avoids rebase conflicts when multiple workflows overlap.
set -euo pipefail

MSG="${1:?commit message required}"
shift
FILES=("$@")

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files specified"
  exit 1
fi

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

# Retry used `reset --soft`, which re-staged engine files from an old checkout
# and reverted BUY-KPI / EOD-expiry fixes onto main. Only commit FILES.
stage_allowed_only() {
  git add "${FILES[@]}"
  local f allowed a
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    allowed=0
    for a in "${FILES[@]}"; do
      if [[ "$f" == "$a" ]]; then
        allowed=1
        break
      fi
    done
    if [[ $allowed -eq 0 ]]; then
      echo "Unstaging unexpected file: $f"
      git restore --staged -- "$f"
    fi
  done < <(git diff --cached --name-only)
}

stage_allowed_only
if git diff --cached --quiet; then
  echo "No CSV changes to commit"
  exit 0
fi

git commit -m "$MSG"

for attempt in 1 2 3 4 5; do
  if git pull --rebase origin main; then
    git push origin main
    echo "Pushed on attempt $attempt"
    exit 0
  fi

  echo "Rebase conflict on attempt $attempt — retry with latest main"
  git rebase --abort 2>/dev/null || true
  git fetch origin main
  # mixed: keep working-tree CSV edits, do not keep an old index (that reverts code)
  git reset --mixed origin/main
  stage_allowed_only
  if git diff --cached --quiet; then
    echo "No CSV changes left after reset to origin/main"
    exit 0
  fi
  git commit -m "$MSG (retry $attempt)"
  sleep $((attempt * 2))
done

echo "Failed to push after 5 attempts"
exit 1
