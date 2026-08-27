#!/usr/bin/env bash
# Push listed paths with retry — avoids rebase conflicts when workflows overlap.
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

# `sessions/` must allow `sessions/foo.md`. Exact equality used to unstage
# every note and skip the ledger. Prefix-match directories; never restage
# engine files after a failed rebase (use mixed reset, not soft).
path_is_allowed() {
  local f="$1"
  local a prefix
  for a in "${FILES[@]}"; do
    prefix="${a%/}"
    if [[ "$f" == "$prefix" || "$f" == "$prefix"/* ]]; then
      return 0
    fi
  done
  return 1
}

stage_allowed_only() {
  git add -- "${FILES[@]}"
  local f
  # core.quotepath=true يلفّ الأسماء العربية بعلامات اقتباس فيكسر المسار
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    f="${f#\"}"
    f="${f%\"}"
    if path_is_allowed "$f"; then
      continue
    fi
    echo "Unstaging unexpected file: $f"
    git restore --staged -- "$f"
  done < <(git -c core.quotepath=false diff --cached --name-only)
}

stage_allowed_only
if git diff --cached --quiet; then
  echo "No CSV changes to commit"
  exit 0
fi

git commit -m "$MSG"

for attempt in 1 2 3 4 5; do
  git fetch origin main
  if ! git pull --rebase origin main; then
    echo "Rebase conflict on attempt $attempt — retry with latest main"
    git rebase --abort 2>/dev/null || true
    git fetch origin main
    git reset --mixed origin/main
    stage_allowed_only
    if git diff --cached --quiet; then
      echo "No CSV changes left after reset to origin/main"
      exit 0
    fi
    git commit -m "$MSG (retry $attempt)"
  fi
  # Push is inside `if` so set -e does not skip retries on a rejected push.
  if git push origin main; then
    echo "Pushed on attempt $attempt"
    exit 0
  fi
  echo "Push rejected on attempt $attempt — retry"
  sleep $((attempt * 2))
done

echo "Failed to push after 5 attempts"
exit 1
