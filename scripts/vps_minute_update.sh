#!/usr/bin/env bash
# تحديث نتائج كل دقيقة على VPS (مو GitHub Actions).
# تحذير: لا تشغّله مع price_update.yml في نفس الوقت — يتعارضان على CSV/GitHub.
# إذا فعّلت VPS: عطّل schedule في price_update.yml أو أوقف الـ workflow.
# الاستخدام:
#   chmod +x scripts/vps_minute_update.sh
#   crontab: * * * * 1-5 /path/to/repo/scripts/vps_minute_update.sh >> /var/log/screener-minute.log 2>&1
#
# يتطلّب في بيئة السيرفر: FINNHUB_API_KEY (وباقي المفاتيح عند الحاجة)
# ووجود venv أو python3 مع requirements مثبتة.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCK="/tmp/screen-option-minute.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') skip — previous run still active"
  exit 0
fi

# ساعات السوق التقريبية (UTC): 14:00–20:05 أيام العمل — عدّل إن لزم
DOW="$(date -u +%u)"  # 1=Mon .. 7=Sun
HOUR="$(date -u +%H)"
MIN="$(date -u +%M)"
if [[ "$DOW" -gt 5 ]]; then
  exit 0
fi
# قبل 14:00 أو بعد 20:05 UTC — توقف
if [[ "$HOUR" -lt 14 ]] || [[ "$HOUR" -gt 20 ]] || { [[ "$HOUR" -eq 20 ]] && [[ "$MIN" -gt 5 ]]; }; then
  exit 0
fi

export SCREENER_SOURCE="${SCREENER_SOURCE:-both}"
PYTHON="${PYTHON:-python3}"

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') minute update start"
"$PYTHON" price_update.py

# الـ outcome ثقيل — كل 5 دقائق فقط
if (( 10#$(date -u +%M) % 5 == 0 )); then
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') outcome_tracker"
  "$PYTHON" outcome_tracker.py || echo "outcome_tracker soft-fail"
fi

# ادفع لـ GitHub حتى الداشبورد المشترك (github.io) يبقى متوافقاً
if [[ "${PUSH_TO_GITHUB:-1}" == "1" ]] && git remote get-url origin >/dev/null 2>&1; then
  bash .github/scripts/push_csv_changes.sh \
    "VPS minute update: $(date -u '+%Y-%m-%d %H:%M UTC')" \
    options_v3_results.csv outcomes.csv || echo "push soft-fail"
fi

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') minute update done"
