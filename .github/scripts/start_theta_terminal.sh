#!/usr/bin/env bash
# يبدأ Theta Terminal إن وُجدت بيانات الحساب — للفشل بصمت حتى يبقى EOD على Yahoo.
set -u

if [ -z "${THETA_EMAIL:-}" ] || [ -z "${THETA_PASSWORD:-}" ]; then
  echo "Theta Terminal skipped — add GitHub secrets THETA_EMAIL and THETA_PASSWORD (free EOD account)."
  exit 0
fi

if ! command -v java >/dev/null 2>&1; then
  echo "Theta Terminal skipped — java not installed."
  exit 0
fi
java -version 2>&1 | head -3 || true

JAR_URL="${THETA_JAR_URL:-https://download-unstable.thetadata.us/ThetaTerminalv3.jar}"
WORKDIR="${RUNNER_TEMP:-/tmp}/theta-terminal"
mkdir -p "$WORKDIR"
JAR="$WORKDIR/ThetaTerminalv3.jar"
CREDS="$WORKDIR/creds.txt"

echo "Downloading Theta Terminal..."
if ! curl -fsSL "$JAR_URL" -o "$JAR"; then
  echo "Theta Terminal skipped — jar download failed."
  exit 0
fi
if ! unzip -t "$JAR" >/dev/null 2>&1; then
  echo "Theta Terminal skipped — download was not a jar."
  exit 0
fi

# v3 يقرأ creds.txt بجانب الـ jar (سطر1 إيميل، سطر2 كلمة المرور)
umask 077
printf '%s\n%s\n' "$THETA_EMAIL" "$THETA_PASSWORD" > "$CREDS"

echo "Starting Theta Terminal..."
(
  cd "$WORKDIR"
  java -jar "$JAR"
) >/tmp/theta-terminal.log 2>&1 &
echo $! >/tmp/theta-terminal.pid

theta_http_code() {
  curl -sS -o /dev/null -w "%{http_code}" --max-time 2 \
    "http://127.0.0.1:25503/v3/option/history/eod?format=json" 2>/dev/null || true
}

for i in $(seq 1 45); do
  if ! kill -0 "$(cat /tmp/theta-terminal.pid 2>/dev/null)" 2>/dev/null; then
    echo "Theta Terminal process exited."
    tail -40 /tmp/theta-terminal.log 2>/dev/null || true
    echo "EOD will use Yahoo fallback."
    exit 0
  fi
  code=$(theta_http_code)
  # اتصال مرفوض = 000 أو فارغ. 2xx/4xx يعني السيرفر ردّ (حتى لو نقصت باراميترات).
  if [[ "$code" =~ ^[12345][0-9][0-9]$ ]]; then
    echo "Theta Terminal ready (${i}s, http $code)."
    exit 0
  fi
  sleep 2
done

echo "Theta Terminal did not open port 25503 — EOD will use Yahoo fallback."
tail -40 /tmp/theta-terminal.log 2>/dev/null || true
exit 0
