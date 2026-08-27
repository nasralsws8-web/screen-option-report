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

JAR_URL="${THETA_JAR_URL:-https://download-unstable.thetadata.us/ThetaTerminalv3.jar}"
JAR="${RUNNER_TEMP:-/tmp}/ThetaTerminalv3.jar"

echo "Downloading Theta Terminal..."
if ! curl -fsSL "$JAR_URL" -o "$JAR"; then
  echo "Theta Terminal skipped — jar download failed."
  exit 0
fi

echo "Starting Theta Terminal..."
java -jar "$JAR" "$THETA_EMAIL" "$THETA_PASSWORD" >/tmp/theta-terminal.log 2>&1 &
echo $! >/tmp/theta-terminal.pid

for i in $(seq 1 45); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 \
    "http://127.0.0.1:25503/v3/option/history/eod?format=json" || echo "000")
  if [ "$code" != "000" ]; then
    echo "Theta Terminal ready (${i}s, http $code)."
    exit 0
  fi
  sleep 2
done

echo "Theta Terminal did not open port 25503 — EOD will use Yahoo fallback."
exit 0
