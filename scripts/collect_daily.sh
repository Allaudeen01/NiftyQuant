#!/usr/bin/env bash
# Daily option-chain + India VIX collection wrapper for Linux / EC2 (cron-driven).
#
# Starts ONE read-only session of collect_market_data.py. The collector waits
# for the open, collects through the session, self-exits at 15:30 IST, and skips
# weekends/NSE holidays on its own. NO ORDERS are ever placed.
#
# Cron (server timezone = Asia/Kolkata), weekdays 09:05 IST:
#   5 9 * * 1-5  /home/ec2-user/trading_assistant/scripts/collect_daily.sh
#
# Override the project path with PROJ=/path if not the default.

set -euo pipefail

PROJ="${PROJ:-$HOME/trading_assistant}"
cd "$PROJ"
mkdir -p logs

STAMP="$(date +%Y%m%d)"
LOG="logs/collect_${STAMP}.log"

# Activate a virtualenv if present (.venv), else use system python3.
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY=python
else
    PY=python3
fi

echo "[collect_daily] $(date -Is) starting session" >> "$LOG"
"$PY" scripts/collect_market_data.py \
    --num-expiries 2 --strike-band-pct 6 --poll 120 --request-pause 1.2 \
    >> "$LOG" 2>&1
echo "[collect_daily] $(date -Is) session ended" >> "$LOG"
