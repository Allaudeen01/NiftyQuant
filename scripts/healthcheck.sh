#!/usr/bin/env bash
# Post-close heartbeat: alert (via SNS) if today's snapshot count is too low,
# so a failed/missed collection day never goes unnoticed on an unattended box.
#
# Requires an instance IAM role with sns:Publish (if SNS_TOPIC_ARN is set).
# Env:
#   MIN_SNAPSHOTS   minimum expected files (default 120; full day ~185 at 2-min)
#   SNS_TOPIC_ARN   optional; if set and count is low, publishes an alert
#
# Cron (Asia/Kolkata), weekdays 15:45 IST (after self-exit):
#   45 15 * * 1-5  SNS_TOPIC_ARN=arn:aws:sns:ap-south-1:...:niftyquant /home/ec2-user/trading_assistant/scripts/healthcheck.sh

set -euo pipefail

PROJ="${PROJ:-$HOME/trading_assistant}"
cd "$PROJ"

Y="$(date +%Y)"; M="$(date +%m)"; D="$(date +%d)"
DIR="data/option_chain/${Y}/${M}/${D}"
COUNT="$(find "$DIR" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
MIN="${MIN_SNAPSHOTS:-120}"

MSG="NiftyQuant ${Y}-${M}-${D}: ${COUNT} option-chain snapshots (min ${MIN})"
echo "[healthcheck] $(date -Is) ${MSG}"

if [ "${COUNT:-0}" -lt "$MIN" ]; then
    echo "[healthcheck] LOW COLLECTION WARNING"
    if [ -n "${SNS_TOPIC_ARN:-}" ]; then
        aws sns publish --topic-arn "$SNS_TOPIC_ARN" \
            --subject "NiftyQuant LOW COLLECTION (${COUNT})" --message "$MSG"
    fi
    exit 1
fi
