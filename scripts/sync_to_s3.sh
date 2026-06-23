#!/usr/bin/env bash
# Off-instance backup of collected data to S3 (run nightly after close).
#
# Intraday option snapshots CANNOT be backfilled, so a durable off-instance
# copy protects months of irreplaceable data against instance/EBS loss.
#
# Requires an instance IAM role (or aws creds) with s3:PutObject on the bucket.
# Set the bucket via env:  export S3_BUCKET=my-niftyquant-data
#
# Cron (Asia/Kolkata), weekdays 16:00 IST (after close):
#   0 16 * * 1-5  S3_BUCKET=my-niftyquant-data /home/ec2-user/trading_assistant/scripts/sync_to_s3.sh

set -euo pipefail

PROJ="${PROJ:-$HOME/trading_assistant}"
: "${S3_BUCKET:?set S3_BUCKET, e.g. export S3_BUCKET=my-niftyquant-data}"
cd "$PROJ"

aws s3 sync data/option_chain "s3://${S3_BUCKET}/option_chain" --no-progress
aws s3 sync data/vix          "s3://${S3_BUCKET}/vix"          --no-progress
aws s3 sync logs              "s3://${S3_BUCKET}/logs"         --no-progress

echo "[sync_to_s3] $(date -Is) synced data/ + logs/ to s3://${S3_BUCKET}"
