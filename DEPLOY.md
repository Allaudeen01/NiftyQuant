# Deploying the NiftyQuant Collector on EC2 (unattended, 6-month run)

Goal: run the **read-only** option-chain + India VIX collector on an always-on
Linux instance so collection never depends on a laptop being awake. The
collector places **no orders** — it only reads market data and writes parquet.

Defaults below: **Amazon Linux 2023**, **ap-south-1 (Mumbai)**, **t4g.small**,
credentials in a `.env` file. Adjust to taste.

---

## 1. Launch the instance
- **Region: ap-south-1 (Mumbai)** — closest to NSE/Angel; lowest latency, fewest
  connection resets. (AWS credits are account-wide, so they apply here.)
- **Type: t4g.small** (ARM, ~$12/mo) or **t3.micro** (free-tier). Workload is tiny.
- **AMI: Amazon Linux 2023** (or Ubuntu 24.04 — adjust package commands).
- **Storage: 20 GB gp3** (far more than needed; 6 months of snapshots is < 1 GB).
- **Security group:** inbound **SSH (22) from YOUR IP only**. No other inbound.
  The collector only makes **outbound** HTTPS calls.
- **IAM instance role** (recommended, avoids putting AWS keys on the box):
  attach a role with least-privilege `s3:PutObject`/`s3:ListBucket` on your
  backup bucket and `sns:Publish` on your alert topic.

## 2. Set the timezone to IST
```bash
sudo timedatectl set-timezone Asia/Kolkata
date   # confirm IST
```
(The collector's internal clock already uses IST, but matching the server TZ
keeps cron times simple.)

## 3. Install dependencies
```bash
sudo dnf install -y python3.11 git        # AL2023  (Ubuntu: apt-get install python3 python3-venv git)
git clone https://github.com/Allaudeen01/NiftyQuant.git ~/trading_assistant
cd ~/trading_assistant
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install smartapi-python pyotp
# awscli is preinstalled on AL2023; else: pip install awscli
```

## 4. Credentials (keep them off the repo)
Create `.env` (it is gitignored) with `chmod 600`:
```bash
umask 077
cat > .env <<'EOF'
ANGEL_API_KEY=...
ANGEL_CLIENT_CODE=...
ANGEL_MPIN=...
ANGEL_TOTP_SECRET=...
EOF
chmod 600 .env
```
(Better: store these in AWS SSM Parameter Store / Secrets Manager and export
them at startup. The `.env` file is the simple option.)

## 5. Smoke test BEFORE trusting cron
```bash
source .venv/bin/activate
python scripts/collect_market_data.py --once --ignore-market-hours --test
```
Confirm: it authenticates from the instance IP, prints a snapshot line, and
writes files under `data_test/`. **If Angel rejects the cloud IP, resolve that
now** (it usually works). Delete `data_test/` after.

## 6. Schedule with cron (times are IST after step 2)
```bash
chmod +x scripts/collect_daily.sh scripts/sync_to_s3.sh scripts/healthcheck.sh
crontab -e
```
Add:
```cron
# Collect one session, weekdays 09:05 IST (collector self-exits at 15:30)
5 9 * * 1-5  /home/ec2-user/trading_assistant/scripts/collect_daily.sh

# Heartbeat: alert if today's collection is low, 15:45 IST
45 15 * * 1-5  SNS_TOPIC_ARN=arn:aws:sns:ap-south-1:ACCT:niftyquant /home/ec2-user/trading_assistant/scripts/healthcheck.sh

# Off-instance backup to S3, 16:00 IST
0 16 * * 1-5  S3_BUCKET=my-niftyquant-data /home/ec2-user/trading_assistant/scripts/sync_to_s3.sh
```

## 7. Backup bucket + alert topic (one-time)
```bash
aws s3 mb s3://my-niftyquant-data --region ap-south-1
aws sns create-topic --name niftyquant --region ap-south-1
aws sns subscribe --topic-arn arn:aws:sns:ap-south-1:ACCT:niftyquant \
    --protocol email --notification-endpoint you@example.com   # confirm via email
```

---

## Operational notes
- **Run it in exactly ONE place.** Once EC2 is collecting, stop the laptop
  collector to avoid duplicate/competing runs.
- **No sleep concerns** on a server — the Windows keep-awake code is a harmless
  no-op on Linux.
- **Logs:** structured JSON at `logs/collector_<date>.log`; console+SDK output
  at `logs/collect_<date>.log`. Both sync to S3.
- **Quality check anytime:** `python scripts/option_quality_report.py --date YYYY-MM-DD`.
- **Cost:** ~$12/mo (t4g.small) + a few dollars EBS/S3/SNS — trivially covered
  by the AWS credits.
- **Security:** this box holds broker credentials. Lock the security group to
  your IP, keep `.env` at `chmod 600` (or use SSM), and patch the OS periodically.
  It is read-only (cannot place orders), but the TOTP secret is still sensitive.
