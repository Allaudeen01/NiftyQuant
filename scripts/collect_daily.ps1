# Daily option-chain + India VIX collection wrapper (for Windows Task Scheduler).
#
# Runs ONE full 09:15-15:30 session of the hardened, read-only collector
# (collect_market_data.py). The collector self-exits after market close, skips
# weekends/NSE holidays on its own, and writes its own structured log to
# logs/collector_<date>.log. This wrapper additionally captures all console +
# SDK stderr output to logs/collect_<date>.log.
#
# NO ORDERS are ever placed. Read-only.
#
# Register the scheduled task ONCE, in an Administrator PowerShell:
#   schtasks /Create /TN "NiftyQuant-Collect" /SC WEEKLY /D MON,TUE,WED,THU,FRI `
#     /ST 09:05 /F /TR `
#     "powershell -ExecutionPolicy Bypass -NoProfile -File D:\trading_assistant\scripts\collect_daily.ps1"
#
# Recommended Task Scheduler settings (set in the GUI or via the XML):
#   - "Run whether user is logged on or not"
#   - "Wake the computer to run this task"
#   - "Start the task only if the computer is on AC power" -> UNCHECK if on a laptop
#   - Stop the task if it runs longer than: 8 hours (safety)

$ErrorActionPreference = "Continue"
$proj = "D:\trading_assistant"
Set-Location $proj

New-Item -ItemType Directory -Force -Path "$proj\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd"
$log = "$proj\logs\collect_$stamp.log"

# Steady-state collection parameters (see scripts/collect_market_data.py).
# Tuned to stay within Angel rate limits over a full session.
python "$proj\scripts\collect_market_data.py" `
    --num-expiries 2 --strike-band-pct 6 --poll 120 --request-pause 1.2 `
    *>> $log
