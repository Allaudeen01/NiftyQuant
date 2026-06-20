# Daily option-chain collection wrapper (for Windows Task Scheduler).
#
# Runs ~one full session of 5-minute snapshots of the nearest NIFTY expiry,
# with synchronized India VIX. Logs to logs/collect_<date>.log.
#
# Schedule it to start at 09:15 IST on weekdays (see README note below).
#
# To register the scheduled task (run once, in an Administrator PowerShell):
#   schtasks /Create /TN "NiftyQuant-Collect" /TR `
#     "powershell -ExecutionPolicy Bypass -File D:\trading_assistant\scripts\collect_daily.ps1" `
#     /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:15 /F

$ErrorActionPreference = "Continue"
$proj = "D:\trading_assistant"
Set-Location $proj

New-Item -ItemType Directory -Force -Path "$proj\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd"
$log = "$proj\logs\collect_$stamp.log"

# ~75 snapshots * 5 min ~= a full 09:15-15:30 session.
python "$proj\scripts\collect_option_chain.py" `
    --underlying NIFTY --expiry nearest --count 75 --interval 300 `
    *>> $log
