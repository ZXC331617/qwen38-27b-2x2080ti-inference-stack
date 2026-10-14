param(
  [string]$Tag = "run",
  [string]$TiersCsv = ""
)

$Tiers = @()
if ($TiersCsv -ne "") {
  $Tiers = $TiersCsv -split '[,;\s]+' | Where-Object { $_ -ne '' }
}

$ErrorActionPreference = "Stop"
$dir = "D:\1\靶场\全层消融\logs"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$code = "ERR"
try {
  $code = (Invoke-WebRequest -Uri "http://127.0.0.1:8788/v1/models" -TimeoutSec 8 -UseBasicParsing).StatusCode
} catch {
  $code = "ERR:" + $_.Exception.Message
}
Write-Output "router_status=$code"
if ($code -ne 200) {
  Write-Output "router not ready, abort"
  exit 1
}

$log = Join-Path $dir ("bench-" + $Tag + ".log")
Remove-Item $log -ErrorAction SilentlyContinue
Remove-Item ($log + ".err") -ErrorAction SilentlyContinue

$py = "C:\Users\jjj\AppData\Local\Programs\Python\Python314\python.exe"
$script = "D:\1\靶场\全层消融\bench-harness-real.py"
$argList = @("-u", $script) + $Tiers

$p = Start-Process -FilePath $py -ArgumentList $argList -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
Write-Output ("launched pid=" + $p.Id + " log=" + $log)
