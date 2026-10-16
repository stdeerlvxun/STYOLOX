param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$serverUrlFile = Join-Path $projectPath 'server-url.txt'
if (Test-Path -LiteralPath $serverUrlFile) {
    $serverUrl = (Get-Content -LiteralPath $serverUrlFile -Raw -Encoding UTF8).Trim()
    if (-not $NoBrowser) { Start-Process $serverUrl }
    return
}
$pythonPath = Join-Path (Split-Path $projectPath -Parent) 'envs\yolox\Scripts\python.exe'
$appUrl = 'http://127.0.0.1:8878'
$ready = $false
try {
    $state = Invoke-RestMethod -Uri ($appUrl + '/api/state') -TimeoutSec 2
    $ready = $null -ne $state.token
} catch {}
if (-not $ready) {
    $outPath = Join-Path $projectPath 'data'
    New-Item -ItemType Directory -Path $outPath -Force | Out-Null
    $processArgs = @('"' + (Join-Path $projectPath 'app.py') + '"')
    Start-Process -FilePath $pythonPath -ArgumentList $processArgs -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outPath 'server.log') -RedirectStandardError (Join-Path $outPath 'server-error.log')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 300
        try { $state = Invoke-RestMethod -Uri ($appUrl + '/api/state') -TimeoutSec 2; $ready = $null -ne $state.token; if ($ready) { break } } catch {}
    }
}
if (-not $ready) { throw 'Local image labeler did not start. See data/server-error.log.' }
if (-not $NoBrowser) { Start-Process $appUrl }
