$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw '请先运行 uv sync --locked --python 3.11' }
foreach ($port in @(3000,8000)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "端口 $port 已占用。请先确认现有服务，或运行本项目 scripts/stop.ps1。" }
}
$logPath = Join-Path $projectRoot 'workspace\logs'
New-Item -ItemType Directory -Force -Path $logPath | Out-Null
$backend = Start-Process -FilePath $pythonPath -ArgumentList '-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logPath 'backend.log') -RedirectStandardError (Join-Path $logPath 'backend-error.log')
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try { $response = Invoke-RestMethod 'http://127.0.0.1:8000/api/health'; if ($response.status -eq 'ok') { $ready = $true; break } } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw '后端未能启动，请查看 workspace/logs/backend-error.log' }
    # Vite keeps the same-origin HTTP/WebSocket proxy on port 3000.
    $nodePath = (Get-Command node.exe).Source
    $vitePath = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
    $viteArgs = @($vitePath,'--host','127.0.0.1','--port','3000','--strictPort')
    $frontend = Start-Process -FilePath $nodePath -ArgumentList $viteArgs -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logPath 'frontend.log') -RedirectStandardError (Join-Path $logPath 'frontend-error.log')
    @{ backend = $backend.Id; frontend = $frontend.Id; root = $projectRoot } | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $logPath 'processes.json')
    Write-Output '前端: http://localhost:3000'
    Write-Output '后端: http://127.0.0.1:8000/api/health'
} catch { Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue; throw }
