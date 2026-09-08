$projectRoot = Split-Path $PSScriptRoot -Parent
$record = Join-Path $projectRoot 'workspace\logs\processes.json'
if (Test-Path -LiteralPath $record) {
    $saved = Get-Content -Raw -LiteralPath $record | ConvertFrom-Json
    foreach ($processId in @($saved.backend,$saved.frontend)) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
        if ($processInfo -and (($processInfo.CommandLine -like '*backend.app.main:app*') -or ($processInfo.CommandLine -like '*miao5_rci_lab*frontend*vite*'))) { Stop-Process -Id $processId }
    }
    Remove-Item -LiteralPath $record
}
