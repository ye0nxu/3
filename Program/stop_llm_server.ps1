$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $projectRoot ".llm_server.pid"

if (-not (Test-Path $pidPath)) {
    Write-Output "No PID file found. Server may already be stopped."
    exit 0
}

try {
    $pid = [int](Get-Content -Path $pidPath -Raw).Trim()
} catch {
    Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    Write-Output "Invalid PID file removed."
    exit 0
}

$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    Write-Output "Process not found. PID file removed."
    exit 0
}

Stop-Process -Id $pid -Force
Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
Write-Output "Stopped LLM server (PID=$pid)."
