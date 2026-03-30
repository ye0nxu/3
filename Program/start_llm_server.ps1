param(
    [string]$CondaEnv = "PJ_310_LLM_SAM3",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8008,
    [string]$ModelId = ""
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $projectRoot ".llm_server.pid"
$defaultRemoteModel = "G:\models\Qwen2.5-7B-Instruct"
$defaultLocalModel = Join-Path $projectRoot "assets\\models\\LLM_models\\Qwen2.5-7B-Instruct"

if (Test-Path $pidPath) {
    try {
        $oldPid = [int](Get-Content -Path $pidPath -Raw).Trim()
        if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
            Write-Output "LLM server already running (PID=$oldPid)"
            exit 0
        }
    } catch {
        # Ignore malformed pid file.
    }
}

$condaExe = Join-Path $env:USERPROFILE "anaconda3\\Scripts\\conda.exe"
if (-not (Test-Path $condaExe)) {
    $condaExe = "conda"
}

$args = @(
    "run",
    "-n",
    $CondaEnv,
    "python",
    "run_llm_server.py",
    "--host",
    $BindHost,
    "--port",
    "$Port"
)

if ([string]::IsNullOrWhiteSpace($ModelId)) {
    if (Test-Path $defaultRemoteModel) {
        $ModelId = $defaultRemoteModel
    } elseif (Test-Path $defaultLocalModel) {
        $ModelId = $defaultLocalModel
    } else {
        $ModelId = "Qwen/Qwen2.5-7B-Instruct"
    }
}
$args += @("--model-id", $ModelId)

$proc = Start-Process -FilePath $condaExe -ArgumentList $args -WorkingDirectory $projectRoot -PassThru
Set-Content -Path $pidPath -Value "$($proc.Id)" -Encoding ascii
Write-Output "Started LLM server on http://${BindHost}:$Port (PID=$($proc.Id), model=$ModelId)"
