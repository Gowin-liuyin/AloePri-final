param(
    [ValidateSet('Prepare', 'Export', 'Check', 'Chat', 'All')]
    [string]$Action = 'Chat',
    [string]$Prompt = 'Explain machine learning in one sentence.',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [ValidateSet('cpu', 'cuda')][string]$CheckDevice = 'cuda',
    [double]$CorrectnessAtol = 0.0002,
    [string]$Archive = 'D:\Models.zip',
    [int]$MaxNewTokens = 32
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Create .venv and install requirements-tested.txt first; see docs/LLAMA_1B.md.'
}
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:HF_DEACTIVATE_ASYNC_LOAD = '1'
$env:MKL_DISABLE_FAST_MM = '1'
$env:PYTHONIOENCODING = 'utf-8'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
function Invoke-ProjectPython {
    & $pythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
Push-Location $projectRoot
try {
    if ($Action -in @('Prepare', 'All')) {
        Invoke-ProjectPython scripts/setup/prepare_llama_1b.py --archive $Archive
    }
    if ($Action -in @('Export', 'All')) {
        # Zero noise and signed permutations preserve the source BF16 weight values.
        Invoke-ProjectPython scripts/offline_obfuscation/export_llama_checkpoint.py --dtype bfloat16 --device cpu
        Invoke-ProjectPython scripts/release/export_llama_bundle.py --materialize
    }
    if ($Action -in @('Check', 'All')) {
        # Compare FP32 models in separate processes; BF16 is used for daily inference.
        Invoke-ProjectPython scripts/evaluation/run_llama_correctness.py --dtype float32 --device $CheckDevice --correctness-atol $CorrectnessAtol --max-new-tokens 8
    }
    if ($Action -in @('Chat', 'All')) {
        Invoke-ProjectPython scripts/release/infer_bundle.py --device $Device --dtype bfloat16 --prompt $Prompt --max-new-tokens $MaxNewTokens
    }
} finally {
    Pop-Location
}
