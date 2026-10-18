$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Chưa có .venv. Chạy: py -3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
Set-Location $ProjectRoot
& $PythonExe scripts\run_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Demo đã nạp vào data\busstop_flow.db. Khởi động API và Streamlit theo README.md."
