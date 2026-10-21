param([switch]$Cuda)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath ".venv")) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($Cuda) {
    # Official Windows CUDA wheels. CUDA 13.0 requires a compatible NVIDIA driver.
    & .\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
    # Restore application pins that the torch resolver may upgrade.
    & .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps numpy==1.26.4 Pillow==11.0.0
}
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe scripts\check_environment.py
Write-Host "Cài đặt hoàn tất. Chạy .\scripts\run_demo.ps1 để tạo chuyến mẫu."
