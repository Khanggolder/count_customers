param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install Cython==3.1.4 gdown==5.2.0 yacs==0.1.8 tensorboard==2.20.0 chardet==5.2.0
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements-reid.txt
if ($Cuda) {
  & .\.venv\Scripts\python.exe -m pip install --upgrade torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
}
Write-Host "Cài xong. Chạy kiểm tra: .\.venv\Scripts\python.exe scripts\check_environment.py"

