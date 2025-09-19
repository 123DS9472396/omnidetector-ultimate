# PowerShell script to download COCO128 dataset
# Usage: .\scripts\download_coco128.ps1

Write-Host "Creating data directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "Downloading COCO128 dataset (small test dataset)..." -ForegroundColor Green

# Download coco128.zip
Write-Host "Downloading coco128.zip..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip" -OutFile "data\coco128.zip"

# Extract the zip file
Write-Host "Extracting coco128.zip..." -ForegroundColor Yellow
Expand-Archive -Path "data\coco128.zip" -DestinationPath "data" -Force

# Clean up zip file
Remove-Item "data\coco128.zip"

Write-Host "COCO128 dataset downloaded successfully to data/coco128/" -ForegroundColor Green
Write-Host "Dataset structure:" -ForegroundColor Cyan
Get-ChildItem -Path "data\coco128" -Recurse -Directory | Select-Object FullName
