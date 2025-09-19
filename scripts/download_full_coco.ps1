# PowerShell script to download full COCO 2017 dataset (large download)
# Usage: .\scripts\download_full_coco.ps1
# Warning: This downloads ~118K images (~20GB+), use only if needed

Write-Host "WARNING: This will download the full COCO 2017 dataset (~20GB+)" -ForegroundColor Red
$confirm = Read-Host "Do you want to continue? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Download cancelled." -ForegroundColor Yellow
    exit
}

Write-Host "Creating COCO directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "data\coco\images" | Out-Null
New-Item -ItemType Directory -Force -Path "data\coco\annotations" | Out-Null

Write-Host "Downloading COCO 2017 dataset..." -ForegroundColor Green

# Download training images
Write-Host "Downloading train2017.zip (~18GB)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile "data\coco\train2017.zip"

# Download validation images
Write-Host "Downloading val2017.zip (~1GB)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "data\coco\val2017.zip"

# Download annotations
Write-Host "Downloading annotations_trainval2017.zip..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "data\coco\annotations_trainval2017.zip"

# Extract files
Write-Host "Extracting train2017.zip..." -ForegroundColor Yellow
Expand-Archive -Path "data\coco\train2017.zip" -DestinationPath "data\coco\images" -Force

Write-Host "Extracting val2017.zip..." -ForegroundColor Yellow
Expand-Archive -Path "data\coco\val2017.zip" -DestinationPath "data\coco\images" -Force

Write-Host "Extracting annotations..." -ForegroundColor Yellow
Expand-Archive -Path "data\coco\annotations_trainval2017.zip" -DestinationPath "data\coco" -Force

# Clean up zip files
Write-Host "Cleaning up zip files..." -ForegroundColor Yellow
Remove-Item "data\coco\train2017.zip"
Remove-Item "data\coco\val2017.zip"
Remove-Item "data\coco\annotations_trainval2017.zip"

Write-Host "Full COCO 2017 dataset downloaded successfully!" -ForegroundColor Green
Write-Host "Dataset structure:" -ForegroundColor Cyan
Get-ChildItem -Path "data\coco" -Recurse -Directory | Select-Object FullName
