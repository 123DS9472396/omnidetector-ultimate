# PowerShell script to download YOLO model weights
# Usage: .\scripts\download_models.ps1

Write-Host "Creating models directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host "Downloading YOLOv8 model weights..." -ForegroundColor Green

# Download yolov8n.pt (nano - fastest on CPU)
Write-Host "Downloading yolov8n.pt (nano)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "models\yolov8n.pt"

# Download yolov8s.pt (small - better accuracy)
Write-Host "Downloading yolov8s.pt (small)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt" -OutFile "models\yolov8s.pt"

# Download yolov8m.pt (medium - even better accuracy)
Write-Host "Downloading yolov8m.pt (medium)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt" -OutFile "models\yolov8m.pt"

Write-Host "Models downloaded successfully to models/ directory!" -ForegroundColor Green
Write-Host "Available models:" -ForegroundColor Cyan
Get-ChildItem -Path "models" -Name "*.pt"
