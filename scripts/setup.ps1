# OmniDetector Setup Script
# This script downloads models and datasets to get you started quickly

param(
    [switch]$SkipModels,
    [switch]$SkipDataset,
    [switch]$FullCoco,
    [switch]$Help
)

if ($Help) {
    Write-Host "OmniDetector Setup Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\scripts\setup.ps1 [options]" -ForegroundColor Green
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -SkipModels    Skip downloading model weights"
    Write-Host "  -SkipDataset   Skip downloading datasets"
    Write-Host "  -FullCoco      Download full COCO dataset (large!)"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\scripts\setup.ps1                    # Download models + COCO128"
    Write-Host "  .\scripts\setup.ps1 -FullCoco          # Download models + full COCO"
    Write-Host "  .\scripts\setup.ps1 -SkipModels        # Download only datasets"
    exit 0
}

Write-Host "🚀 OmniDetector Setup" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ Please run this script from the OmniDetector root directory" -ForegroundColor Red
    exit 1
}

# Download models
if (-not $SkipModels) {
    Write-Host "📦 Downloading YOLO model weights..." -ForegroundColor Green
    try {
        & ".\scripts\download_models.ps1"
        Write-Host "✅ Models downloaded successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to download models: $_" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# Download datasets
if (-not $SkipDataset) {
    if ($FullCoco) {
        Write-Host "📊 Downloading full COCO 2017 dataset..." -ForegroundColor Green
        Write-Host "⚠️  Warning: This is a large download (~20GB+)" -ForegroundColor Yellow
        try {
            & ".\scripts\download_full_coco.ps1"
            Write-Host "✅ Full COCO dataset downloaded!" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Failed to download full COCO: $_" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "📊 Downloading COCO128 test dataset..." -ForegroundColor Green
        try {
            & ".\scripts\download_coco128.ps1"
            Write-Host "✅ COCO128 dataset downloaded!" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Failed to download COCO128: $_" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host ""
}

# Verify setup
Write-Host "🔍 Verifying setup..." -ForegroundColor Green

$allGood = $true

# Check models
if (Test-Path "models/yolov8n.pt") {
    Write-Host "✅ yolov8n.pt found" -ForegroundColor Green
} else {
    Write-Host "❌ yolov8n.pt not found" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path "models/yolov8s.pt") {
    Write-Host "✅ yolov8s.pt found" -ForegroundColor Green
} else {
    Write-Host "❌ yolov8s.pt not found" -ForegroundColor Red
    $allGood = $false
}

# Check datasets
if (Test-Path "data/coco128") {
    Write-Host "✅ COCO128 dataset found" -ForegroundColor Green
} elseif (Test-Path "data/coco") {
    Write-Host "✅ Full COCO dataset found" -ForegroundColor Green
} else {
    Write-Host "❌ No dataset found" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path "data/coco128.yaml") {
    Write-Host "✅ COCO128 config found" -ForegroundColor Green
} else {
    Write-Host "❌ COCO128 config not found" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

if ($allGood) {
    Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Test inference:     python src\inference.py --image <path_to_image>" -ForegroundColor White
    Write-Host "2. Quick training:     python src\train.py --quick" -ForegroundColor White
    Write-Host "3. Start Streamlit UI: streamlit run app_streamlit.py" -ForegroundColor White
    Write-Host "4. Start Gradio UI:    python app_gradio.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Training examples:" -ForegroundColor Yellow
    Write-Host "  python src\train.py --test                    # 2 epochs, fast test"
    Write-Host "  python src\train.py --quick                   # 5 epochs, quick training"
    Write-Host "  python src\train.py --epochs 20 --batch 8    # Custom training"
    Write-Host ""
    Write-Host "Export examples:" -ForegroundColor Yellow
    Write-Host "  python src\export.py --cpu-optimized         # Export to ONNX for CPU"
    Write-Host "  python src\export.py --mobile                # Export to TFLite for mobile"
} else {
    Write-Host "❌ Setup incomplete. Please check the errors above." -ForegroundColor Red
    exit 1
}
