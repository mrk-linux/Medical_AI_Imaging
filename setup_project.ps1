# ==========================================
# Medical AI Imaging Project Setup
# ==========================================

$ProjectName = "Medical_AI_Imaging"

Write-Host ""
Write-Host "Creating project structure..." -ForegroundColor Cyan

# Root
New-Item -ItemType Directory -Force -Path $ProjectName | Out-Null

# ----------------------------
# app
# ----------------------------
$folders = @(
    "app",
    "app\camera",
    "app\preprocessing",
    "app\models",
    "app\visualization",
    "app\utils",
    "app\tests",

    "data",
    "data\raw",
    "data\processed",
    "data\sample",

    "trained_models",

    "notebooks",

    "docs"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path "$ProjectName\$folder" | Out-Null
}

# ----------------------------
# __init__.py
# ----------------------------

$initFiles = @(
    "app",
    "app\camera",
    "app\preprocessing",
    "app\models",
    "app\visualization",
    "app\utils"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path "$ProjectName\$file\__init__.py" | Out-Null
}

# ----------------------------
# Python Files
# ----------------------------

$pythonFiles = @(
    "app\main.py",
    "app\config.py",

    "app\camera\camera.py",

    "app\preprocessing\preprocess.py",

    "app\models\model_loader.py",
    "app\models\inference.py",

    "app\visualization\display.py",

    "app\utils\helpers.py"
)

foreach ($file in $pythonFiles) {
    New-Item -ItemType File -Force -Path "$ProjectName\$file" | Out-Null
}

# ----------------------------
# Root Files
# ----------------------------

New-Item -ItemType File -Force -Path "$ProjectName\README.md" | Out-Null
New-Item -ItemType File -Force -Path "$ProjectName\requirements.txt" | Out-Null
New-Item -ItemType File -Force -Path "$ProjectName\.gitignore" | Out-Null
New-Item -ItemType File -Force -Path "$ProjectName\LICENSE" | Out-Null

Write-Host ""
Write-Host "Project created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Project Name: $ProjectName"