# deploy_to_cloud.ps1
# PowerShell script to deploy your Pawnder ML model to Google Cloud Run
# Run this from VS Code terminal or PowerShell

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "pawnder-ml-api"
)

Write-Host "🚀 Deploying Pawnder ML API to Google Cloud Run" -ForegroundColor Green
Write-Host "Project ID: $ProjectId" -ForegroundColor Yellow
Write-Host "Region: $Region" -ForegroundColor Yellow
Write-Host "Service Name: $ServiceName" -ForegroundColor Yellow

# Set project
Write-Host "`n📋 Setting up Google Cloud project..." -ForegroundColor Cyan
gcloud config set project $ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to set project. Please check your project ID." -ForegroundColor Red
    exit 1
}

# Enable required APIs
Write-Host "`n🔌 Enabling required APIs..." -ForegroundColor Cyan
$apis = @(
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "containerregistry.googleapis.com",
    "artifactregistry.googleapis.com"
)

foreach ($api in $apis) {
    Write-Host "Enabling $api..." -ForegroundColor Gray
    gcloud services enable $api
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ Warning: Failed to enable $api" -ForegroundColor Yellow
    }
}

# Navigate to export directory
$exportDir = "C:\Users\kelly\Documents\GitHub\Pawnder\cloud_export"
if (!(Test-Path $exportDir)) {
    Write-Host "❌ Export directory not found: $exportDir" -ForegroundColor Red
    Write-Host "Please run export_model_safe.py first!" -ForegroundColor Red
    exit 1
}

Set-Location $exportDir
Write-Host "`n📁 Working in directory: $exportDir" -ForegroundColor Cyan

# Check required files
$requiredFiles = @("app.py", "Dockerfile", "requirements.txt", "model_metadata.json")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Required files missing:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Red
    }
    Write-Host "Please run export_model_safe.py first!" -ForegroundColor Red
    exit 1
}

# Check if saved_model directory exists
if (!(Test-Path "saved_model")) {
    Write-Host "❌ saved_model directory not found" -ForegroundColor Red
    Write-Host "Please run export_model_safe.py first!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ All required files found" -ForegroundColor Green

# Test the model locally first (optional)
Write-Host "`n🧪 Would you like to test the API locally first? (y/n)" -ForegroundColor Cyan
$testLocal = Read-Host
if ($testLocal -eq "y" -or $testLocal -eq "Y") {
    Write-Host "Starting local test server..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the local server and continue with deployment" -ForegroundColor Yellow
    try {
        python app.py
    } catch {
        Write-Host "Local test completed or interrupted" -ForegroundColor Gray
    }
}

# Create .gcloudignore file to exclude unnecessary files
$gcloudignore = @"
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache
.coverage
.tox
README.md
.DS_Store
*.log
.vscode
"@

$gcloudignore | Out-File -FilePath ".gcloudignore" -Encoding UTF8

# Build and deploy to Cloud Run
Write-Host "`n🏗️ Building and deploying to Cloud Run..." -ForegroundColor Cyan
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow

# Deploy with optimized settings for ML workloads
$deployCommand = @"
gcloud run deploy $ServiceName \
  --source . \
  --platform managed \
  --region $Region \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --port 8080 \
  --set-env-vars MODEL_PATH=/app/saved_model,METADATA_PATH=/app/model_metadata.json
"@

# Execute the deployment
Invoke-Expression $deployCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Deployment successful!" -ForegroundColor Green
    
    # Get service URL
    $serviceUrl = gcloud run services describe $ServiceName --region $Region --format "value(status.url)" 2>$null
    
    if ($serviceUrl) {
        Write-Host "`n🌐 Service URL: $serviceUrl" -ForegroundColor Green
        Write-Host "`n📋 Testing endpoints:" -ForegroundColor Cyan
        Write-Host "Health check: $serviceUrl/health" -ForegroundColor Gray
        Write-Host "Model info: $serviceUrl/model-info" -ForegroundColor Gray
        Write-Host "Prediction: $serviceUrl/predict (POST)" -ForegroundColor Gray
        
        Write-Host "`n🔧 Testing health endpoint..." -ForegroundColor Cyan
        try {
            $healthResponse = Invoke-RestMethod -Uri "$serviceUrl/health" -Method Get -TimeoutSec 30
            Write-Host "✅ Health check passed: $($healthResponse.status)" -ForegroundColor Green
            
            # Test model info endpoint
            Write-Host "`n🔧 Testing model info endpoint..." -ForegroundColor Cyan
            $modelInfo = Invoke-RestMethod -Uri "$serviceUrl/model-info" -Method Get -TimeoutSec 30
            Write-Host "✅ Model info retrieved successfully" -ForegroundColor Green
            Write-Host "   Model classes: $($modelInfo.model_info.class_names.Count)" -ForegroundColor Gray
            Write-Host "   Behavior input size: $($modelInfo.preprocessing.behavior_input_size)" -ForegroundColor Gray
            
        } catch {
            Write-Host "⚠️ Health check failed: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "The service might still be starting up. Try again in a few minutes." -ForegroundColor Yellow
        }
        
        Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Save this service URL for your FlutterFlow app:" -ForegroundColor Gray
        Write-Host "   $serviceUrl" -ForegroundColor White
        Write-Host "2. Test the prediction endpoint with a sample image" -ForegroundColor Gray
        Write-Host "3. Set up Firebase for your mobile app" -ForegroundColor Gray
        Write-Host "4. Configure FlutterFlow custom actions" -ForegroundColor Gray
        
        # Save deployment info
        $deploymentInfo = @{
            service_name = $ServiceName
            service_url = $serviceUrl
            project_id = $ProjectId
            region = $Region
            deployed_at = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            model_info = @{
                behavior_input_size = 46
                image_size = @(224, 224, 3)
                class_names = @(
                    "Aggressive/Threatening",
                    "Curiosity/Alertness", 
                    "Fearful/Anxious",
                    "Happy/Playful",
                    "Relaxed",
                    "Stressed",
                    "Submissive/Appeasement"
                )
            }
            endpoints = @{
                health = "$serviceUrl/health"
                predict = "$serviceUrl/predict"
                model_info = "$serviceUrl/model-info"
            }
        }
        
        $deploymentInfo | ConvertTo-Json -Depth 4 | Out-File -FilePath "deployment_info.json" -Encoding UTF8
        Write-Host "`n💾 Deployment info saved to deployment_info.json" -ForegroundColor Green
        
        # Create a simple test script
        $testScript = @"
# test_api.py - Simple test script for your deployed API
import requests
import base64
import json

API_URL = "$serviceUrl"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")

def test_model_info():
    response = requests.get(f"{API_URL}/model-info")
    print(f"Model info: {response.json()}")

def test_prediction_with_sample():
    # This would need a real base64 encoded image
    print("To test prediction, you need to provide a base64 encoded dog image")
    print("Example usage:")
    print("  data = {'image': 'base64_encoded_image_here'}")
    print("  response = requests.post(f'{API_URL}/predict', json=data)")

if __name__ == "__main__":
    test_health()
    test_model_info()
    test_prediction_with_sample()
"@
        
        $testScript | Out-File -FilePath "test_api.py" -Encoding UTF8
        Write-Host "📄 Test script created: test_api.py" -ForegroundColor Green
        
    } else {
        Write-Host "⚠️ Could not retrieve service URL" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "`n❌ Deployment failed!" -ForegroundColor Red
    Write-Host "Common issues and solutions:" -ForegroundColor Yellow
    Write-Host "1. Check if all required APIs are enabled" -ForegroundColor Gray
    Write-Host "2. Verify your Google Cloud project has billing enabled" -ForegroundColor Gray
    Write-Host "3. Make sure you have the required permissions" -ForegroundColor Gray
    Write-Host "4. Check the Cloud Build logs in the Google Cloud Console" -ForegroundColor Gray
    exit 1
}

Write-Host "`n🎉 Cloud Run deployment process completed!" -ForegroundColor Green
Write-Host "Your Pawnder ML API is now running in the cloud! 🐕" -ForegroundColor Green