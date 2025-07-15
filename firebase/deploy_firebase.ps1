# deploy_firebase.ps1
# Deploy Firebase configuration for Pawnder

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "pawnder-457917"
)

Write-Host "# Deploying Firebase configuration for Pawnder" -ForegroundColor Green
Write-Host "Project ID: $ProjectId" -ForegroundColor Yellow

# Set Firebase project
firebase use $ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-Host "# Failed to set Firebase project. Make sure you're logged in:" -ForegroundColor Red
    Write-Host "firebase login" -ForegroundColor Yellow
    exit 1
}

Write-Host "# Deploying Firestore rules..." -ForegroundColor Cyan
firebase deploy --only firestore:rules

Write-Host "# Deploying Firestore indexes..." -ForegroundColor Cyan  
firebase deploy --only firestore:indexes

Write-Host "# Deploying Storage rules..." -ForegroundColor Cyan
firebase deploy --only storage

Write-Host "# Firebase deployment complete!" -ForegroundColor Green
Write-Host "# Firebase Console: https://console.firebase.google.com/project/$ProjectId" -ForegroundColor Green
