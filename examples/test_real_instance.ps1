# Test script to verify real instance integration (PowerShell version)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Testing Real Instance Integration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Test instance
$INSTANCE = "1047675"
$TEST_CSV = "real_instances_50_weeks/$INSTANCE/test.csv"
$TRAIN_CSV = "real_instances_50_weeks/$INSTANCE/train.csv"

Write-Host "Test Instance: $INSTANCE"
Write-Host "Test CSV: $TEST_CSV"
Write-Host "Train CSV: $TRAIN_CSV"
Write-Host ""

# Test 1: OR baseline with real instance
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "Test 1: OR Baseline with Real Instance" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow

python or_csv_demo.py `
    --demand-file $TEST_CSV `
    --promised-lead-time 4 `
    --policy capped `
    --real-instance-train $TRAIN_CSV

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Test completed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
