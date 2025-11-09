#!/bin/bash
# Test script to verify real instance integration

echo "=========================================="
echo "Testing Real Instance Integration"
echo "=========================================="
echo ""

# Test instance
INSTANCE="1047675"
TEST_CSV="real_instances_50_weeks/${INSTANCE}/test.csv"
TRAIN_CSV="real_instances_50_weeks/${INSTANCE}/train.csv"

echo "Test Instance: ${INSTANCE}"
echo "Test CSV: ${TEST_CSV}"
echo "Train CSV: ${TRAIN_CSV}"
echo ""

# Test 1: OR baseline with real instance
echo "=========================================="
echo "Test 1: OR Baseline with Real Instance"
echo "=========================================="
python or_csv_demo.py \
    --demand-file "${TEST_CSV}" \
    --promised-lead-time 4 \
    --policy capped \
    --real-instance-train "${TRAIN_CSV}"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
