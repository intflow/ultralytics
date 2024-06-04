#!/bin/bash

# Description of each step
# Step 1: Load target FP32 model
# Step 2: Add Q/DQ Layers to the YOLO model
# Step 3: Get Calibration values for the Q/DQ model
# Step 4: Load Q/DQ model with calibration values for QAT
# Step 5: Convert .pt to .onnx
# Step 6: Remove redundant Q/DQ layer
# Step 7: Remove Conv Q/DQ in DFL block
# Step 8: Test final model's accuracy

# Path to the configuration file
CONFIG_FILE="qat_setting.cfg"

# Steps to execute
#STEPS=("1" "2" "3" "4" "5" "6" "7" "8")
STEPS=("5" "6" "7" "8")

# Convert the steps array to a space-separated string
STEPS_STR="${STEPS[@]}"

# Run the Python script with specified steps and configuration file
python qat_pt2onnx.py --steps $STEPS_STR --config $CONFIG_FILE
