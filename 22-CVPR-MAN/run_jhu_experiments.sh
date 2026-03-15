#!/bin/bash

# Ensure conda activates smoothly in shell scripts
eval "$(conda shell.bash hook)"
conda activate p2pnet

# Define the dataset path and GPU
DATA_DIR="../JHU_Processed"
GPU_ID="0"

# Loop through the alpha values
for alpha in 0.25 0.5 1.0; do
    echo "=========================================================="
    echo "Starting MAN training on JHU++ with NPoint Alpha: $alpha"
    echo "=========================================================="
    
    # We use a unique save-dir prefix for each training session
    SAVE_DIR="model/jhu_alpha_${alpha}"
    LOG_FILE="jhu_alpha_${alpha}_log.txt"
    
    # Run training in the foreground so the loop waits for it to finish!
    nohup python train.py --dataset jhu \
                    --data-dir "$DATA_DIR" \
                    --device "$GPU_ID" \
                    --save-dir "$SAVE_DIR" \
                    --alpha "$alpha" > "$LOG_FILE" 2>&1
                    
    echo "Finished training for Alpha $alpha. Logs saved to $LOG_FILE"
done

echo "🎉 All experiments completed successfully!"
