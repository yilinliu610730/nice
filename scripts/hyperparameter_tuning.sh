#!/bin/bash

# Define hyperparameters to test
LEARNING_RATES=("5e-5" "1e-4" "1e-3")
MAX_SAMPLES=800
OPTIMIZER="adam"
BATCH_SIZE=4

# Loop through each learning rate
for lr in "${LEARNING_RATES[@]}"; do
  echo "Running training with LR=$lr, Samples=$MAX_SAMPLES, Optimizer=$OPTIMIZER, Batch Size=$BATCH_SIZE"
  python3 scripts/train_ofa.py --learning-rate $lr --max-samples $MAX_SAMPLES --optimizer $OPTIMIZER --batch-size $BATCH_SIZE
done

echo "All experiments completed."