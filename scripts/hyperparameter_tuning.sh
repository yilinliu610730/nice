#!/bin/bash

LEARNING_RATES=("1e-5" "1e-4" "1e-3")
MAX_SAMPLES=1600
OPTIMIZER="adam"
BATCH_SIZE=1

for lr in "${LEARNING_RATES[@]}"; do
  echo "Running training with LR=$lr, Samples=$MAX_SAMPLES, Optimizer=$OPTIMIZER, Batch Size=$BATCH_SIZE"
  python3 scripts/train_ofa.py --learning-rate $lr --max-samples $MAX_SAMPLES --optimizer $OPTIMIZER --batch-size $BATCH_SIZE
done

echo "All experiments completed."

OPTIMIZERS=("adam" "sgd")
LEARNING_RATE=1e-4
MAX_SAMPLES=1600
BATCH_SIZE=1

# Loop through each optimizer
for opt in "${OPTIMIZERS[@]}"; do
  echo "Running training with Optimizer=$opt, Learning Rate=$LEARNING_RATE, Samples=$MAX_SAMPLES, Batch Size=$BATCH_SIZE"
  python3 scripts/train_ofa.py --learning-rate $LEARNING_RATE --max-samples $MAX_SAMPLES --optimizer $opt --batch-size $BATCH_SIZE
done

echo "All experiments completed."


BATCH_SIZES=(1 2 4)
LEARNING_RATE=1e-4
OPTIMIZER="adam"

# Set iterations and initial gradient accumulation steps
ITERATIONS=200
INITIAL_GRADIENT_ACCUMULATION_STEPS=16

# Loop through each batch size
for bs in "${BATCH_SIZES[@]}"; do
  # Start with the initial gradient accumulation steps
  gradient_accumulation_steps=$INITIAL_GRADIENT_ACCUMULATION_STEPS
  
  while [ $gradient_accumulation_steps -gt 0 ]; do
    # Calculate max samples to ensure 200 iterations
    MAX_SAMPLES=$((ITERATIONS * bs * gradient_accumulation_steps))
    echo "Running training with Optimizer=$OPTIMIZER, Learning Rate=$LEARNING_RATE, Samples=$MAX_SAMPLES, Batch Size=$bs, Gradient Accumulation Steps=$gradient_accumulation_steps"
    
    python3 scripts/train_ofa.py --learning-rate $LEARNING_RATE --max-samples $MAX_SAMPLES --optimizer $OPTIMIZER --batch-size $bs --gradient-accumulation-steps $gradient_accumulation_steps
    
    if [ $? -eq 0 ]; then
      # If the script runs successfully, break the loop
      break
    else
      # If out of memory error occurs, reduce gradient accumulation steps
      echo "Out of memory error with Gradient Accumulation Steps=$gradient_accumulation_steps. Reducing Gradient Accumulation Steps."
      gradient_accumulation_steps=$((gradient_accumulation_steps / 2))
    fi
  done
  
  # If gradient accumulation steps reached 0, break the batch size loop
  if [ $gradient_accumulation_steps -eq 0 ]; then
    echo "Unable to run with Batch Size=$bs due to memory constraints."
  else
    break
  fi
done

echo "All experiments completed."