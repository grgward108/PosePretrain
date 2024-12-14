#!/bin/bash

# Usage function
usage() {
  echo "Usage: $0 --script [spatial|temporal] --exp_name EXP_NAME"
  echo "  --script     Specify the script to run: 'spatial' or 'temporal'."
  echo "  --exp_name   Specify the experiment name for the training run."
  echo
  echo "Example:"
  echo "  $0 --script spatial --exp_name test_run_06"
  echo "  $0 --script temporal --exp_name ABC"
  exit 1
}

# Default values
SCRIPT=""
EXP_NAME=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --script)
      SCRIPT="$2"
      shift 2
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

# Validate arguments
if [[ -z "$SCRIPT" || -z "$EXP_NAME" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

# Run the appropriate script
if [[ "$SCRIPT" == "spatial" ]]; then
  echo "Running spatial pretrain with experiment name: $EXP_NAME"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train_spatial_transformer.py --exp_name "$EXP_NAME"
elif [[ "$SCRIPT" == "temporal" ]]; then
  echo "Running temporal pretrain with experiment name: $EXP_NAME"
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train_temporal_lazyloading.py --exp_name "$EXP_NAME" --multi-gpu
else
  echo "Error: Invalid script name. Use 'spatial' or 'temporal'."
  usage
fi
