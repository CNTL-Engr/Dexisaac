#!/bin/bash
# CUDA 显存优化配置
# expandable_segments: 减少内存碎片化
# max_split_size_mb: 限制单次分配的最大块大小，避免大块分配失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

cd /home/k/Projects/equi/IsaacLab/scripts/workspace/train
python train.py "$@"  --no-headless