#!/bin/bash
# eval_exp.sh - 模型评估实验脚本

# CUDA 显存优化配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# 切换到脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 默认参数，可通过命令行覆盖
python eval_exp.py "$@"
