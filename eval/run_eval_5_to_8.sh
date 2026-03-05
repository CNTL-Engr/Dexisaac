#!/usr/bin/env bash

# 对 PushNet 模型进行评估：
# 分别评估物体数量为 5, 6, 7, 8 的场景，每种场景评估 200 轮

set -e

# 当前脚本所在目录 (eval/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Dexisaac 根目录：脚本上一级目录
DEX_ROOT="${SCRIPT_DIR%/eval}"
# 模型结果根目录
MODEL_ROOT="${DEX_ROOT}/model_results"

cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EPISODES=200

for N in 5 6 7 8; do
  MODEL_PATH="${MODEL_ROOT}/equi_obj_9/model_final.pth"

  echo "============================================================"
  echo "  开始评估: 物体数量 = ${N}, 轮数 = ${EPISODES}"
  echo "  使用模型: ${MODEL_PATH}"
  echo "============================================================"

  python eval.py \
    --model_path "${MODEL_PATH}" \
    --n_episodes "${EPISODES}" \
    --num_objects_min "${N}" \
    --num_objects_max "${N}"
done

echo "所有评估完成。"

#使用方法
# python eval.py \
#   --model_path "${MODEL_ROOT}/equi_obj_${N}/model_final.pth" \
#   --n_episodes 200 \
#   --num_objects_min N \
#   --num_objects_max N