#!/bin/bash
# ==============================================================================
# 批量评估脚本
# 使用 eval/eval.py 进行多次评估，支持自定义评估次数、轮次和物体数量
#
# 使用方法:
#   chmod +x eval/batch_eval.sh
#   bash eval/batch_eval.sh
#
# 修改下方参数即可自定义评估配置
# ==============================================================================

# ==================== 可修改参数 ====================

# 评估次数（运行 eval.py 的总次数）
NUM_EVALS=3

# 每次评估的轮数（每次运行多少个 episode）
N_EPISODES=500

# 物体数量范围
NUM_OBJECTS_MIN=4
NUM_OBJECTS_MAX=5

# 每轮最大步数
EPISODE_MAX_STEPS=8

# 模型路径
MODEL_PATH="/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/model_results/equi_obj_9/model_final.pth"

# 是否使用等变网络（默认开启）
USE_EQUIVARIANT="--use_equivariant"

# 日志保存目录（留空则默认保存到 eval/ 目录）
LOG_DIR=""

# ==================== 执行逻辑 ====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.py"

echo "========================================================================"
echo "  批量评估 - 共 ${NUM_EVALS} 次"
echo "========================================================================"
echo "  每次轮数: ${N_EPISODES}"
echo "  物体数量: ${NUM_OBJECTS_MIN} ~ ${NUM_OBJECTS_MAX}"
echo "  最大步数: ${EPISODE_MAX_STEPS}"
echo "  模型路径: ${MODEL_PATH}"
echo "========================================================================"

for i in $(seq 1 $NUM_EVALS); do
    # 每次使用不同的随机种子
    SEED=$((RANDOM * 10 + i))

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  第 ${i}/${NUM_EVALS} 次评估  (Seed: ${SEED})"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    CMD="python ${EVAL_SCRIPT} \
        --model_path ${MODEL_PATH} \
        --n_episodes ${N_EPISODES} \
        --seed ${SEED} \
        --episode_max_steps ${EPISODE_MAX_STEPS} \
        --num_objects_min ${NUM_OBJECTS_MIN} \
        --num_objects_max ${NUM_OBJECTS_MAX} \
        ${USE_EQUIVARIANT}"

    # 如果设置了日志目录
    if [ -n "${LOG_DIR}" ]; then
        CMD="${CMD} --log_dir ${LOG_DIR}"
    fi

    echo "  运行命令: ${CMD}"
    echo ""

    eval ${CMD}

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ❌ 第 ${i} 次评估异常退出 (exit code: ${EXIT_CODE})"
    else
        echo "  ✓ 第 ${i} 次评估完成"
    fi
done

echo ""
echo "========================================================================"
echo "  全部 ${NUM_EVALS} 次评估已完成"
echo "========================================================================"
