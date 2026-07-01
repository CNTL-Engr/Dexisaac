SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT/train"
export CUDA_VISIBLE_DEVICES=3
python train.py \
  --task_sampling fixed_plus_random \
  --task_batch_size 4 \
  --n_meta_iterations 150 \
  --epsilon_query 0.0 \
  "$@"
  #--resume_meta_iter 350 \
  #--resume_path model_results/OBB_judge/equi_obj_5_8/model_meta_350.pth
