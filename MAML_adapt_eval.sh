SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT/eval"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python MAMLadapt_eval.py \
  --model_path model_results/PCA_judge/equi_obj_5_8/model_final.pth \
  --num_objects_min 5 \
  --num_objects_max 5 \
  --support_episodes 2 \
  --support_epsilon 0.2 \
  --inner_lr 1e-3 \
  --inner_steps 2 \
  --adapt_batch_size 2 \
  --n_episodes 300 \
  --seeds 88888 \
  --no-headless
  #--save_depth_debug \
  #--save_adapted_model
  #--seed 38098
