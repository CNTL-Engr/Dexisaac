SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd /data/kjy25/Projects/equi/IsaacLab
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PUBLIC_IP=172.21.100.11 LIVESTREAM=2 CUDA_VISIBLE_DEVICES=6 \
./isaaclab.sh -p scripts/Dexisaac/eval/MAMLadapt_eval.py \
  "$@" \
  --kit_args="--/app/livestream/fixedHostPort=47998" \
  --model_path model_results/PCA_judge/equi_obj_5_8/model_final.pth \
  --num_objects_min 9 \
  --num_objects_max 9 \
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
