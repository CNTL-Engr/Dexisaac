SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd /data/kjy25/Projects/equi/IsaacLab
PUBLIC_IP=172.21.100.11 LIVESTREAM=2 CUDA_VISIBLE_DEVICES=6 \
./isaaclab.sh -p scripts/Dexisaac/train/train.py \
  --task_sampling random \
  --task_batch_size 4 \
  --n_meta_iterations 300 \
  --epsilon_query 0.0 \
  --no-load_model \
  "$@"
  #--resume_meta_iter 350 \
  #--resume_path model_results/OBB_judge/equi_obj_5_8/model_meta_350.pth
  #--algorithm dqn