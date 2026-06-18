cd /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/eval
export CUDA_VISIBLE_DEVICES=3
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python MAMLadapt_eval.py \
  --model_path ../model_results/PCA_judge/equi_obj_5_10/model_final.pth \
  --num_objects_min 11 \
  --num_objects_max 11 \
  --support_episodes 20 \
  --support_epsilon 0.2 \
  --inner_lr 1e-3 \
  --inner_steps 2 \
  --adapt_batch_size 2 \
  --n_episodes 300 \
  --seeds 33333 99999 66666 \
  #--no-headless
  #--save_depth_debug \
  #--save_adapted_model
  #--seed 38098
