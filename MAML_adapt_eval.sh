cd /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/eval
export CUDA_VISIBLE_DEVICES=3
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python MAMLadapt_eval.py \
  --model_path ../model_results/new_MAML/equi_obj_5_8/model_meta_700.pth \
  --num_objects_min 9 \
  --num_objects_max 9 \
  --support_episodes 6 \
  --support_epsilon 0.2 \
  --inner_lr 1e-3 \
  --inner_steps 2 \
  --adapt_batch_size 2 \
  --n_episodes 300 \
  --no-headless
