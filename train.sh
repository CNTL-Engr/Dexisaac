cd /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/train
export CUDA_VISIBLE_DEVICES=1
python train.py \
  --task_sampling random \
  --task_batch_size 4 \
  --n_meta_iterations 700 \
  --epsilon_query 0.0 \
  --resume_meta_iter 350 \
  --resume_path /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/new_MAML/equi_obj_5_8/model_meta_350.pth
