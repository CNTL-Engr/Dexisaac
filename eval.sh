cd /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/eval
export CUDA_VISIBLE_DEVICES=3
python eval.py --model_path /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/new_MAML/equi_obj_9/model_meta_200.pth --seeds 45596 77051 --n_episodes 300 --num_objects_min 9 --num_objects_max 9 --no-headless

