cd /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/eval
export CUDA_VISIBLE_DEVICES=1
python eval.py --model_path /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/bounding_box_judge_success/2_envs/equi_obj_5_8/model_final.pth --n_episodes 300 --num_objects_min 8 --num_objects_max 8 --seed 51270 #--no-headless

