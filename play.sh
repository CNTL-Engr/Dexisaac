cd /data/kjy25/Projects/equi/IsaacLab

PUBLIC_IP=172.21.100.11 LIVESTREAM=2 CUDA_VISIBLE_DEVICES=6 \
  ./isaaclab.sh -p scripts/Dexisaac/tests/inspect_sim.py \
  --check push_point \
  --target_object meshdata/meshdata_target/005 \
  --num_objects_min 5 \
  --num_objects_max 9 \
  --action 0 1 2 3 4 5 6 7 \
  --kit_args="--/app/livestream/fixedHostPort=47998" \
  "$@"
