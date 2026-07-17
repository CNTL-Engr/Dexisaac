cd /data/kjy25/Projects/equi/IsaacLab

PUBLIC_IP=172.21.100.11 LIVESTREAM=2 CUDA_VISIBLE_DEVICES=6 \
  ./isaaclab.sh -p scripts/Dexisaac/debug/inspect_sim.py \
  --check push_point \
  --model_path meshdata/meshdata_target/622 \
  --action 0 1 2 3 4 5 6 7 \
  --kit_args="--/app/livestream/fixedHostPort=47998"
  "$@" \

