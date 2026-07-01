SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT/eval"
python eval.py --model_path model_results/new_MAML/equi_obj_9/model_meta_200.pth --seeds 45596 77051 --n_episodes 300 --num_objects_min 9 --num_objects_max 9 --no-headless
