"""
MAML few-shot adaptation evaluation.

This script loads a meta-trained MAML-DQN checkpoint, collects a small
support set on the requested object-count task, adapts the policy with
MAMLDQNAgent.adapt(), and then reuses eval.py's evaluation loop with the
in-memory fast weights. It does not call meta_update().
"""

import argparse
import gc
import importlib.util
import os
import random
import sys

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
train_path = os.path.join(repo_dir, "train")
src_path = os.path.join(repo_dir, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)

from scene import Scene
from env_wrapper import PushEnv
from maml_dqn import MAMLDQNAgent, ObstacleCountTaskGenerator
from project_paths import resolve_project_path


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_module = _load_module("push_eval_module", os.path.join(current_dir, "eval.py"))
train_module = _load_module("maml_train_module", os.path.join(train_path, "train.py"))

run_evaluation_batch = eval_module.run_evaluation_batch
make_force_task_config = train_module.make_force_task_config
print_spawned_positions = train_module.print_spawned_positions
run_episode = train_module.run_episode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAML-DQN fast adaptation with in-memory fast weights"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="model_results/OBB_judge/equi_obj_5_8/model_meta_700.pth",
        help="Meta-trained MAML checkpoint path",
    )
    parser.add_argument(
        "--use_equivariant",
        action="store_true",
        default=True,
        help="Use the C4 equivariant PushNet",
    )

    parser.add_argument("--n_episodes", default=300, type=int, help="Query eval episodes")
    parser.add_argument("--n_batches", default=1, type=int, help="Auto-generated seed count")
    parser.add_argument("--seed", default=None, type=int, help="Single query seed")
    parser.add_argument("--seeds", default=None, type=int, nargs="+", help="Query seeds")
    parser.add_argument("--episode_max_steps", default=8, type=int)
    parser.add_argument(
        "--empty_push_displacement_threshold", default=0.01, type=float,
        help="有效推动所需的目标物体物理质心 XY 位移阈值（米，严格大于）",
    )
    parser.add_argument(
        "--empty_push_force_threshold", default=1.0, type=float,
        help="有效推动所需的任一夹爪手指峰值接触力阈值（N，严格大于）",
    )
    parser.add_argument(
        "--explosion_linear_speed_threshold", default=1.0, type=float,
        help="动力学崩飞的物体三维总线速度阈值（m/s，严格大于）",
    )
    parser.add_argument(
        "--explosion_linear_acceleration_threshold", default=50.0, type=float,
        help="动力学崩飞的物体三维总线加速度阈值（m/s^2，严格大于）",
    )
    parser.add_argument(
        "--explosion_abnormal_steps_threshold", default=5, type=int,
        help="判为崩飞所需的连续线速度异常物理步数",
    )
    parser.add_argument(
        "--explosion_acceleration_speed_step_window", default=5, type=int,
        help="加速度异常步到连续速度异常区间允许的最大物理步距",
    )

    parser.add_argument("--num_objects_min", default=9, type=int, help="Minimum total objects")
    parser.add_argument("--num_objects_max", default=9, type=int, help="Maximum total objects")
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    # isaaclab.sh 只负责把该参数转交给 Python。这里先接收但不消费，后续
    # Scene.initialize_app() 会从 sys.argv 再解析并交给 AppLauncher/Kit。
    parser.add_argument(
        "--kit_args",
        type=str,
        default="",
        help="Arguments forwarded verbatim by AppLauncher to Omniverse Kit",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Base log directory; default is eval/maml_adapt",
    )
    parser.add_argument(
        "--save_depth_debug",
        action="store_true",
        default=False,
        help="开启成功判定调试图保存；物理空推判定不再生成深度差分图，默认关闭",
    )
    parser.add_argument(
        "--save_adapted_model",
        action="store_true",
        default=False,
        help="Save the inner-loop adapted fast weights as a checkpoint",
    )
    parser.add_argument(
        "--adapted_model_dir",
        type=str,
        default=None,
        help="Directory for adapted checkpoints; default is <log_dir>/adapted_models",
    )

    parser.add_argument("--support_episodes", default=2, type=int)
    parser.add_argument("--support_epsilon", default=0.2, type=float)
    parser.add_argument("--support_seed_offset", default=100000, type=int)
    parser.add_argument("--support_max_retries", default=5, type=int)

    parser.add_argument("--inner_lr", default=1e-3, type=float)
    parser.add_argument("--inner_steps", default=2, type=int)
    parser.add_argument("--adapt_batch_size", default=4, type=int)
    parser.add_argument("--first_order", action="store_true", default=True)
    parser.add_argument("--no_first_order", dest="first_order", action="store_false")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_seed_list(args):
    if args.seeds and args.seed is not None:
        raise ValueError("--seed and --seeds cannot be used together")
    if args.seeds:
        return args.seeds
    if args.seed is not None:
        if args.n_batches > 1:
            raise ValueError("--seed supports one batch; use --seeds or omit --seed")
        return [args.seed]
    if args.n_batches <= 0:
        raise ValueError("--n_batches must be greater than 0")
    return np.random.choice(100000, size=args.n_batches, replace=False).tolist()


def load_checkpoint(agent, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if "policy_net" in checkpoint:
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        if "target_net" in checkpoint:
            agent.target_net.load_state_dict(checkpoint["target_net"])
        else:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("  MAML checkpoint loaded")
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("  Raw policy weights loaded")
    del checkpoint
    agent.policy_net.eval()
    agent.target_net.eval()
    torch.cuda.empty_cache()
    gc.collect()


def build_task_generator(args):
    min_obstacles = args.num_objects_min - 1
    max_obstacles = args.num_objects_max - 1
    num_tasks = max_obstacles - min_obstacles + 1
    return ObstacleCountTaskGenerator(
        num_tasks=num_tasks,
        base_obstacle_count=min_obstacles,
        radius=0.21,
    )


def sample_support_task(args, task_generator):
    total_objects = random.randint(args.num_objects_min, args.num_objects_max)
    obstacle_count = total_objects - 1
    task_id = obstacle_count - task_generator.BASE_OBSTACLE_COUNT
    task_model_dirs = task_generator.get_model_dirs(task_id)
    return total_objects, obstacle_count, task_model_dirs


def collect_support_transitions(args, env, agent, task_generator, seed):
    set_seed(seed)
    target_pos = [0.75, 0.0, 0.06]
    robot_pos = [0.0, 0.0, 0.0]
    support_transitions = []

    print("\n" + "=" * 80)
    print("  MAML Support Collection")
    print("=" * 80)
    print(f"  Support seed: {seed}")
    print(f"  Support episodes: {args.support_episodes}")
    print(f"  Support epsilon: {args.support_epsilon}")
    print(f"  Inner loop: lr={args.inner_lr}, steps={args.inner_steps}")

    for ep in range(args.support_episodes):
        total_objects, obstacle_count, task_model_dirs = sample_support_task(
            args, task_generator
        )
        force_task_config = make_force_task_config(
            task_generator, target_pos, robot_pos, obstacle_count, task_model_dirs
        )

        print(f"\n  [Support Episode {ep + 1}/{args.support_episodes}]")
        print(f"    Total objects: {total_objects} (obstacles={obstacle_count}, target=1)")
        print_spawned_positions(None, force_task_config, label=f"Support layout ep{ep + 1}")

        result = None
        for retry in range(args.support_max_retries):
            result = run_episode(
                env,
                agent,
                force_task_config,
                args.support_epsilon,
                args.episode_max_steps,
                fast_weights=None,
            )
            if result is not None:
                break
            print(
                f"    Support episode {ep + 1} retry "
                f"{retry + 1}/{args.support_max_retries} (IK/explode)"
            )
            torch.cuda.empty_cache()
            gc.collect()

        if result is None:
            print(f"    Support episode {ep + 1} failed; skip this seed")
            return None

        support_transitions.extend(result)
        print(f"    Collected transitions: {len(result)}")

    print(f"\n  Support total transitions: {len(support_transitions)}")
    return support_transitions


class FastWeightsEvalAgent:
    def __init__(self, maml_agent, fast_weights):
        self.maml_agent = maml_agent
        self.fast_weights = fast_weights

    def select_action(self, state, epsilon, invalid_actions=None, env_idx=0, debug=False):
        return self.maml_agent.select_action(
            state,
            epsilon,
            invalid_actions=invalid_actions,
            fast_weights=self.fast_weights,
        )


def detach_fast_weights(fast_weights):
    return {name: value.detach() for name, value in fast_weights.items()}


def save_adapted_checkpoint(
    args, agent, fast_weights, seed, support_seed, support_transition_count
):
    save_dir = args.adapted_model_dir or os.path.join(args.log_dir, "adapted_models")
    os.makedirs(save_dir, exist_ok=True)

    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    save_path = os.path.join(
        save_dir,
        f"{model_stem}_adapted_seed{seed}_support{support_seed}.pth",
    )
    adapted_state = {
        name: value.detach().cpu().clone()
        for name, value in agent.policy_net.state_dict().items()
    }
    adapted_state.update({
        name: value.detach().cpu().clone()
        for name, value in fast_weights.items()
    })
    torch.save(
        {
            "policy_net": adapted_state,
            "target_net": adapted_state,
            "adaptation": {
                "meta_checkpoint": args.model_path,
                "query_seed": seed,
                "support_seed": support_seed,
                "support_episodes": args.support_episodes,
                "support_epsilon": args.support_epsilon,
                "support_transitions": support_transition_count,
                "inner_lr": args.inner_lr,
                "inner_steps": args.inner_steps,
                "adapt_batch_size": args.adapt_batch_size,
                "first_order": args.first_order,
                "num_objects_min": args.num_objects_min,
                "num_objects_max": args.num_objects_max,
            },
        },
        save_path,
    )
    print(f"  [Adapt] Adapted model saved: {save_path}")
    return save_path


def append_adapt_metadata(
    log_path, args, support_seed, support_transition_count, adapted_model_path=None
):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        f.write("\n")
        f.write("# MAML adaptation config\n")
        f.write(f"support_seed,{support_seed}\n")
        f.write(f"support_episodes,{args.support_episodes}\n")
        f.write(f"support_epsilon,{args.support_epsilon}\n")
        f.write(f"support_transitions,{support_transition_count}\n")
        f.write(f"inner_lr,{args.inner_lr}\n")
        f.write(f"inner_steps,{args.inner_steps}\n")
        f.write(f"adapt_batch_size,{args.adapt_batch_size}\n")
        f.write(f"first_order,{args.first_order}\n")
        f.write("meta_update_called,False\n")
        if adapted_model_path:
            f.write(f"adapted_model_path,{adapted_model_path}\n")


def main():
    args = parse_args()
    args.model_path = resolve_project_path(args.model_path)
    if args.log_dir is not None:
        args.log_dir = resolve_project_path(args.log_dir)
    if args.adapted_model_dir is not None:
        args.adapted_model_dir = resolve_project_path(args.adapted_model_dir)

    if args.num_objects_min < 1 or args.num_objects_max < args.num_objects_min:
        print("ERROR: invalid object-count range")
        sys.exit(1)
    if args.support_episodes <= 0:
        print("ERROR: --support_episodes must be greater than 0")
        sys.exit(1)
    if args.n_episodes <= 0:
        print("ERROR: --n_episodes must be greater than 0")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"ERROR: model checkpoint does not exist: {args.model_path}")
        sys.exit(1)

    try:
        seed_list = build_seed_list(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.log_dir is None:
        args.log_dir = os.path.join(current_dir, "maml_adapt")

    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if args.headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")

    total_batches = len(seed_list)
    all_results = []

    print("\n" + "=" * 80)
    print("  MAML Few-Shot Adaptation Evaluation")
    print("=" * 80)
    print(f"  Meta checkpoint: {args.model_path}")
    print(f"  Query seeds: {seed_list}")
    print(f"  Query episodes per seed: {args.n_episodes}")
    print(f"  Object range: {args.num_objects_min}-{args.num_objects_max} total objects")
    print(f"  Logs: {args.log_dir}")
    print("=" * 80)

    print("\n[Init] Creating evaluation resources")
    print("  [1/4] Creating scene...")
    scene = Scene(description="MAML Adaptation Evaluation", num_envs=args.num_envs)

    print("  [2/4] Creating environment...")
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps

    print("  [3/4] Creating MAML-DQN agent...")
    agent = MAMLDQNAgent(
        device=args.device,
        lr=args.learning_rate,
        inner_lr=args.inner_lr,
        gamma=args.gamma,
        use_equivariant=args.use_equivariant,
        first_order=args.first_order,
        inner_steps=args.inner_steps,
    )

    print("  [4/4] Loading meta checkpoint...")
    load_checkpoint(agent, args.model_path, args.device)

    task_generator = build_task_generator(args)
    task_generator.print_model_dirs_config()

    for batch_idx, seed in enumerate(seed_list):
        support_seed = seed + args.support_seed_offset
        print("\n" + "=" * 80)
        print(f"  Seed {seed} ({batch_idx + 1}/{total_batches})")
        print("=" * 80)

        # support 采集阶段不保存深度调试图片，避免污染上一批 query 目录
        env.depth_debug_dir = None

        support_transitions = collect_support_transitions(
            args, env, agent, task_generator, support_seed
        )
        if not support_transitions:
            all_results.append(
                {
                    "seed": seed,
                    "success_rate": 0.0,
                    "log_path": "",
                    "status": "support_failed",
                }
            )
            continue

        print("\n  [Adapt] Computing fast_weights from support transitions...")
        fast_weights = agent.adapt(
            support_transitions,
            inner_steps=args.inner_steps,
            first_order=args.first_order,
            adapt_batch_size=args.adapt_batch_size,
        )
        fast_weights = detach_fast_weights(fast_weights)
        eval_agent = FastWeightsEvalAgent(agent, fast_weights)
        support_transition_count = len(support_transitions)
        adapted_model_path = None
        if args.save_adapted_model:
            adapted_model_path = save_adapted_checkpoint(
                args,
                agent,
                fast_weights,
                seed,
                support_seed,
                support_transition_count,
            )

        del support_transitions
        torch.cuda.empty_cache()
        gc.collect()

        print("  [Query] Evaluating adapted fast_weights...")
        success_rate, log_path = run_evaluation_batch(
            args,
            seed,
            env,
            eval_agent,
            batch_idx=batch_idx,
            total_batches=total_batches,
        )
        append_adapt_metadata(
            log_path,
            args,
            support_seed,
            support_transition_count,
            adapted_model_path=adapted_model_path,
        )

        all_results.append(
            {
                "seed": seed,
                "success_rate": success_rate,
                "log_path": log_path,
                "adapted_model_path": adapted_model_path,
                "status": "ok",
            }
        )

        del fast_weights, eval_agent
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("  MAML Adaptation Evaluation Summary")
    print("=" * 80)
    ok_results = [r for r in all_results if r["status"] == "ok"]
    for result in all_results:
        if result["status"] == "ok":
            print(
                f"  Seed {result['seed']:>6d}: "
                f"success_rate {result['success_rate']:.2f}% | log: {result['log_path']}"
            )
            if result.get("adapted_model_path"):
                print(f"            adapted_model: {result['adapted_model_path']}")
        else:
            print(f"  Seed {result['seed']:>6d}: support collection failed")
    if ok_results:
        avg_rate = sum(r["success_rate"] for r in ok_results) / len(ok_results)
        print("  ----------------------------------------")
        print(f"  average_success_rate: {avg_rate:.2f}%")
    print("=" * 80)

    os._exit(0)


if __name__ == "__main__":
    main()
