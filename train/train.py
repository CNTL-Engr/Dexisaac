"""
MAML-DQN 训练脚本 (标准 MAML 算法)
使用 Meta-Iteration 结构: 采样任务批 → 收集 Support → 适应 → 收集 Query → 元更新
"""

import os
import time
import sys
import argparse
import gc
import random
import torch
import numpy as np
from collections import deque
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
train_path = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)

from scene import Scene
from maml_dqn import MAMLDQNAgent, ObstacleCountTaskGenerator
from env_wrapper import PushEnv
from utils import compute_epsilon, generate_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser(description='MAML-DQN Training')

    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--num_objects_min', default=4, type=int, help='最小障碍物数 (即 BASE_OBSTACLE_COUNT)')
    parser.add_argument('--num_objects_max', default=7, type=int, help='最大障碍物数 (即 BASE_OBSTACLE_COUNT + NUM_TASKS - 1)')
    parser.add_argument('--episode_max_steps', default=8, type=int, help='每个 episode 最大步数')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--no-headless', dest='headless', default=False, action='store_false')

    # MAML 参数
    parser.add_argument('--n_meta_iterations', default=700, type=int, help='元迭代总数')
    parser.add_argument('--task_batch_size', default=4, type=int, help='每次元更新的任务数')
    parser.add_argument(
        '--task_sampling', default='random',
        choices=['random', 'curriculum', 'curriculum_random', 'balanced'],
        help='任务采样模式: random/curriculum/curriculum_random/balanced'
    )
    parser.add_argument('--curriculum_interval', default=175, type=int, help='课程等级切换间隔 (meta-iterations)')
    parser.add_argument('--support_episodes', default=2, type=int, help='每个任务的 Support 轮数')
    parser.add_argument('--query_episodes', default=1, type=int, help='每个任务的 Query 轮数')
    parser.add_argument('--inner_lr', default=1e-3, type=float, help='内循环学习率')
    parser.add_argument('--inner_steps', default=2, type=int, help='内循环梯度步数')
    parser.add_argument('--first_order', action='store_true', default=True, help='使用 FOMAML')
    parser.add_argument('--no_first_order', dest='first_order', action='store_false', help='使用完整 MAML (二阶)')

    # DQN 参数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='外循环学习率')
    parser.add_argument('--gamma', default=0.99, type=float, help='折扣因子')
    parser.add_argument('--epsilon_start', default=0.8, type=float, help='初始探索率')
    parser.add_argument('--epsilon_end', default=0.08, type=float, help='最终探索率')
    parser.add_argument('--epsilon_decay_steps', default=550, type=int, help='探索衰减步数 (meta-iterations)')
    parser.add_argument('--epsilon_query', default=0.0, type=float, help='Query 阶段探索率，默认 0 表示完全 exploit')
    parser.add_argument('--target_update_freq', default=5, type=int, help='目标网络更新频率 (meta-iterations)')

    # 保存参数
    parser.add_argument('--save_every', default=50, type=int, help='保存频率 (meta-iterations)')
    parser.add_argument('--checkpoint_base_dir', default='/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/new_MAML', type=str)
    parser.add_argument('--save_intermediate', action='store_true', default=True)

    # 模型加载参数
    parser.add_argument('--load_model', action='store_true', default=True)
    parser.add_argument('--model_path', default='/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/pre_equi_obj_4/model_episode_3000.pth', type=str)
    parser.add_argument('--resume_meta_iter', default=0, type=int, help='从指定 meta-iteration 恢复')
    parser.add_argument('--resume_path', default='', type=str, help='显式指定恢复训练 checkpoint 路径')
    parser.add_argument('--use_equivariant', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', default=525, type=int)

    return parser.parse_args()


ACTION_NAMES = [
    "推目标 0°", "推目标 90°", "推目标 180°", "推目标 270°",
    "推障碍 0°", "推障碍 90°", "推障碍 180°", "推障碍 270°",
]


def make_force_task_config(task_generator, target_pos, robot_pos, obstacle_count, task_model_dirs):
    """为一个 episode 采样 task 内布局。retry 时复用返回的配置。"""
    obstacle_positions = task_generator.generate_positions_for_task(
        target_pos, robot_pos, obstacle_count
    )
    return {
        'target_pos': target_pos,
        'obstacle_positions': obstacle_positions,
        'obstacle_model_dir': task_model_dirs['obstacle_dir'],
        'target_model_dir': task_model_dirs['target_dir'],
    }


def print_spawned_positions(spawned_objects, force_task_config, label="环境布局"):
    """打印环境生成时的物体位置信息"""
    print(f"    [{label}] 目标位置: {force_task_config['target_pos']}")
    print(f"    [{label}] 障碍物数量: {len(force_task_config['obstacle_positions'])}")
    for i, pos in enumerate(force_task_config['obstacle_positions']):
        print(f"      障碍物 {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


def run_episode(env, agent, force_task_config, epsilon, max_steps, fast_weights=None):
    """
    运行单个 episode，返回 transitions 列表。
    如果发生 IK 失败或崩飞，返回 None。
    """
    states, spawned_objects = env.reset(force_task_config=force_task_config)

    transitions = []
    invalid_actions_list = [[] for _ in range(env.num_envs)]

    for step in range(max_steps):
        # 动作前崩飞检测
        exploded = False
        for env_idx in range(env.num_envs):
            _, _, is_exploded = env._check_out_of_bounds(env_idx, spawned_objects)
            if is_exploded:
                print(f"      Step {step+1}: 崩飞检测触发, episode 终止")
                exploded = True
                break
        if exploded:
            return None

        # 选择动作
        actions = []
        strategy_types = []
        for env_idx in range(env.num_envs):
            state = states[env_idx:env_idx+1]
            action, strategy = agent.select_action(
                state, epsilon,
                invalid_actions=invalid_actions_list[env_idx],
                fast_weights=fast_weights
            )
            actions.append(action)
            strategy_types.append(strategy)

        torch.cuda.empty_cache()

        # 打印动作选择信息
        for env_idx in range(env.num_envs):
            masked_str = ""
            if invalid_actions_list[env_idx]:
                masked_str = f", 屏蔽={[ACTION_NAMES[a] for a in invalid_actions_list[env_idx]]}"
            print(f"      Step {step+1} Env{env_idx}: "
                  f"动作={ACTION_NAMES[actions[env_idx]]} ({strategy_types[env_idx]})"
                  f"{masked_str}")

        # 执行动作
        try:
            next_states, rewards, dones, infos = env.step(actions, spawned_objects)
        except Exception:
            print(f"      Step {step+1}: env.step() 异常, episode 终止")
            return None

        # 检查 IK 失败或崩飞
        for env_idx in range(env.num_envs):
            if infos[env_idx].get('ik_failed', False):
                print(f"      Step {step+1} Env{env_idx}: IK 失败, episode 终止")
                return None
            if infos[env_idx].get('is_exploded', False):
                print(f"      Step {step+1} Env{env_idx}: 崩飞, episode 终止")
                return None

        # 打印奖励信息
        for env_idx in range(env.num_envs):
            reward_val = rewards[env_idx].item()
            info = infos[env_idx]
            parts = []
            if 'reward_breakdown' in info:
                for k, v in info['reward_breakdown'].items():
                    if v != 0:
                        parts.append(f"{k}={v:+.2f}")
            success_str = " [成功]" if info.get('success', False) else ""
            empty_str = " [空推]" if info.get('empty_push', False) else ""
            done_str = " [结束]" if dones[env_idx] else ""
            breakdown = f" ({', '.join(parts)})" if parts else ""
            print(f"        → 奖励={reward_val:+.2f}{breakdown}{success_str}{empty_str}{done_str}")

        # 存储 transitions 并更新无效动作列表
        for env_idx in range(env.num_envs):
            if dones[env_idx]:
                invalid_actions_list[env_idx] = []

            if infos[env_idx].get('empty_push', False):
                if actions[env_idx] not in invalid_actions_list[env_idx]:
                    invalid_actions_list[env_idx].append(actions[env_idx])
            else:
                invalid_actions_list[env_idx] = []

            state_cpu = states[env_idx].cpu()
            next_state_cpu = next_states[env_idx].cpu()
            if state_cpu.dtype != torch.uint8:
                state_cpu = (state_cpu * 255).to(torch.uint8)
            if next_state_cpu.dtype != torch.uint8:
                next_state_cpu = (next_state_cpu * 255).to(torch.uint8)

            transitions.append((state_cpu, actions[env_idx], rewards[env_idx].item(),
                                next_state_cpu, dones[env_idx].item()))

        del states
        torch.cuda.empty_cache()
        states = next_states

        if dones.all():
            break

    return transitions


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.checkpoint_dir = generate_checkpoint_dir(
        args.checkpoint_base_dir, args.use_equivariant,
        args.num_objects_min, args.num_objects_max
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if args.headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")

    print("=" * 80)
    network_type = "C4等变网络" if args.use_equivariant else "普通CNN网络"
    print(f"MAML-DQN 训练 ({network_type})")
    print(f"  任务采样: {args.task_sampling}, 任务批大小: {args.task_batch_size}")
    print(f"  内循环: lr={args.inner_lr}, steps={args.inner_steps}, FOMAML={args.first_order}")
    print(f"  元迭代总数: {args.n_meta_iterations}")
    print("=" * 80)

    # 1. 创建场景和环境
    print("\n[1/3] 初始化场景...")
    scene = Scene(description="MAML-DQN Training", num_envs=args.num_envs)
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps

    # 2. 创建 Agent 和任务生成器
    print("[2/3] 创建 MAML-DQN Agent...")
    agent = MAMLDQNAgent(
        device=args.device,
        lr=args.learning_rate,
        inner_lr=args.inner_lr,
        gamma=args.gamma,
        use_equivariant=args.use_equivariant,
        first_order=args.first_order,
        inner_steps=args.inner_steps
    )
    num_tasks = args.num_objects_max - args.num_objects_min + 1
    task_generator = ObstacleCountTaskGenerator(
        num_tasks=num_tasks,
        base_obstacle_count=args.num_objects_min,
        radius=0.21
    )
    task_generator.print_model_dirs_config()

    # 模型加载
    if args.resume_path or args.resume_meta_iter > 0:
        if args.resume_path and args.resume_meta_iter <= 0:
            print("错误: 使用 --resume_path 时也需要指定 --resume_meta_iter，避免无法确定续训起点")
            os._exit(1)
        resume_path = args.resume_path or os.path.join(args.checkpoint_dir, f"model_meta_{args.resume_meta_iter}.pth")
        if os.path.exists(resume_path):
            agent.load(resume_path)
            print(f"[Agent] 恢复训练: meta-iter {args.resume_meta_iter} ({resume_path})")
        else:
            print(f"错误: 恢复模型不存在: {resume_path}")
            os._exit(1)
    elif args.load_model and args.model_path and os.path.exists(args.model_path):
        print(f"[Agent] 加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'policy_net' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            agent.target_net.load_state_dict(checkpoint['target_net'])
        else:
            agent.policy_net.load_state_dict(checkpoint)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("[Agent] 模型加载成功")

    # 3. 训练循环
    print("[3/3] 开始 MAML 训练...")
    print("=" * 80)

    start_meta_iter = args.resume_meta_iter
    target_pos = [0.75, 0.0, 0.06]
    robot_pos = [0.0, 0.0, 0.0]

    # CSV 日志
    csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    csv_header = ['meta_iter', 'meta_loss', 'avg_reward', 'epsilon_support', 'epsilon_query', 'curriculum_level']
    csv_use_legacy_epsilon_column = False
    if start_meta_iter > 0 and os.path.exists(csv_path):
        with open(csv_path, newline='') as existing_csv:
            existing_header = next(csv.reader(existing_csv), [])
        csv_use_legacy_epsilon_column = 'epsilon_query' not in existing_header
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

    recent_results = deque(maxlen=100)
    meta_losses = []

    for meta_iter in range(start_meta_iter, args.n_meta_iterations):
        iter_start = time.time()

        # 课程等级 (仅 curriculum 模式使用)
        curriculum_level = min(meta_iter // args.curriculum_interval, ObstacleCountTaskGenerator.NUM_TASKS - 1)

        # 采样任务批
        task_ids = task_generator.sample_task_batch(
            args.task_batch_size, mode=args.task_sampling, curriculum_level=curriculum_level
        )

        # Support 负责收集适应用数据，保留按 meta-iteration 衰减的探索。
        epsilon_support = compute_epsilon(
            meta_iter, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
        )

        print(f"\n{'='*80}")
        print(f"[Meta-Iteration {meta_iter+1}/{args.n_meta_iterations}] "
              f"Tasks={task_ids}, EpsilonSupport={epsilon_support:.3f}, "
              f"EpsilonQuery={args.epsilon_query:.3f}, Curriculum={curriculum_level}")
        print(f"{'='*80}")

        task_batch_data = []
        iter_reward = 0.0
        iter_successes = 0
        iter_episodes = 0

        for task_idx, task_id in enumerate(task_ids):
            obstacle_count, task_name = task_generator.get_task_by_id(task_id)
            task_model_dirs = task_generator.get_model_dirs(task_id)

            print(f"\n  [Task {task_idx+1}/{len(task_ids)}] {task_name} (id={task_id})")

            # --- 收集 Support Set (使用基础策略 θ + 探索) ---
            print(f"    --- Support Set (base policy + epsilon_support={epsilon_support:.3f}) ---")
            support_transitions = []
            support_ok = True
            for ep in range(args.support_episodes):
                force_task_config = make_force_task_config(
                    task_generator, target_pos, robot_pos, obstacle_count, task_model_dirs
                )
                print_spawned_positions(None, force_task_config, label=f"Support布局 ep{ep+1}")
                result = None
                for retry in range(5):
                    result = run_episode(env, agent, force_task_config, epsilon_support,
                                         args.episode_max_steps, fast_weights=None)
                    if result is not None:
                        break
                    print(f"    Support episode {ep+1} retry {retry+1}/5 (IK/explode)")
                    torch.cuda.empty_cache()
                    gc.collect()
                if result is None:
                    print(f"    Support episode {ep+1} failed after 5 retries, skipping task")
                    support_ok = False
                    break
                support_transitions.extend(result)
                iter_episodes += 1

            if not support_ok or not support_transitions:
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # --- Inner-loop 适应: θ' = θ - α∇L(support) ---
            fast_weights = agent.adapt(support_transitions)
            print(f"    Support: {len(support_transitions)} transitions, adapted")

            # --- 收集 Query Set (使用适应后策略 θ' + held-out 布局) ---
            print(f"    --- Query Set (adapted policy + epsilon_query={args.epsilon_query:.3f}) ---")
            query_transitions = []
            query_ok = True
            for ep in range(args.query_episodes):
                force_task_config = make_force_task_config(
                    task_generator, target_pos, robot_pos, obstacle_count, task_model_dirs
                )
                print_spawned_positions(None, force_task_config, label=f"Query布局 ep{ep+1}")
                result = None
                for retry in range(5):
                    result = run_episode(env, agent, force_task_config, args.epsilon_query,
                                         args.episode_max_steps, fast_weights=fast_weights)
                    if result is not None:
                        break
                    print(f"    Query episode {ep+1} retry {retry+1}/5 (IK/explode)")
                    torch.cuda.empty_cache()
                    gc.collect()
                if result is None:
                    print(f"    Query episode {ep+1} failed after 5 retries, skipping task")
                    query_ok = False
                    break
                query_transitions.extend(result)
                iter_episodes += 1

            if not query_ok or not query_transitions:
                del fast_weights
                torch.cuda.empty_cache()
                gc.collect()
                continue

            print(f"    Query: {len(query_transitions)} transitions")

            # 统计奖励
            task_reward = sum(t[2] for t in query_transitions)
            iter_reward += task_reward
            task_successes = sum(1 for t in query_transitions if t[4] > 0.5)
            iter_successes += task_successes

            task_batch_data.append({'support': support_transitions, 'query': query_transitions})

            del fast_weights
            torch.cuda.empty_cache()
            gc.collect()

        # --- Meta-Update: θ ← θ - β∇_θ Σ L_Ti(f_θ') ---
        if task_batch_data:
            meta_loss = agent.meta_update(task_batch_data)
            meta_losses.append(meta_loss)
            print(f"\n  >> Meta-Update: loss={meta_loss:.4f}, tasks={len(task_batch_data)}/{args.task_batch_size}")
        else:
            meta_loss = 0.0
            print(f"\n  >> No valid tasks this iteration, skipping meta-update")

        # 目标网络更新
        if (meta_iter + 1) % args.target_update_freq == 0:
            agent.update_target_network()
            print(f"  >> Target network updated")

        # 记录
        avg_reward = iter_reward / max(len(task_batch_data), 1)
        recent_results.append(avg_reward)
        iter_elapsed = time.time() - iter_start

        if csv_use_legacy_epsilon_column:
            csv_writer.writerow([meta_iter + 1, meta_loss, avg_reward, epsilon_support, curriculum_level])
        else:
            csv_writer.writerow([
                meta_iter + 1, meta_loss, avg_reward,
                epsilon_support, args.epsilon_query, curriculum_level
            ])
        csv_file.flush()

        print(f"\n  Reward={avg_reward:.2f}, Time={iter_elapsed:.1f}s, "
              f"Recent100={np.mean(list(recent_results)):.2f}")

        # 保存
        if args.save_intermediate and (meta_iter + 1) % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f"model_meta_{meta_iter+1}.pth")
            agent.save(save_path)
            print(f"  >> Model saved: {save_path}")

    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "model_final.pth")
    agent.save(final_path)
    csv_file.close()

    print(f"\n{'='*80}")
    print(f"训练完成! 最终模型: {final_path}")
    print(f"{'='*80}")
    os._exit(0)


if __name__ == "__main__":
    main()
