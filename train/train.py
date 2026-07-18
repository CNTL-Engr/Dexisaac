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
from agent import DQNAgent
from maml_dqn import MAMLDQNAgent, ObstacleCountTaskGenerator
from env_wrapper import PushEnv
from project_paths import resolve_project_path
from utils import (
    format_contact_action,
    format_push_effectiveness,
    print_training_log,
    compute_epsilon,
    generate_checkpoint_dir,
)


DEFAULT_MODEL_PATH = 'model_results/PCA_judge/equi_obj_5_9/'
DEFAULT_CHECKPOINT_BASE_DIR = 'model_results/PCA_judge'


def parse_args():
    parser = argparse.ArgumentParser(description='MAML-DQN Training')

    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--num_objects_min', default=4, type=int, help='最小总物体数 (目标物体1个 + 障碍物)')
    parser.add_argument('--num_objects_max', default=7, type=int, help='最大总物体数 (目标物体1个 + 障碍物)')
    parser.add_argument('--episode_max_steps', default=8, type=int, help='每个 episode 最大步数')
    parser.add_argument(
        '--empty_push_displacement_threshold', default=0.01, type=float,
        help='有效推动所需的目标物体物理质心 XY 位移阈值（米，严格大于）'
    )
    parser.add_argument(
        '--empty_push_force_threshold', default=1.0, type=float,
        help='有效推动所需的任一夹爪手指对目标物体的峰值接触力阈值（N，严格大于）'
    )
    parser.add_argument(
        '--empty_push_rotation_arc_threshold', default=0.01, type=float,
        help='有效推动所需的目标物体旋转边缘位移 S=theta*D 阈值（米，严格大于）'
    )
    parser.add_argument(
        '--explosion_linear_speed_threshold', default=1.0, type=float,
        help='动力学崩飞的物体三维总线速度阈值（m/s，严格大于）'
    )
    parser.add_argument(
        '--explosion_linear_acceleration_threshold', default=50.0, type=float,
        help='动力学崩飞的物体三维总线加速度阈值（m/s^2，严格大于）'
    )
    parser.add_argument(
        '--explosion_abnormal_steps_threshold', default=5, type=int,
        help='判为崩飞所需的连续线速度异常物理步数'
    )
    parser.add_argument(
        '--explosion_acceleration_speed_step_window', default=5, type=int,
        help='加速度异常步到连续速度异常区间允许的最大物理步距'
    )
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--no-headless', dest='headless', default=False, action='store_false')
    parser.add_argument(
        '--algorithm',
        default='maml',
        choices=['maml', 'dqn'],
        help='训练方式: maml 使用当前 MAML-DQN；dqn 使用普通 DQN replay-buffer 训练'
    )

    # MAML 参数
    parser.add_argument('--n_meta_iterations', default=700, type=int, help='元迭代总数')
    parser.add_argument('--task_batch_size', default=4, type=int, help='每次元更新的任务数')
    parser.add_argument(
        '--task_sampling', default='fixed_plus_random',
        choices=['random', 'curriculum', 'curriculum_random', 'balanced', 'fixed_plus_random'],
        help='任务采样模式: random/curriculum/curriculum_random/balanced/fixed_plus_random'
    )
    parser.add_argument(
        '--fixed_task_obstacle_count',
        default=9,
        type=int,
        help='fixed_plus_random 模式下固定抽取的子任务障碍物数量'
    )
    parser.add_argument(
        '--fixed_task_count',
        default=2,
        type=int,
        help='fixed_plus_random 模式下固定子任务的抽取次数'
    )
    parser.add_argument('--curriculum_interval', default=175, type=int, help='课程等级切换间隔 (meta-iterations)')
    parser.add_argument('--support_episodes', default=2, type=int, help='每个任务的 Support 轮数')
    parser.add_argument('--query_episodes', default=1, type=int, help='每个任务的 Query 轮数')
    parser.add_argument('--inner_lr', default=1e-3, type=float, help='内循环学习率')
    parser.add_argument('--inner_steps', default=2, type=int, help='内循环梯度步数')
    parser.add_argument('--first_order', action='store_true', default=True, help='使用 FOMAML')
    parser.add_argument('--no_first_order', dest='first_order', action='store_false', help='使用完整 MAML (二阶)')

    # DQN/共享参数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='外循环学习率')
    parser.add_argument('--gamma', default=0.99, type=float, help='折扣因子')
    parser.add_argument('--epsilon_start', default=0.8, type=float, help='初始探索率')
    parser.add_argument('--epsilon_end', default=0.08, type=float, help='最终探索率')
    parser.add_argument('--epsilon_decay_steps', default=550, type=int, help='探索衰减步数')
    parser.add_argument('--epsilon_query', default=0.0, type=float, help='Query 阶段探索率，默认 0 表示完全 exploit')
    parser.add_argument('--target_update_freq', default=5, type=int, help='目标网络更新频率')
    parser.add_argument('--n_episodes', default=888, type=int, help='DQN 总训练轮数')
    parser.add_argument('--batch_size', default=16, type=int, help='DQN 训练批大小')
    parser.add_argument('--replay_buffer_size', default=12000, type=int, help='DQN 经验池大小')
    parser.add_argument('--min_buffer_size', default=16, type=int, help='DQN 日志显示的开始训练最小经验数')

    # 保存参数
    parser.add_argument('--save_every', default=50, type=int, help='保存频率 (meta-iterations)')
    parser.add_argument('--checkpoint_base_dir', default=DEFAULT_CHECKPOINT_BASE_DIR, type=str)
    parser.add_argument('--save_intermediate', action='store_true', default=True)

    # 模型加载参数
    parser.add_argument('--load_model', action='store_true', default=True)
    parser.add_argument('--no-load_model', '--no-load-model', dest='load_model', action='store_false')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str)
    parser.add_argument('--resume_meta_iter', default=0, type=int, help='从指定 meta-iteration 恢复')
    parser.add_argument('--resume_path', default='', type=str, help='显式指定恢复训练 checkpoint 路径')
    parser.add_argument('--use_equivariant', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', default=666, type=int)

    return parser.parse_args()


ACTION_NAMES = [
    "推目标 0°", "推目标 90°", "推目标 180°", "推目标 270°",
    "推障碍 0°", "推障碍 90°", "推障碍 180°", "推障碍 270°",
]


def make_force_task_config(task_generator, target_pos, robot_pos, obstacle_count, task_model_dirs):
    """为一个 episode 构造 task 约束；具体布局由 Scene 生成并回写。"""
    return {
        'obstacle_count': obstacle_count,
        'obstacle_model_dir': task_model_dirs['obstacle_dir'],
        'target_model_dir': task_model_dirs['target_dir'],
    }


def print_spawned_positions(spawned_objects, force_task_config, label="环境布局"):
    """打印环境生成时的物体位置信息"""
    target_pos = force_task_config.get('target_pos', [0.75, 0.0, 0.06])
    obstacle_positions = force_task_config.get('obstacle_positions')
    obstacle_count = force_task_config.get(
        'obstacle_count',
        len(obstacle_positions) if obstacle_positions is not None else 0
    )
    print(f"    [{label}] 目标位置: {target_pos}")
    print(f"    [{label}] 障碍物数量: {obstacle_count}")
    if obstacle_positions is None:
        print(f"      障碍物位置: 将由 Scene.create_clutter_environment 生成")
        return
    for i, pos in enumerate(obstacle_positions):
        print(f"      障碍物 {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")


def validate_fixed_plus_random_args(args):
    if args.task_sampling != 'fixed_plus_random':
        return None

    if args.fixed_task_count <= 0:
        raise ValueError('--fixed_task_count 必须大于 0')

    if args.fixed_task_count > args.task_batch_size:
        raise ValueError(
            '--fixed_task_count 不能大于 --task_batch_size: '
            f'{args.fixed_task_count} > {args.task_batch_size}'
        )

    min_obstacles = args.num_objects_min - 1
    max_obstacles = args.num_objects_max - 1
    obstacle_count = args.fixed_task_obstacle_count
    if obstacle_count < min_obstacles or obstacle_count > max_obstacles:
        raise ValueError(
            '--fixed_task_obstacle_count 必须位于 '
            f'[{min_obstacles}, {max_obstacles}]，'
            f'当前为 {obstacle_count}。'
            '注意: --num_objects_min/max 表示总物体数，'
            '--fixed_task_obstacle_count 表示障碍物数。'
        )

    fixed_task_id = obstacle_count - min_obstacles
    return [fixed_task_id] * args.fixed_task_count


def sample_train_task_batch(args, task_generator, curriculum_level, fixed_task_ids=None):
    if args.task_sampling != 'fixed_plus_random':
        return task_generator.sample_task_batch(
            args.task_batch_size,
            mode=args.task_sampling,
            curriculum_level=curriculum_level
        )

    task_ids = list(fixed_task_ids or [])
    random_count = args.task_batch_size - len(task_ids)
    task_ids.extend(
        random.randint(0, task_generator.NUM_TASKS - 1)
        for _ in range(random_count)
    )
    return task_ids


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
            if not info.get('already_done', False):
                print(f"        {format_contact_action(info)}")
                print(f"        {format_push_effectiveness(info)}")
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
            elif infos[env_idx].get('empty_push', False):
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


def run_maml_training(args):
    fixed_task_ids = validate_fixed_plus_random_args(args)
    print("=" * 80)
    network_type = "C4等变网络" if args.use_equivariant else "普通CNN网络"
    print(f"MAML-DQN 训练 ({network_type})")
    print(f"  任务采样: {args.task_sampling}, 任务批大小: {args.task_batch_size}")
    if args.task_sampling == 'fixed_plus_random':
        random_task_count = args.task_batch_size - len(fixed_task_ids)
        print(f"  固定障碍物数量: {args.fixed_task_obstacle_count}")
        print(f"  固定抽取次数: {args.fixed_task_count}")
        print(f"  固定 task_id: {fixed_task_ids}")
        print(f"  每轮随机补齐任务数: {random_task_count}")
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
        base_obstacle_count=args.num_objects_min - 1,
        radius=0.24
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
        curriculum_level = min(meta_iter // args.curriculum_interval, task_generator.NUM_TASKS - 1)

        # 采样任务批
        task_ids = sample_train_task_batch(
            args, task_generator, curriculum_level, fixed_task_ids=fixed_task_ids
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


def load_dqn_model_if_requested(agent, args):
    if not args.load_model:
        return
    if getattr(args, 'model_path_is_default', False):
        print("[Agent] DQN 分支跳过默认 MAML model_path；如需加载 DQN checkpoint，请显式传 --model_path")
        return
    if args.model_path and os.path.exists(args.model_path):
        print(f"[Agent] 加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'policy_net' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
            agent.target_net.load_state_dict(checkpoint['target_net'])
            if 'optimizer' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer'])
            print("[Agent] 模型加载成功 (checkpoint 格式)")
        else:
            agent.policy_net.load_state_dict(checkpoint)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print("[Agent] 模型加载成功 (权重格式)")
    else:
        print(f"⚠ 警告: 指定加载模型但路径无效或不存在: {args.model_path}")
        print("  将从头开始训练")


def run_dqn_training(args):
    print("=" * 80)
    network_type = "C4等变网络" if args.use_equivariant else "普通CNN网络"
    print(f"PushNet 强化学习训练（{network_type}）")
    print("=" * 80)
    print(f"Q 网络: {'EquivariantPushNet (C4等变)' if args.use_equivariant else 'CNNPushNet (非等变)'}")
    print("算法: DQN")
    print(f"环境数量: {args.num_envs}")
    print(f"Episode 总数: {args.n_episodes}")
    print(f"设备: {args.device}")
    print(f"无界面模式 (Headless): {args.headless}")
    print("=" * 80)

    print("\n[1/4] 初始化场景...")
    scene = Scene(description="DQN Training", num_envs=args.num_envs)

    print("[2/4] 创建环境...")
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps

    print("[3/4] 创建 DQN Agent...")
    agent = DQNAgent(
        device=args.device,
        lr=args.learning_rate,
        gamma=args.gamma,
        buffer_capacity=args.replay_buffer_size,
        use_equivariant=args.use_equivariant
    )
    load_dqn_model_if_requested(agent, args)

    print("[4/4] 开始训练...")
    print("=" * 80)

    global_step = 0
    episode_rewards_history = []
    train_loss_buffer = []
    action_counts = [0] * 8
    action_explore_counts = [0] * 8
    action_exploit_counts = [0] * 8
    success_count = 0
    ik_failed_count = 0
    valid_task_count = 0
    recent_100_env_results = deque(maxlen=100)

    csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'step', 'loss', 'reward'])
    print(f"[数据记录] 训练日志将保存到: {csv_path}")

    invalid_actions_list = [[] for _ in range(args.num_envs)]

    for episode in range(args.n_episodes):
        episode_retry_count = 0
        max_episode_retries = 5
        episode_valid = False
        step = 0
        episode_reward = 0.0
        env_rewards = [0.0] * args.num_envs
        infos = []

        while not episode_valid and episode_retry_count < max_episode_retries:
            episode_retry_count += 1
            ik_failed_this_episode = False
            episode_experiences = []

            if episode_retry_count == 1:
                print("\n" + "=" * 80)
                print(f"  Episode {episode+1}/{args.n_episodes}")
                print("=" * 80)
            else:
                print(f"\n  [重试 {episode_retry_count}/{max_episode_retries}] Episode {episode+1}")

            states, spawned_objects = env.reset()
            print("  [环境状态] 生成成功")
            torch.cuda.empty_cache()

            episode_reward = 0
            env_rewards = [0.0] * args.num_envs

            for step in range(args.episode_max_steps):
                pre_check_exploded = False
                for env_idx in range(args.num_envs):
                    _, out_reason, is_exploded = env._check_out_of_bounds(env_idx, spawned_objects)
                    if is_exploded:
                        print(f"\n  ⚠ [动作前检测] Env {env_idx} 物体已崩飞: {out_reason}")
                        pre_check_exploded = True
                        ik_failed_this_episode = True
                        break

                if pre_check_exploded:
                    print(f"  ✗ Episode {episode+1} 因动作前检测到崩飞而终止...")
                    episode_experiences.clear()
                    break

                epsilon = compute_epsilon(
                    global_step, args.epsilon_start,
                    args.epsilon_end, args.epsilon_decay_steps
                )

                actions = []
                strategy_types = []
                debug_print = (episode % 10 == 0) and (step == 0)

                for env_idx in range(args.num_envs):
                    state = states[env_idx:env_idx+1]
                    action, strategy_type = agent.select_action(
                        state, epsilon,
                        invalid_actions=invalid_actions_list[env_idx],
                        env_idx=env_idx,
                        debug=debug_print
                    )
                    actions.append(action)
                    strategy_types.append(strategy_type)

                    action_counts[action] += 1
                    if strategy_type == 'explore':
                        action_explore_counts[action] += 1
                    else:
                        action_exploit_counts[action] += 1

                torch.cuda.empty_cache()

                try:
                    next_states, rewards, dones, infos = env.step(actions, spawned_objects)
                except Exception as exc:
                    print("\n!!! 错误：env.step() 执行失败 !!!")
                    print(f"错误类型: {type(exc).__name__}")
                    print(f"错误信息: {str(exc)}")
                    import traceback
                    traceback.print_exc()
                    raise

                for env_idx in range(args.num_envs):
                    if infos[env_idx].get('ik_failed', False):
                        ik_failed_this_episode = True
                        ik_failed_count += 1
                        print(f"\n  ⚠ IK解算失败 (Env {env_idx})! 本Episode将终止并重试...")
                        break
                    if infos[env_idx].get('is_exploded', False):
                        ik_failed_this_episode = True
                        print(f"\n  ⚠ 崩飞 (Env {env_idx}: {infos[env_idx].get('out_reason', 'unknown')})! 本Episode将终止并重试...")
                        break

                if ik_failed_this_episode:
                    episode_experiences.clear()
                    break

                for env_idx in range(args.num_envs):
                    is_ik_failed = infos[env_idx].get('ik_failed', False)
                    is_exploded = infos[env_idx].get('is_exploded', False)

                    if not is_ik_failed and not is_exploded:
                        valid_task_count += 1
                        if infos[env_idx].get('success', False):
                            success_count += 1

                    if infos[env_idx].get('is_exploded', False):
                        print(f"  [跳过崩飞经验] Env {env_idx}: {infos[env_idx].get('out_reason', 'unknown')}")
                        continue

                    if dones[env_idx]:
                        invalid_actions_list[env_idx] = []
                    elif infos[env_idx].get('empty_push', False):
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

                    episode_experiences.append({
                        'state': state_cpu,
                        'action': actions[env_idx],
                        'reward': rewards[env_idx].item(),
                        'next_state': next_state_cpu,
                        'done': dones[env_idx].item()
                    })

                step_loss = None
                buffer_size = len(agent.replay_buffer)
                if buffer_size >= 4:
                    dynamic_batch_size = min(buffer_size, args.batch_size)
                    step_loss = agent.train_step(batch_size=dynamic_batch_size)
                    if step_loss is not None:
                        train_loss_buffer.append(step_loss)
                    torch.cuda.empty_cache()

                temp_step = global_step + step + 1
                if temp_step % args.target_update_freq == 0:
                    print(f"\n{'='*40}")
                    print(f"[目标网络更新] 预估step={temp_step}")
                    print(f"{'='*40}")
                    agent.update_target_network()
                    print(f"{'='*40}\n")

                print_training_log(
                    'step',
                    step=step+1,
                    max_steps=args.episode_max_steps,
                    infos=infos,
                    step_loss=step_loss,
                    epsilon=epsilon,
                    agent=agent,
                    min_buffer_size=args.min_buffer_size,
                    rewards=rewards,
                    actions=actions,
                    invalid_actions_list=invalid_actions_list,
                    dones=dones,
                    strategy_types=strategy_types
                )

                del states
                torch.cuda.empty_cache()

                states = next_states
                episode_reward += rewards.sum().item()
                for env_idx in range(args.num_envs):
                    env_rewards[env_idx] += rewards[env_idx].item()

                if dones.all():
                    print("\n  >> 所有环境已完成")
                    break

            if ik_failed_this_episode:
                print(f"  ✗ Episode {episode+1} 因IK失败而无效，正在重试...")
                episode_experiences.clear()
                torch.cuda.empty_cache()
                gc.collect()
                continue

            episode_valid = True
            for exp in episode_experiences:
                agent.store_transition(
                    state=exp['state'],
                    action=exp['action'],
                    reward=exp['reward'],
                    next_state=exp['next_state'],
                    done=exp['done']
                )
            episode_experiences.clear()

        if not episode_valid:
            print(f"\n  ⚠ Episode {episode+1} 重试 {max_episode_retries} 次后仍失败，跳过此回合")

        if episode_valid and infos:
            for env_idx in range(args.num_envs):
                if env_idx < len(infos):
                    recent_100_env_results.append(infos[env_idx].get('success', False))

        states = None
        rewards = None
        dones = None
        infos = None
        torch.cuda.empty_cache()
        gc.collect()

        episode_rewards_history.append(episode_reward)

        if episode_valid:
            global_step += step + 1
            episode_avg_loss = np.mean(train_loss_buffer[-step-1:]) if train_loss_buffer else 0.0
            csv_writer.writerow([episode + 1, global_step, episode_avg_loss, episode_reward])
            csv_file.flush()

        print_training_log(
            'episode',
            episode=episode+1,
            total_steps=step+1,
            max_steps=args.episode_max_steps,
            total_reward=episode_reward,
            buffer_size=len(agent.replay_buffer),
            buffer_capacity=args.replay_buffer_size,
            num_envs=args.num_envs,
            env_rewards=env_rewards,
            action_counts=action_counts,
            action_explore_counts=action_explore_counts,
            action_exploit_counts=action_exploit_counts,
            success_count=success_count,
            valid_task_count=valid_task_count,
            ik_failed_count=ik_failed_count,
            recent_100_env_results=recent_100_env_results
        )

        if (episode + 1) % args.save_every == 0:
            avg_reward_10 = np.mean(episode_rewards_history[-10:])
            print_training_log(
                'progress',
                episode=episode+1,
                total_episodes=args.n_episodes,
                avg_reward_10=avg_reward_10,
                global_step=global_step
            )
            if args.save_intermediate:
                csv_file.flush()
                print(f"✓ 训练数据已刷新到: {csv_path}")

        if args.save_intermediate and (episode + 1) % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f"model_episode_{episode+1}.pth")
            agent.save(save_path)
            print(f"\n✓ 中间模型已保存: {save_path}\n")

    final_path = os.path.join(args.checkpoint_dir, "model_final.pth")
    agent.save(final_path)
    csv_file.close()
    print(f"\n✓ 训练数据已保存到: {csv_path}")

    print("\n" + "=" * 80)
    print(f"训练完成！最终模型已保存到: {final_path}")
    print("=" * 80)

    scene.close()


def main():
    args = parse_args()
    args.model_path_is_default = args.model_path == DEFAULT_MODEL_PATH
    args.checkpoint_base_dir = resolve_project_path(args.checkpoint_base_dir)
    args.model_path = resolve_project_path(args.model_path)
    args.resume_path = resolve_project_path(args.resume_path)

    try:
        if args.algorithm == 'maml':
            validate_fixed_plus_random_args(args)
    except ValueError as exc:
        print(f"错误: {exc}")
        os._exit(1)

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

    if args.algorithm == 'dqn':
        run_dqn_training(args)
    else:
        run_maml_training(args)

    os._exit(0)


if __name__ == "__main__":
    main()
