"""
C6PushNet 强化学习训练脚本
使用 DQN 算法训练 EquivariantPushNet (C6等变) 策略网络
"""

import os
import time
import sys
import argparse
import gc
import torch
import numpy as np
from pathlib import Path

# 添加 src 和 train 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
train_path = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)

from scene import Scene
from maml_dqn import MAMLDQNAgent, ObstacleCountTaskGenerator
from env_wrapper import PushEnv
from utils import print_training_log, compute_epsilon, generate_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train EquivariantPushNet with DQN')
    
    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--num_objects_min', default=5, type=int, help='最小物体数')
    parser.add_argument('--num_objects_max', default=8, type=int, help='最大物体数') # 物体数由 MAML Task 控制, 仅用于命名检查点文件夹
    parser.add_argument('--episode_max_steps', default=8, type=int, help='每个 episode 最大步数')
    parser.add_argument('--headless', action='store_true', default=True, help='无界面模式 (默认开启)')
    parser.add_argument('--no-headless', dest='headless',default=False, action='store_false', help='启用可视化界面')
    
    # 训练参数
    parser.add_argument('--n_episodes', default=2800, type=int, help='总训练轮数')
    parser.add_argument('--curriculum_interval', default=700, type=int, help='每隔多少轮增加一个障碍物')
    parser.add_argument('--batch_size', default=16, type=int, help='训练批大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
    parser.add_argument('--gamma', default=0.99, type=float, help='折扣因子')
    parser.add_argument('--epsilon_start', default=0.8, type=float, help='初始探索率')
    parser.add_argument('--epsilon_end', default=0.08, type=float, help='最终探索率')
    parser.add_argument('--epsilon_decay_steps', default=2800, type=int, help='探索衰减步数')
    parser.add_argument('--target_update_freq', default=10, type=int, help='目标网络更新频率(步数)')
    parser.add_argument('--replay_buffer_size', default=18000, type=int, help='经验池大小')
    parser.add_argument('--min_buffer_size', default=16, type=int, help='开始训练的最小经验数')
    
    # 保存参数
    parser.add_argument('--save_every', default=50, type=int, help='保存频率(episodes)')
    parser.add_argument('--checkpoint_base_dir', default='/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/bounding_box_judge_success/2_envs', type=str, help='检查点根目录（将自动生成子目录名）')
    parser.add_argument('--save_intermediate', action='store_true', default=True, help='是否保存中间模型')
    
    # 模型加载参数
    parser.add_argument('--load_model', action='store_true', default=True, help='是否加载预训练模型')
    parser.add_argument('--model_path', default='/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/model_results/pre_equi_obj_4/model_final.pth', type=str, help='预训练模型路径')
    parser.add_argument('--resume_episode', default=0, type=int, help='从指定episode恢复训练（0=从头开始）')
    parser.add_argument('--use_equivariant', action='store_true', default=True, help='是否使用C4等变网络（默认开启）')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # [自动生成] 根据训练参数生成检查点目录
    args.checkpoint_dir = generate_checkpoint_dir(
        args.checkpoint_base_dir,
        args.use_equivariant,
        args.num_objects_min,
        args.num_objects_max
    )
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 确保启用相机
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    
    # 同步 headless 参数到 sys.argv，确保 AppLauncher 能识别
    if args.headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")
    
    print("=" * 80)
    network_type = "C4等变网络" if args.use_equivariant else "普通CNN网络"
    print(f"PushNet 强化学习训练（{network_type}）")
    print("=" * 80)
    print(f"Q 网络: {'EquivariantPushNet (C4等变)' if args.use_equivariant else 'CNNPushNet (非等变)'}")
    print(f"算法: DQN")
    print(f"环境数量: {args.num_envs}")
    print(f"Episode 总数: {args.n_episodes}")
    print(f"设备: {args.device}")
    print(f"无界面模式 (Headless): {args.headless}")
    print("=" * 80)
    
    # 1. 创建场景
    print("\n[1/4] 初始化场景...")
    scene = Scene(description="DQN Training", num_envs=args.num_envs)
    
    # 2. 创建环境包装器
    print("[2/4] 创建环境...")
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps
    
    # 3. 创建 MAML DQN Agent 和任务生成器
    print("[3/4] 创建 MAML DQN Agent...")
    agent = MAMLDQNAgent(
        device=args.device,
        lr=args.learning_rate,
        gamma=args.gamma,
        use_equivariant=args.use_equivariant
    )
    
    task_generator = ObstacleCountTaskGenerator(radius=0.21)
    
    '''
    # [可选] 为每个子任务指定不同的物体模型文件夹，取消注释下面的代码并修改路径:
    task_generator.set_task_model_dirs({
        0: {'obstacle_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_3obs', 'target_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_target'},
        1: {'obstacle_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_4obs', 'target_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_target'},
        2: {'obstacle_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_5obs', 'target_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_target'},
        3: {'obstacle_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_6obs', 'target_dir': '/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_target'},
    })
    '''

    # 打印当前模型文件夹配置
    task_generator.print_model_dirs_config()
    
    current_task_id = 0
    task_batch_size = 1  # 单子任务课程学习，每次只有 1 个子任务
    prev_curriculum_level = -1  # 用于检测课程等级切换
    
    # [模型加载] 如果指定加载预训练模型
    if args.resume_episode > 0:
        # 恢复训练模式：从checkpoint恢复完整状态（含optimizer）
        resume_path = os.path.join(args.checkpoint_dir, f"model_episode_{args.resume_episode}.pth")
        if os.path.exists(resume_path):
            agent.load(resume_path)
            print(f"[Agent] ✓ 恢复训练: 从 Episode {args.resume_episode} 继续 ({resume_path})")
        else:
            print(f"❌ 错误: 恢复模型不存在: {resume_path}")
            os._exit(1)
    elif args.load_model:
        if args.model_path and os.path.exists(args.model_path):
            print(f"[Agent] 加载预训练模型: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=args.device)
            
            # 检测 checkpoint 格式
            if 'policy_net' in checkpoint:
                # 完整 checkpoint 格式（包含 policy_net, target_net, optimizer等）
                agent.policy_net.load_state_dict(checkpoint['policy_net'])
                agent.target_net.load_state_dict(checkpoint['target_net'])
                print("[Agent] ✓ 模型加载成功 (checkpoint 格式)")
            else:
                # 纯网络权重格式
                agent.policy_net.load_state_dict(checkpoint)
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print("[Agent] ✓ 模型加载成功 (权重格式)")
        else:
            print(f"⚠ 警告: 指定加载模型但路径无效或不存在: {args.model_path}")
            print("  将从头开始训练")
    
    # 4. 训练循环
    print("[4/4] 开始训练...")
    print("=" * 80)
    
    # [恢复训练] 根据 resume_episode 估算已有的 global_step
    start_episode = args.resume_episode
    # 每个 episode 平均步数 ≈ episode_max_steps/2，用于估算 global_step
    estimated_steps_per_episode = args.episode_max_steps * args.num_envs
    global_step = start_episode * estimated_steps_per_episode // 2  # 保守估计
    
    # 恢复课程等级和 epsilon 对应的 task_start_step
    if start_episode > 0:
        resume_curriculum = min(start_episode // args.curriculum_interval, 3)
        # 找到当前课程等级开始的 episode，然后估算 task_start_step
        curriculum_start_ep = resume_curriculum * args.curriculum_interval
        task_start_step = curriculum_start_ep * estimated_steps_per_episode // 2
        prev_curriculum_level = resume_curriculum
        current_task_id = start_episode // 2  # 每2个episode一个task
        print(f"[恢复训练] Episode={start_episode}, 课程等级={resume_curriculum}, "
              f"估算global_step={global_step}, task_start_step={task_start_step}")
    else:
        task_start_step = 0
    
    episode_rewards_history = [] # Renamed to avoid conflict with episode_reward in loop
    train_loss_buffer = [] # Buffer for storing training losses
    
    # 动作选择统计 (8个动作: 0-7)
    action_counts = [0] * 8  # 总次数
    action_explore_counts = [0] * 8  # 探索次数
    action_exploit_counts = [0] * 8  # 利用次数
    success_count = 0  # 记录总成功次数
    ik_failed_count = 0  # 记录IK失败次数（不计入成功率）
    valid_task_count = 0  # 记录有效任务次数（排除IK失败）
    
    # 记录最近100次环境任务的成功/失败（每个环境算一次任务）
    from collections import deque
    import csv
    recent_100_env_results = deque(maxlen=100)  # 每个元素是True/False，表示单个环境的成功/失败
    
    # CSV 数据记录
    csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    os.makedirs(args.checkpoint_dir, exist_ok=True)  # 确保目录存在
    if start_episode > 0 and os.path.exists(csv_path):
        # 恢复训练时追加写入
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        print(f"[数据记录] 训练日志追加到: {csv_path}")
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['episode', 'step', 'loss', 'reward'])  # 写入表头
        print(f"[数据记录] 训练日志将保存到: {csv_path}")
    
    # 为每个环境维护一个无效动作列表 (Invalid Action Suppression)
    invalid_actions_list = [[] for _ in range(args.num_envs)]

    for episode in range(start_episode, args.n_episodes):
        # Episode重试循环（IK失败时重新开始）
        episode_retry_count = 0
        max_episode_retries = 5  # 最大重试次数
        episode_valid = False
        # 初始化变量，确保在 while/for 循环后始终有绑定值
        step = 0
        episode_reward = 0.0
        env_rewards = [0.0] * args.num_envs
        infos: list = []
        
        while not episode_valid and episode_retry_count < max_episode_retries:
            episode_retry_count += 1
            ik_failed_this_episode = False  # 标记本episode是否发生IK失败
            episode_experiences = []  # 临时存储本episode的经验
            
            # [MAML] 决定当前是 Support 还是 Query 阶段
            is_support = (episode % 2 == 0)
            
            # [课程学习] 根据 episode 决定当前障碍物数量等级
            curriculum_level = min(episode // args.curriculum_interval, 3)
            sampled_task_id = curriculum_level
            obstacle_count, task_name = task_generator.get_task_by_id(sampled_task_id)
            
            if is_support:
                if episode_retry_count == 1:
                    current_task_id += 1
                
                phase_name = "Support"
                
                # [课程学习] 检测课程等级切换
                if episode_retry_count == 1 and curriculum_level != prev_curriculum_level:
                    prev_curriculum_level = curriculum_level
                    task_start_step = global_step  # 重置相对步数，用于探索率重置
                    print("\n" + "*" * 80)
                    print(f"[课程学习] 等级切换 Level {curriculum_level}: 障碍物数量 = {obstacle_count} (Episode {episode+1})")
                    print(f"  Task {sampled_task_id} ({task_name})")
                    remaining = args.n_episodes - episode
                    print(f"  剩余训练轮数: {remaining}")
                    print("*" * 80)
                
                if episode_retry_count == 1:
                    meta_iter = (current_task_id - 1) // task_batch_size + 1
                    print("\n" + "=" * 80)
                    print(f"[Outer-Loop: Meta-Iteration {meta_iter}] 收集子任务 (课程等级 {curriculum_level}, {obstacle_count} 障碍物)")
                    print("=" * 80)
            else:
                phase_name = "Query"
                
            # [MAML] 生成任务布局
            target_pos = [0.75, 0.0, 0.06]
            robot_pos = [0.0, 0.0, 0.0]  # 假设基座原点
            obstacle_positions = task_generator.generate_positions_for_task(target_pos, robot_pos, obstacle_count)
            
            # 获取当前子任务对应的模型文件夹路径
            task_model_dirs = task_generator.get_model_dirs(sampled_task_id)
            
            force_task_config = {
                'target_pos': target_pos,
                'obstacle_positions': obstacle_positions,
                'obstacle_model_dir': task_model_dirs['obstacle_dir'],
                'target_model_dir': task_model_dirs['target_dir'],
            }
            
            # Episode 开始标题
            if episode_retry_count == 1:
                print("\n" + "-" * 60)
                print(f"  [课程等级 {curriculum_level}] 全局任务进度: {current_task_id} | Task {sampled_task_id} ({task_name}), 障碍物数量: {obstacle_count}")
                print(f"  [{'Inner-Loop: Support Set' if is_support else 'Outer-Loop: Query Set'}] 收集数据 (Episode {episode+1}/{args.n_episodes})")
                print("-" * 60)
            else:
                print(f"\n  [重试 {episode_retry_count}/{max_episode_retries}] Episode {episode+1}")
            
            # 重置环境 (传入 force_task_config)
            reset_start = time.time()
            states, spawned_objects = env.reset(force_task_config=force_task_config)
            reset_elapsed = time.time() - reset_start
            if reset_elapsed > 60:
                print(f"\n  ⚠ [超时] env.reset() 耗时 {reset_elapsed:.1f}s > 60s，强制重试...")
                ik_failed_this_episode = True
                episode_experiences.clear()
                if hasattr(env.scene, '_global_spawn_config'):
                    del env.scene._global_spawn_config
                torch.cuda.empty_cache()
                gc.collect()
                continue  # 重试 while 循环
            print(f"  [环境状态] 生成成功 (MAML {phase_name} Set)")
            
            # [MAML] 如果是 Query 阶段，获取 fast_weights 用于推理
            fast_weights = None
            if not is_support and current_task_id in agent.replay_buffer.buffer:
                support_data = agent.replay_buffer.buffer[current_task_id]['support']
                if support_data:
                    print(f"\n  >> [Inner-Loop] 正在执行快速适应 (Fast Weights Adaptation) ...")
                    fast_weights = agent.adapt(support_data, first_order=True)
                    print(f"  >> [Inner-Loop] 适应完成，获取到 Task-Specific 策略参数。")

            # [显存优化] Reset后立即清理GPU缓存
            torch.cuda.empty_cache()
            
            episode_reward = 0
            # [新增] 每个环境的奖励跟踪
            env_rewards = [0.0] * args.num_envs  # 每个环境的累计奖励
            
            for step in range(args.episode_max_steps):
                step_start_time = time.time()  # 记录每步开始时间

                # [新增] 动作前崩飞检测 - 检查物体是否已经崩飞
                pre_check_exploded = False
                for env_idx in range(args.num_envs):
                    is_out, out_reason, is_exploded = env._check_out_of_bounds(env_idx, spawned_objects)
                    if is_exploded:
                        print(f"\n  ⚠ [动作前检测] Env {env_idx} 物体已崩飞: {out_reason}")
                        pre_check_exploded = True
                        ik_failed_this_episode = True  # 复用标志
                        break
                
                if pre_check_exploded:
                    print(f"  ✗ Episode {episode} 因动作前检测到崩飞而终止...")
                    episode_experiences.clear()
                    break

                # 保存调试图像 (每10步)
                # if global_step % 1 == 0:
                #     for i in range(args.num_envs):
                #         save_debug_images(global_step, states, env_idx=i)

                # 计算当前epsilon (相对于当前子任务重置探索率)
                relative_step = global_step - task_start_step
                epsilon = compute_epsilon(relative_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
                
                # 选择动作（单流网络：只返回 u, v, direction）
                actions = []
                strategy_types = []  # 记录每个环境的策略类型
                
                # [调试] 每10个episode打印状态和Q值
                debug_print = (episode % 10 == 0) and (step == 0)
                
                for env_idx in range(args.num_envs):
                    state = states[env_idx:env_idx+1]  # (1, 3, 320, 320)
                    
                    action, strategy_type = agent.select_action(state, epsilon, invalid_actions=invalid_actions_list[env_idx], fast_weights=fast_weights)
                    actions.append(action)
                    strategy_types.append(strategy_type)
                    
                    # [新增] 统计动作选择（分探索/利用）
                    action_counts[action] += 1
                    if strategy_type == 'explore':
                        action_explore_counts[action] += 1
                    else:  # 'exploit'
                        action_exploit_counts[action] += 1
                
                # [激进内存优化] 每个环境选择完动作后清理GPU缓存
                torch.cuda.empty_cache()
                
                # 执行动作（添加异常捕获）
                try:
                    next_states, rewards, dones, infos = env.step(actions, spawned_objects)
                except Exception as e:
                    print(f"\n!!! 错误：env.step() 执行失败 !!!")
                    print(f"错误类型: {type(e).__name__}")
                    print(f"错误信息: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise  # 重新抛出异常以便调试
                
                # [超时检测] 单步执行超过60秒则判定卡死
                step_elapsed = time.time() - step_start_time
                if step_elapsed > 60:
                    print(f"\n  ⚠ [超时] Step {step+1} 耗时 {step_elapsed:.1f}s > 60s，清理缓存并重试本Episode...")
                    ik_failed_this_episode = True
                    episode_experiences.clear()
                    # 清理场景缓存，确保下次重新生成
                    if hasattr(env.scene, '_global_spawn_config'):
                        del env.scene._global_spawn_config
                    torch.cuda.empty_cache()
                    gc.collect()
                    break  # 退出步骤循环，进入 while 重试
                
                # [关键] 检查是否有任何环境发生IK失败或崩飞
                for env_idx in range(args.num_envs):
                    if infos[env_idx].get('ik_failed', False):
                        ik_failed_this_episode = True
                        ik_failed_count += 1
                        print(f"\n  ⚠ IK解算失败 (Env {env_idx})! 本Episode将终止并重试...")
                        break
                    if infos[env_idx].get('is_exploded', False):
                        ik_failed_this_episode = True  # 复用这个标志，因为处理逻辑相同
                        print(f"\n  ⚠ 崩飞 (Env {env_idx}: {infos[env_idx].get('out_reason', 'unknown')})! 本Episode将终止并重试...")
                        break
                
                # 如果发生IK失败或崩飞，立即退出内层步骤循环
                if ik_failed_this_episode:
                    # 清理临时经验
                    episode_experiences.clear()
                    break
                
                # 3. 处理反馈与无效动作管理
                for env_idx in range(args.num_envs):
                    # [修改] 统计成功次数（排除IK失败和崩飞）
                    is_ik_failed = infos[env_idx].get('ik_failed', False)
                    is_exploded = infos[env_idx].get('is_exploded', False)
                    
                    if not is_ik_failed and not is_exploded:
                        valid_task_count += 1
                        if infos[env_idx].get('success', False):
                            success_count += 1
                    
                    # [修复] 如果done，清空无效动作列表（准备下一轮）
                    if dones[env_idx]:
                        invalid_actions_list[env_idx] = []
                    
                    # 跳过崩飞的经验
                    if infos[env_idx].get('is_exploded', False):
                        print(f"  [跳过崩飞经验] Env {env_idx}: {infos[env_idx].get('out_reason', 'unknown')}")
                        continue
                    
                    # 如果是空推，将此动作加入屏蔽列表
                    if infos[env_idx].get('empty_push', False):
                        if actions[env_idx] not in invalid_actions_list[env_idx]:
                            invalid_actions_list[env_idx].append(actions[env_idx])
                    else:
                        invalid_actions_list[env_idx] = []
                    
                    # [临时存储] 将经验存到临时列表，episode成功完成后再批量提交
                    # [内存优化] 立即转移到CPU并转换为uint8，避免GPU内存泄漏
                    state_cpu = states[env_idx].cpu()
                    next_state_cpu = next_states[env_idx].cpu()
                    
                    # 如果不是uint8，转换为uint8节省内存
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

                
                # [MAML] 替换为 Meta-Update 逻辑
                # 将本 Episode 存入 Buffer，但在 Episode 结束后批量存入
                # 所以每步的在线训练被移除，改为 Episode 级别的 Meta-Update
                step_loss = None
                
                # 更新目标网络（基于有效步数）
                # 注意：global_step在episode有效结束后才更新，这里用临时计算的step数
                temp_step = global_step + step + 1  # 预估当前步数
                if temp_step % args.target_update_freq == 0:
                    print(f"\n{'='*40}")
                    print(f"[目标网络更新] 预估step={temp_step}")
                    print(f"{'='*40}")
                    agent.update_target_network()
                    print(f"{'='*40}\n")
                
                # 打印每步信息
                print_training_log('step',
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
                
                # [显存优化] 明确删除旧状态，释放GPU显存
                del states
                torch.cuda.empty_cache()
                
                # 更新状态
                states = next_states
                episode_reward += rewards.sum().item()
                for env_idx in range(args.num_envs):
                    env_rewards[env_idx] += rewards[env_idx].item()
                
                # 检查是否所有环境都结束
                if dones.all():
                    print(f"\n  >> 所有环境已完成")
                    break
            
            # Episode内层循环结束后检查
            if ik_failed_this_episode:
                # IK失败，不提交经验，继续重试
                print(f"  ✗ Episode {episode+1} 因IK失败而无效，正在重试...")
                episode_experiences.clear()
                torch.cuda.empty_cache()
                gc.collect()  # [内存优化] 重试前清理内存
                continue  # 重试while循环
            else:
                # Episode有效，提交所有临时经验
                episode_valid = True
                for exp in episode_experiences:
                    agent.replay_buffer.push(
                        task_id=current_task_id,
                        is_support=is_support,
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        next_state=exp['next_state'],
                        done=exp['done']
                    )
                episode_experiences.clear()
                
                # [MAML] 在 Query 阶段结束后，严格每收集齐 task_batch_size 个任务执行一次 Meta Update
                if not is_support and current_task_id % task_batch_size == 0:
                    print(f"\n" + "=" * 80)
                    meta_iter = current_task_id // task_batch_size
                    print(f"[Outer-Loop: Meta-Iteration {meta_iter}] 收集完毕。执行元更新 (Meta-Update) ...")
                    print(f"  >> 当前 Buffer Task 数量: {len(agent.replay_buffer)}")
                    sampled_tasks = agent.replay_buffer.sample_tasks(task_batch_size)
                    meta_loss = agent.meta_update(sampled_tasks, first_order=True)  # 默认降级为FOMAML以防escnn报错
                    print(f"  >> Meta Loss: {meta_loss:.4f}")
                    train_loss_buffer.append(meta_loss)
                    print("=" * 80 + "\n")
        
        # 如果重试次数用尽仍然失败
        if not episode_valid:
            print(f"\n  ⚠ Episode {episode+1} 重试 {max_episode_retries} 次后仍失败，跳过此回合")
        
        # [修改] 记录每个环境的成功/失败（只记录有效episode）
        if episode_valid and infos:
            for env_idx in range(args.num_envs):
                if env_idx < len(infos):
                    is_success = infos[env_idx].get('success', False)
                    recent_100_env_results.append(is_success)
        
        # [显存优化] Episode结束后清理显存，释放tensor引用
        states = None  # type: ignore[assignment]
        rewards = None  # type: ignore[assignment]
        dones = None  # type: ignore[assignment]
        infos = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
        gc.collect()  # [内存优化] 强制Python垃圾回收
                
        # 记录 episode 奖励
        episode_rewards_history.append(episode_reward)
        
        # [CSV记录] 只记录有效episode（排除IK失败和崩飞）
        if episode_valid:
            # 累加有效步数
            global_step += step + 1  # step是0-indexed，所以+1
            # 计算本episode的平均loss
            episode_avg_loss = np.mean(train_loss_buffer[-step-1:]) if train_loss_buffer else 0.0
            csv_writer.writerow([episode + 1, global_step, episode_avg_loss, episode_reward])
            csv_file.flush()
        
        # Episode 总结
        print_training_log('episode',
            episode=episode+1,
            total_steps=step+1,
            max_steps=args.episode_max_steps,
            total_reward=episode_reward,
            buffer_size=len(agent.replay_buffer),
            buffer_capacity=args.replay_buffer_size,
            num_envs=args.num_envs,
            env_rewards=env_rewards,  # [新增] 传递每个环境的奖励
            action_counts=action_counts,  # [新增] 传递动作统计
            action_explore_counts=action_explore_counts,  # [新增] 探索次数
            action_exploit_counts=action_exploit_counts,  # [新增] 利用次数
            success_count=success_count,  # 成功次数
            valid_task_count=valid_task_count,  # [新增] 有效任务数（排除IK失败）
            ik_failed_count=ik_failed_count,  # [新增] IK失败次数
            recent_100_env_results=recent_100_env_results  # [修改] 最近100次环境结果
        )
        
        # 每50个episode打印统计并保存曲线
        if (episode + 1) % args.save_every == 0:
            avg_reward_10 = np.mean(episode_rewards_history[-10:])
            print_training_log('progress',
                episode=episode+1,
                total_episodes=args.n_episodes,
                avg_reward_10=avg_reward_10,
                global_step=global_step
            )
            
            # [修改] 定期刷新CSV缓冲区（替代之前的绘图保存）
            if args.save_intermediate:
                csv_file.flush()  # 确保数据写入磁盘
                print(f"✓ 训练数据已刷新到: {csv_path}")
        
        # 保存中间模型（如果开启）
        if args.save_intermediate and (episode + 1) % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f"model_episode_{episode+1}.pth")
            agent.save(save_path)
            print(f"\n✓ 中间模型已保存: {save_path}\n")
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "model_final.pth")
    agent.save(final_path)
    
    # ============================================================
    # 关闭CSV文件
    # ============================================================
    csv_file.close()
    print(f"\n✓ 训练数据已保存到: {csv_path}")
    
    print("\n" + "=" * 80)
    print(f"训练完成！最终模型已保存到: {final_path}")
    print("=" * 80)
    
    # [Isaac Sim Fix] 使用 os._exit(0) 强制结束进程，
    # 避免 simulation_app.close() 在清理 Replicator/SyntheticData 图节点时出现无限报错。
    os._exit(0)


if __name__ == "__main__":
    main()