"""
C6PushNet 强化学习训练脚本
使用 DQN 算法训练 EquivariantPushNet (C6等变) 策略网络
"""

import os
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
from agent import DQNAgent
from env_wrapper import PushEnv
from utils import print_training_log, compute_epsilon, generate_checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train EquivariantPushNet with DQN')
    
    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--num_objects_min', default=8, type=int, help='最小物体数')
    parser.add_argument('--num_objects_max', default=8, type=int, help='最大物体数')
    parser.add_argument('--episode_max_steps', default=8, type=int, help='每个 episode 最大步数')
    parser.add_argument('--headless', action='store_true', default=True, help='无界面模式 (默认开启)')
    parser.add_argument('--no-headless', dest='headless',default=False, action='store_false', help='启用可视化界面')
    
    # 训练参数
    parser.add_argument('--n_episodes', default=500, type=int, help='总训练轮数')
    parser.add_argument('--batch_size', default=16, type=int, help='训练批大小（降至16以节省显存）')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
    parser.add_argument('--gamma', default=0.99, type=float, help='折扣因子')
    parser.add_argument('--epsilon_start', default=0.7, type=float, help='初始探索率')
    parser.add_argument('--epsilon_end', default=0.05, type=float, help='最终探索率')
    parser.add_argument('--epsilon_decay_steps', default=2500, type=int, help='探索衰减步数')
    parser.add_argument('--target_update_freq', default=10, type=int, help='目标网络更新频率(步数)')
    parser.add_argument('--replay_buffer_size', default=10000, type=int, help='经验池大小')
    parser.add_argument('--min_buffer_size', default=16, type=int, help='开始训练的最小经验数')
    
    # 保存参数
    parser.add_argument('--save_every', default=50, type=int, help='保存频率(episodes)')
    parser.add_argument('--checkpoint_base_dir', default='/home/wyq/xc/equi/IsaacLab/scripts/workspace/train/results', type=str, help='检查点根目录（将自动生成子目录名）')
    parser.add_argument('--save_intermediate', action='store_true', default=True, help='是否保存中间模型（默认关闭，只保存最终模型）')
    
    # 模型加载参数
    parser.add_argument('--load_model', action='store_true', default=True, help='是否加载预训练模型')
    parser.add_argument('--model_path', default='/home/wyq/xc/equi/IsaacLab/scripts/workspace/train/results/equi_obj_7/model_final.pth', type=str, help='预训练模型路径')
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
    
    # 3. 创建 DQN Agent (根据参数选择等变或非等变网络)
    print("[3/4] 创建 DQN Agent...")
    agent = DQNAgent(
        device=args.device,
        lr=args.learning_rate,
        gamma=args.gamma,
        buffer_capacity=args.replay_buffer_size,
        use_equivariant=args.use_equivariant
    )
    
    # [模型加载] 如果指定加载预训练模型
    if args.load_model:
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
    
    global_step = 0
    episode_rewards_history = [] # Renamed to avoid conflict with episode_reward in loop
    train_loss_buffer = [] # Buffer for storing training losses
    
    # [新增] 动作选择统计 (8个动作: 0-7)
    action_counts = [0] * 8  # 总次数
    action_explore_counts = [0] * 8  # 探索次数
    action_exploit_counts = [0] * 8  # 利用次数
    success_count = 0  # 记录总成功次数
    ik_failed_count = 0  # [新增] 记录IK失败次数（不计入成功率）
    valid_task_count = 0  # [新增] 记录有效任务次数（排除IK失败）
    
    # [修改] 记录最近100次环境任务的成功/失败（每个环境算一次任务）
    from collections import deque
    import csv
    recent_100_env_results = deque(maxlen=100)  # 每个元素是True/False，表示单个环境的成功/失败
    
    # [新增] CSV 数据记录
    csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    os.makedirs(args.checkpoint_dir, exist_ok=True)  # 确保目录存在
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'step', 'loss', 'reward'])  # 写入表头
    print(f"[数据记录] 训练日志将保存到: {csv_path}")
    
    # 为每个环境维护一个无效动作列表 (Invalid Action Suppression)
    invalid_actions_list = [[] for _ in range(args.num_envs)]

    for episode in range(args.n_episodes):
        # [新增] Episode重试循环（IK失败时重新开始）
        episode_retry_count = 0
        max_episode_retries = 5  # 最大重试次数
        episode_valid = False
        
        while not episode_valid and episode_retry_count < max_episode_retries:
            episode_retry_count += 1
            ik_failed_this_episode = False  # 标记本episode是否发生IK失败
            episode_experiences = []  # 临时存储本episode的经验
            
            # Episode 开始标题
            if episode_retry_count == 1:
                print("\n" + "=" * 80)
                print(f"  Episode {episode+1}/{args.n_episodes}")
                print("=" * 80)
            else:
                print(f"\n  [重试 {episode_retry_count}/{max_episode_retries}] Episode {episode+1}")
            
            # 重置环境
            states, spawned_objects = env.reset()
            print(f"  [环境状态] 生成成功")
            
            # [显存优化] Reset后立即清理GPU缓存
            torch.cuda.empty_cache()
            
            episode_reward = 0
            # [新增] 每个环境的奖励跟踪
            env_rewards = [0.0] * args.num_envs  # 每个环境的累计奖励
            
            for step in range(args.episode_max_steps):
                # [修复] 先使用当前的global_step，最后再递增
                # 这样可以确保在global_step=0,10,20,30...时正确触发更新

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

                # 计算当前epsilon
                epsilon = compute_epsilon(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
                
                # 选择动作（单流网络：只返回 u, v, direction）
                actions = []
                strategy_types = []  # 记录每个环境的策略类型
                
                # [调试] 每10个episode打印状态和Q值
                debug_print = (episode % 10 == 0) and (step == 0)
                
                for env_idx in range(args.num_envs):
                    state = states[env_idx:env_idx+1]  # (1, 3, 320, 320)
                    
                    action, strategy_type = agent.select_action(state, epsilon, invalid_actions=invalid_actions_list[env_idx], env_idx=env_idx, debug=debug_print)
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

                
                # 训练
                step_loss = None
                buffer_size = len(agent.replay_buffer)
                if buffer_size >= 4:
                    dynamic_batch_size = min(buffer_size, args.batch_size)
                    step_loss = agent.train_step(batch_size=dynamic_batch_size)
                    if step_loss is not None:
                        train_loss_buffer.append(step_loss)
                    torch.cuda.empty_cache()
                
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
                    agent.store_transition(
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        next_state=exp['next_state'],
                        done=exp['done']
                    )
                episode_experiences.clear()
        
        # 如果重试次数用尽仍然失败
        if not episode_valid:
            print(f"\n  ⚠ Episode {episode+1} 重试 {max_episode_retries} 次后仍失败，跳过此回合")
        
        # [修改] 记录每个环境的成功/失败（只记录有效episode）
        if episode_valid and infos:
            for env_idx in range(args.num_envs):
                if env_idx < len(infos):
                    is_success = infos[env_idx].get('success', False)
                    recent_100_env_results.append(is_success)
        
        # [显存优化] Episode结束后清理显存
        if 'states' in dir():
            del states
        if 'rewards' in dir():
            del rewards  
        if 'dones' in dir():
            del dones
        if 'infos' in dir():
            del infos
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
    
    # 关闭仿真
    scene.simulation_app.close()


if __name__ == "__main__":
    main()