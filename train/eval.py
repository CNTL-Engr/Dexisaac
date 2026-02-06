"""
训练模型评估脚本
加载训练好的 C4PushNet 模型并在环境中测试性能
"""

import os
import sys
import argparse
import torch
import numpy as np

# 添加 src 和 train 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
train_path = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)

from scene import Scene
from agent import DQNAgent
from env_wrapper import PushEnv
from utils import print_training_log


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Trained EquivariantPushNet')
    
    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--num_objects_min', default=2, type=int, help='最小物体数')
    parser.add_argument('--num_objects_max', default=2, type=int, help='最大物体数')
    parser.add_argument('--episode_max_steps', default=100, type=int, help='每个 episode 最大步数')
    parser.add_argument('--headless', action='store_true', help='是否不可视化 (无界面模式)')
    
    # 评估参数
    parser.add_argument('--n_episodes', default=50, type=int, help='评估轮数')
    
    # 模型加载参数（必需）
    parser.add_argument('--model_path', default="/home/wyq/xc/equi/IsaacLab/scripts/workspace/train/checkpoints2/model_final.pth", type=str, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确保启用相机
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    
    # 同步 headless 参数
    if args.headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")
    
    print("=" * 80)
    print("C4PushNet 模型评估")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"环境数量: {args.num_envs}")
    print(f"评估轮数: {args.n_episodes}")
    print(f"设备: {args.device}")
    print(f"无界面模式: {args.headless}")
    print("=" * 80)
    
    # 1. 创建场景
    print("\n[1/4] 初始化场景...")
    scene = Scene(description="Model Evaluation", num_envs=args.num_envs)
    
    # 2. 创建环境包装器
    print("[2/4] 创建环境...")
    env = PushEnv(scene=scene, device=args.device)
    env.max_steps_per_episode = args.episode_max_steps
    
    # 3. 创建 DQN Agent
    print("[3/4] 创建 Agent...")
    agent = DQNAgent(
        device=args.device,
        lr=1e-4,  # 评估时不使用，但需要初始化
        gamma=0.99,
        buffer_capacity=100  # 最小容量，不实际使用
    )
    
    # 4. 加载模型（必需）
    print(f"[4/4] 加载模型: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # 检测 checkpoint 格式
    if 'policy_net' in checkpoint:
        # 完整 checkpoint 格式
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        print("[✓] 模型加载成功 (checkpoint 格式)")
    else:
        # 纯网络权重格式
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("[✓] 模型加载成功 (权重格式)")
    
    # 设置为评估模式
    agent.policy_net.eval()
    agent.target_net.eval()
    
    # 5. 评估循环
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    epsilon = 0.0  # 纯利用，不探索
    
    # 统计数据
    episode_rewards_history = []
    episode_steps_history = []
    success_count = 0
    out_of_bounds_count = 0
    
    for episode in range(args.n_episodes):
        print("\n" + "=" * 80)
        print(f"  Episode {episode+1}/{args.n_episodes}")
        print("=" * 80)
        
        # 重置环境
        states, spawned_objects = env.reset()
        print(f"  [环境状态] 生成成功")
        
        episode_reward = 0
        env_rewards = [0.0] * args.num_envs
        env_success = [False] * args.num_envs
        env_out_of_bounds = [False] * args.num_envs
        
        for step in range(args.episode_max_steps):
            # 选择动作（纯利用，epsilon=0）
            actions = []
            for env_idx in range(args.num_envs):
                state = states[env_idx:env_idx+1]
                # 不需要可视化和IAS
                action = agent.select_action(
                    state, 
                    epsilon=epsilon,  # 纯利用
                    invalid_actions=[],  # 评估时不使用IAS
                    step=0, 
                    visualize=False, 
                    env_idx=env_idx
                )
                actions.append(action)
            
            # 执行动作
            next_states, rewards, dones, infos = env.step(actions, spawned_objects)
            
            # 记录统计信息
            for env_idx in range(args.num_envs):
                if not dones[env_idx] or step == 0:  # 只统计未完成环境
                    env_rewards[env_idx] += rewards[env_idx].item()
                    if infos[env_idx].get('success', False):
                        env_success[env_idx] = True
                    if infos[env_idx].get('out_of_bounds', False):
                        env_out_of_bounds[env_idx] = True
            
            # 打印每步信息（简化版）
            print_training_log('step',
                step=step+1,
                max_steps=args.episode_max_steps,
                infos=infos,
                step_loss=None,  # 评估时无loss
                epsilon=epsilon,
                agent=agent,
                min_buffer_size=0,
                rewards=rewards,
                actions=actions,
                invalid_actions_list=[[] for _ in range(args.num_envs)],
                dones=dones
            )
            
            states = next_states
            episode_reward += rewards.sum().item()
            
            if dones.all():
                print(f"\n  >> 所有环境已完成")
                break
        
        # Episode 统计
        episode_rewards_history.append(episode_reward)
        episode_steps_history.append(step + 1)
        
        # 统计成功和出界
        for env_idx in range(args.num_envs):
            if env_success[env_idx]:
                success_count += 1
            if env_out_of_bounds[env_idx]:
                out_of_bounds_count += 1
        
        # 打印 Episode 总结
        print_training_log('episode',
            episode=episode+1,
            total_steps=step+1,
            max_steps=args.episode_max_steps,
            total_reward=episode_reward,
            buffer_size=0,  # 评估时不使用buffer
            buffer_capacity=0,
            num_envs=args.num_envs,
            env_rewards=env_rewards
        )
    
    # 最终统计
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    print(f"评估轮数: {args.n_episodes}")
    print(f"总环境数: {args.n_episodes * args.num_envs}")
    print(f"平均奖励: {np.mean(episode_rewards_history):.2f} ± {np.std(episode_rewards_history):.2f}")
    print(f"平均步数: {np.mean(episode_steps_history):.2f} ± {np.std(episode_steps_history):.2f}")
    print(f"成功次数: {success_count} / {args.n_episodes * args.num_envs} ({100*success_count/(args.n_episodes * args.num_envs):.1f}%)")
    print(f"出界次数: {out_of_bounds_count} / {args.n_episodes * args.num_envs} ({100*out_of_bounds_count/(args.n_episodes * args.num_envs):.1f}%)")
    print("=" * 80)
    
    # 关闭仿真
    scene.simulation_app.close()


if __name__ == "__main__":
    main()