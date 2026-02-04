"""
eval_exp.py - 模型评估实验脚本

加载训练好的检查点，每一轮在随机生成的场景中进行量化评估，
输出详细的 CSV 日志和统计结果。
"""

import os
import sys
import argparse
import csv
import random
from datetime import datetime

import torch
import numpy as np

# 添加 src 和 train 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
train_path = os.path.abspath(os.path.join(current_dir, "../train"))
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)
sys.path.insert(0, current_dir)  # 同时也包含当前 eval 目录

from scene import Scene
from agent import DQNAgent
from env_wrapper import PushEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Trained Model (Experiment)')
    
    # 模型加载参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='模型检查点路径 (必需)')
    parser.add_argument('--use_equivariant', action='store_true', default=True,
                        help='使用等变网络 (默认True)')
    parser.add_argument('--no_equivariant', dest='use_equivariant', action='store_false',
                        help='使用非等变CNN网络')
    
    # 环境参数
    parser.add_argument('--num_envs', default=1, type=int, 
                        help='并行环境数量 (评估时建议为1)')
    parser.add_argument('--num_objects_min', default=9, type=int, 
                        help='最小物体数')
    parser.add_argument('--num_objects_max', default=9, type=int, 
                        help='最大物体数')
    parser.add_argument('--episode_max_steps', default=8, type=int, 
                        help='每个 episode 最大步数')
    # 可视化选项：默认 headless，使用 --visualize 或 --no-headless 启用可视化
    parser.add_argument('--headless', action='store_true', default=False,
                        help='无界面模式')
    parser.add_argument('--visualize', '--no-headless', dest='visualize', action='store_true', default=False,
                        help='启用可视化界面 (等同于 --no-headless)')
    
    # 评估参数
    parser.add_argument('--eval_episodes', default=100, type=int, 
                        help='评估总轮次')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='设备: cuda 或 cpu')
    parser.add_argument('--seed', default=None, type=int, 
                        help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                        default='/home/k/Projects/equi/IsaacLab/scripts/workspace/train/results/log',
                        help='CSV输出目录')
    
    # [新增] 场景随机化和力监测参数
    parser.add_argument('--force_scene_regenerate', action='store_true', default=True,
                        help='每轮评估强制重新生成场景 (默认True)')
    parser.add_argument('--no_scene_regenerate', dest='force_scene_regenerate', action='store_false',
                        help='禁用场景随机化，保持与训练一致的场景')
    parser.add_argument('--enable_contact_sensors', action='store_true', default=False,
                        help='启用接触传感器进行崩飞力监测 (默认False, 启用会增加仿真开销)')
    
    return parser.parse_args()


def set_global_seed(seed):
    """设置全局随机种子以确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_failure_reason(info, step, max_steps):
    """
    根据 info 字典判断失败原因
    
    Returns:
        str: None (成功), 'OOB', 'Timeout', 'IK_Failed', 'Exploded'
    """
    if info.get('success', False):
        return None  # 成功
    
    if info.get('ik_failed', False):
        return 'IK_Failed'
    
    if info.get('is_exploded', False):
        return 'Exploded'
    
    if info.get('out_of_bounds', False):
        return 'OOB'
    
    if step >= max_steps - 1 or info.get('max_steps_exceeded', False):
        return 'Timeout'
    
    return 'Unknown'


def main():
    args = parse_args()
    
    # 1. 设置随机种子（默认使用时间戳，每次运行不同）
    if args.seed is None:
        import time
        args.seed = int(time.time()) % 100000  # 使用时间戳作为种子
    set_global_seed(args.seed)
    
    # 2. 确保启用相机
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    
    # 3. 处理 headless/visualize 参数
    # 逻辑：如果用户指定 --visualize，则启用可视化；否则默认 headless
    use_headless = not args.visualize  # visualize=True 时不使用 headless
    if args.headless:
        use_headless = True  # 显式指定 --headless 优先
    
    if use_headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")
    
    # 4. 打印配置信息
    print("=" * 80)
    print("模型评估实验 (eval_exp.py)")
    print("=" * 80)
    print(f"检查点路径: {args.checkpoint}")
    print(f"网络类型: {'C4等变网络' if args.use_equivariant else 'CNN网络'}")
    print(f"评估轮次: {args.eval_episodes}")
    print(f"物体数量: {args.num_objects_min}-{args.num_objects_max}")
    print(f"最大步数/轮: {args.episode_max_steps}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print(f"可视化模式: {'开启' if not use_headless else '关闭 (headless)'}")
    print(f"场景随机化: {'开启 (每轮重新生成)' if args.force_scene_regenerate else '关闭 (与训练一致)'}")
    print(f"崩飞力监测: {'开启' if args.enable_contact_sensors else '关闭'}")
    print("=" * 80)
    
    # 5. 验证检查点存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    # 6. 创建场景
    print("\n[1/4] 初始化场景...")
    scene = Scene(description="Model Evaluation", num_envs=args.num_envs)
    
    # 7. 创建环境包装器 (使用与 train.py 相同的配置)
    print("[2/4] 创建环境...")
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps
    
    # 8. 创建 DQN Agent
    print("[3/4] 创建 Agent...")
    agent = DQNAgent(
        device=args.device,
        lr=1e-4,  # 评估时不使用，但需要初始化
        gamma=0.99,
        buffer_capacity=100,  # 最小容量，不实际使用
        use_equivariant=args.use_equivariant
    )
    
    # 9. 加载模型
    print(f"[4/4] 加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 检测 checkpoint 格式
    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        print("[✓] 模型加载成功 (checkpoint 格式)")
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("[✓] 模型加载成功 (权重格式)")
    
    # 设置为评估模式
    agent.policy_net.eval()
    agent.target_net.eval()
    
    # 10. 准备 CSV 输出
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    csv_filename = f"eval_{checkpoint_name}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # 写入元数据头部
    csv_writer.writerow(['# Evaluation Metadata'])
    csv_writer.writerow(['Random_Seed', args.seed])
    csv_writer.writerow(['Checkpoint', args.checkpoint])
    csv_writer.writerow(['Network_Type', 'C4_Equivariant' if args.use_equivariant else 'CNN'])
    csv_writer.writerow(['Eval_Episodes', args.eval_episodes])
    csv_writer.writerow(['Num_Objects', f'{args.num_objects_min}-{args.num_objects_max}'])
    csv_writer.writerow(['Max_Steps_Per_Episode', args.episode_max_steps])
    csv_writer.writerow(['Timestamp', timestamp])
    csv_writer.writerow([])  # 空行分隔
    
    # 写入评估结果表头
    csv_writer.writerow(['# Evaluation Results'])
    csv_writer.writerow(['Episode_ID', 'Success', 'Steps', 'Total_Reward', 'Failure_Reason', 'Num_Objects'])
    
    print(f"\n[数据记录] 评估日志将保存到: {csv_path}")
    
    # 11. 评估循环
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    # 统计变量
    success_count = 0
    total_steps_success = 0  # 成功case的总步数
    total_reward = 0.0
    failure_reasons = {'OOB': 0, 'Timeout': 0, 'IK_Failed': 0, 'Exploded': 0, 'Unknown': 0}
    reset_failures = 0  # 重置失败计数
    consecutive_reset_failures = 0  # 连续重置失败计数
    
    for episode in range(args.eval_episodes):
        # [显存优化] 每个 episode 开始前清理 GPU 缓存
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 重置环境
        try:
            states, spawned_objects = env.reset()
            consecutive_reset_failures = 0  # 重置成功，清零连续失败计数
        except Exception as e:
            print(f"[Episode {episode+1}] 环境重置失败: {e}")
            csv_writer.writerow([episode + 1, False, 0, 0.0, 'Reset_Failed', args.num_objects_min])
            reset_failures += 1
            consecutive_reset_failures += 1
            
            # [关键] 强制清理显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 如果连续失败超过5次，很可能是不可恢复的显存泄漏
            if consecutive_reset_failures >= 5:
                print(f"\n⚠️ 连续 {consecutive_reset_failures} 次重置失败，可能存在显存泄漏！")
                print("建议：重启仿真环境或减少评估轮次")
                
                # 尝试继续，但记录警告
                if consecutive_reset_failures >= 10:
                    print("❌ 连续失败超过10次，终止评估")
                    break
            
            continue
        
        episode_reward = 0.0
        episode_success = False
        failure_reason = None
        final_step = 0
        
        # Episode 内循环
        with torch.no_grad():
            for step in range(args.episode_max_steps):
                final_step = step + 1
                
                # 选择动作 (Greedy: epsilon=0)
                state = states[0:1]  # (1, 3, 320, 320)
                
                try:
                    action, _ = agent.select_action(
                        state, 
                        epsilon=0.0,  # 完全贪婪策略
                        invalid_actions=[],
                        env_idx=0
                    )
                except Exception as e:
                    print(f"[Episode {episode+1}, Step {step+1}] 动作选择失败: {e}")
                    failure_reason = 'Action_Failed'
                    break
                
                # 执行动作
                try:
                    next_states, rewards, dones, infos = env.step([action], spawned_objects)
                except Exception as e:
                    print(f"[Episode {episode+1}, Step {step+1}] 环境步进失败: {e}")
                    failure_reason = 'Step_Failed'
                    break
                
                # 累计奖励
                episode_reward += rewards[0].item()
                
                # 检查结果
                info = infos[0]
                
                if info.get('success', False):
                    episode_success = True
                    failure_reason = None
                    break
                
                if dones[0]:
                    # Episode 结束但未成功
                    failure_reason = get_failure_reason(info, step, args.episode_max_steps)
                    break
                
                states = next_states
            
            # 如果循环结束但未触发 done，检查最终状态
            if not episode_success and failure_reason is None:
                failure_reason = get_failure_reason(infos[0], final_step - 1, args.episode_max_steps)
        
        # 更新统计
        if episode_success:
            success_count += 1
            total_steps_success += final_step
        elif failure_reason:
            if failure_reason in failure_reasons:
                failure_reasons[failure_reason] += 1
            else:
                failure_reasons['Unknown'] += 1
        
        total_reward += episode_reward
        
        # 写入 CSV
        csv_writer.writerow([
            episode + 1,
            episode_success,
            final_step,
            round(episode_reward, 4),
            failure_reason,
            args.num_objects_min
        ])
        csv_file.flush()
        
        # 打印实时日志
        result_str = "Success" if episode_success else f"Failed ({failure_reason})"
        print(f"[Eval] Episode {episode+1}/{args.eval_episodes} | Result: {result_str} | Steps: {final_step} | Reward: {episode_reward:.2f}")
    
    # 12. 追加崩飞力监测日志到同一CSV文件
    if args.enable_contact_sensors and env.explosion_logs:
        csv_writer.writerow([])  # 空行分隔
        csv_writer.writerow(['# Explosion Monitoring Logs'])
        csv_writer.writerow(['Timestamp', 'Env_Idx', 'Object_Name', 'Exceed_Distance', 'Axis', 'Net_Force_Magnitude', 'Error'])
        
        for log in env.explosion_logs:
            csv_writer.writerow([
                log.get('timestamp', ''),
                log.get('env_idx', ''),
                log.get('obj_name', ''),
                log.get('exceed_distance', ''),
                log.get('axis', ''),
                log.get('net_force_magnitude', ''),
                log.get('error', '')
            ])
    # 计算统计数据
    actual_episodes = episode + 1
    valid_episodes = actual_episodes - reset_failures
    success_rate = 100.0 * success_count / valid_episodes if valid_episodes > 0 else 0
    avg_reward = total_reward / valid_episodes if valid_episodes > 0 else 0
    avg_steps_success = total_steps_success / success_count if success_count > 0 else 0
    explosion_count = len(env.explosion_logs) if hasattr(env, 'explosion_logs') else 0
    
    # 写入统计摘要到CSV
    csv_writer.writerow([])  # 空行分隔
    csv_writer.writerow(['# Evaluation Summary'])
    csv_writer.writerow(['Valid_Episodes', valid_episodes])
    csv_writer.writerow(['Success_Count', success_count])
    csv_writer.writerow(['Success_Rate', f'{success_rate:.2f}%'])
    csv_writer.writerow(['Avg_Reward', f'{avg_reward:.4f}'])
    csv_writer.writerow(['Avg_Steps_Success', f'{avg_steps_success:.2f}'])
    csv_writer.writerow(['Explosion_Count', explosion_count])
    
    # 关闭 CSV
    csv_file.close()
    
    # 13. 最终统计（使用之前已计算的变量）
    print("\n" + "=" * 80)
    print("评估完成")
    print("=" * 80)
    
    print(f"\n【统计结果】")
    print(f"  计划评估轮数: {args.eval_episodes}")
    print(f"  实际完成轮数: {actual_episodes}")
    print(f"  有效评估轮数: {valid_episodes} (排除 {reset_failures} 次重置失败)")
    print(f"  成功率: {success_rate:.2f}% ({success_count}/{valid_episodes})")
    print(f"  平均奖励: {avg_reward:.4f}")
    print(f"  成功case平均步数: {avg_steps_success:.2f}")
    print(f"  崩飞次数: {explosion_count}")
    
    print(f"\n【失败原因分布】")
    for reason, count in failure_reasons.items():
        if count > 0:
            pct = 100.0 * count / valid_episodes if valid_episodes > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
    
    if reset_failures > 0:
        print(f"\n【警告】")
        print(f"  重置失败次数: {reset_failures}")
        print(f"  这可能表明存在显存泄漏，建议减少评估轮次或重启仿真环境")
    
    print(f"\n【输出文件】")
    print(f"  评估日志: {csv_path}")
    if args.enable_contact_sensors and env.explosion_logs:
        print(f"         + 崩飞力监测日志 {len(env.explosion_logs)} 条记录)")
    else:
        print(f")")
    
    print("=" * 80)
    
    # 14. 关闭仿真
    scene.simulation_app.close()


if __name__ == "__main__":
    main()
