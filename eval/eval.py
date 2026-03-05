"""
PushNet 模型评估脚本
加载训练好的模型进行仿真评估，统计成功率、崩飞次数等指标，并输出 CSV 日志

使用示例:
    python eval.py --model_path /path/to/model.pth --n_episodes 100 --seed 42
"""

import os
import sys
import argparse
import csv
import gc
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加 src 和 train 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.abspath(os.path.join(current_dir, "../train"))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_path)
sys.path.insert(0, train_path)

from scene import Scene
from agent import DQNAgent
from env_wrapper import PushEnv


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained PushNet model')

    # 模型参数
    parser.add_argument('--model_path', type=str,
                        default='/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/model_results/equi_obj_7/model_final.pth',
                        help='训练好的模型文件路径')
    parser.add_argument('--use_equivariant', action='store_true', default=True,
                        help='是否使用C4等变网络（默认开启）')

    # 评估参数
    parser.add_argument('--n_episodes', default=100, type=int, help='评估轮数')
    parser.add_argument('--seed', default=None, type=int,
                        help='随机种子（不指定则自动随机生成）')
    parser.add_argument('--episode_max_steps', default=8, type=int,
                        help='每个 episode 最大步数')

    # 环境参数
    parser.add_argument('--num_objects_min', default=9, type=int, help='最小物体数')
    parser.add_argument('--num_objects_max', default=9, type=int, help='最大物体数')
    parser.add_argument('--num_envs', default=1, type=int, help='并行环境数量')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='无界面模式 (默认开启)')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='启用可视化界面')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')

    # 日志参数
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志保存目录（默认保存到 eval/ 目录下）')

    return parser.parse_args()


# ============================================================
# 每步日志打印
# ============================================================
def print_step_log(step, max_steps, action_idx, invalid_actions, info,
                   is_exploded_step=False):
    """
    [功能]: 打印单步评估日志（动作、屏蔽、出界/成功/空推判定）
    """
    # 动作信息
    if action_idx <= 3:
        action_type = "推目标"
        direction_deg = action_idx * 90
    else:
        action_type = "推障碍"
        direction_deg = (action_idx - 4) * 90

    print(f"  动作选择: {action_type} (Index {action_idx}, 方向{direction_deg}°)")

    # 屏蔽动作
    if invalid_actions:
        ias_details = ", ".join([f"Act{a}" for a in invalid_actions])
        print(f"  [IAS] 屏蔽动作: {ias_details}")

    print("-" * 10)

    # 出界判定
    is_out = info.get('out_of_bounds', False)
    out_msg = (f"✗ 出界 (原因: {info.get('out_reason', 'none')})"
               if is_out else "✓ 未出界")
    print(f"  出界判定: {out_msg}")

    # 成功判定
    success = info.get('success', False)
    sep = info.get('separation_metrics', {})
    sim_val = sep.get('similarity', 0.0)
    thr_val = sep.get('threshold', 0.95)
    print(f"  成功判定: {'✓ 成功' if success else '× 未成功'} "
          f"(相似度: {sim_val:.2%}, 阈值: {thr_val})")

    # 空推判定
    is_empty = info.get('empty_push', False)
    emp = info.get('empty_metrics', {})
    change_val = int(emp.get('change_value', 0))
    total_px = emp.get('total_pixels', 1)
    change_ratio = emp.get('change_ratio', 0.0)
    print(f"  空推判定: {'⚠ 空推' if is_empty else '✓ 有效'} "
          f"(变化: {change_val}/{total_px} ({change_ratio:.2f}%))")

    # 崩飞判定
    if is_exploded_step:
        print(f"  💥 检测到物体崩飞!")

    print("-" * 8)


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    # ---- 随机种子 ----
    if args.seed is None:
        args.seed = np.random.randint(0, 100000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 记录运行开始时间（用于日志文件命名）
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")

    # ---- 打印评估配置 ----
    print("=" * 80)
    print("  PushNet 模型评估")
    print("=" * 80)
    print(f"  随机种子 (Seed): {args.seed}")
    print(f"  模型路径: {args.model_path}")
    print(f"  评估轮数: {args.n_episodes}")
    print(f"  每轮最大步数: {args.episode_max_steps}")
    print(f"  物体数量范围: {args.num_objects_min}-{args.num_objects_max}")
    print(f"  并行环境数: {args.num_envs}")
    print(f"  设备: {args.device}")
    net_type = "C4等变网络" if args.use_equivariant else "普通CNN网络"
    print(f"  网络类型: {net_type}")
    print("=" * 80)

    # ---- 检查模型文件 ----
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    # 确保 Isaac Sim 参数
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if args.headless and "--headless" not in sys.argv:
        sys.argv.append("--headless")

    # ============================================================
    # 初始化场景、环境、Agent
    # ============================================================
    print("\n[1/3] 初始化场景...")
    scene = Scene(description="Model Evaluation", num_envs=args.num_envs)

    print("[2/3] 创建环境...")
    env = PushEnv(scene=scene, args=args)
    env.max_steps_per_episode = args.episode_max_steps

    print("[3/3] 加载模型...")
    agent = DQNAgent(
        device=args.device,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=100,
        use_equivariant=args.use_equivariant
    )

    checkpoint = torch.load(args.model_path, map_location=args.device)
    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        if 'target_net' in checkpoint:
            agent.target_net.load_state_dict(checkpoint['target_net'])
        print("  ✓ 模型加载成功 (checkpoint 格式)")
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("  ✓ 模型加载成功 (权重格式)")

    agent.policy_net.eval()
    agent.target_net.eval()

    # ============================================================
    # 评估循环
    # ============================================================
    print("\n" + "=" * 80)
    print("  开始评估...")
    print("=" * 80)

    total_success = 0
    total_exploded = 0
    total_out_of_bounds = 0
    total_empty_pushes = 0
    total_steps_all = 0
    episode_results = []

    MAX_RETRIES = 5  # IK失败 / 动作前崩飞时的重试上限

    for episode in range(args.n_episodes):
        # ---- Episode 重试循环（仅对 IK 失败或动作前崩飞重试） ----
        retry_count = 0
        episode_valid = False

        # 默认初始化（确保循环后有值）
        episode_success = False
        episode_exploded = False
        episode_out_of_bounds = False
        episode_empty_push_count = 0
        episode_steps = 0
        episode_fail_reason = ""

        while not episode_valid and retry_count < MAX_RETRIES:
            retry_count += 1
            should_retry = False  # 是否需要重试

            # Episode 标题
            if retry_count == 1:
                print("\n" + "=" * 80)
                print(f"  Episode {episode + 1}/{args.n_episodes}")
                print("=" * 80)
            else:
                print(f"\n  [重试 {retry_count}/{MAX_RETRIES}] "
                      f"Episode {episode + 1}")

            # 重置环境（随机生成场景，随机布置所有物体的位置和旋转角度）
            states, spawned_objects = env.reset()
            torch.cuda.empty_cache()

            # 重置 episode 状态
            episode_success = False
            episode_exploded = False
            episode_out_of_bounds = False
            episode_empty_push_count = 0
            episode_steps = 0
            episode_fail_reason = ""
            invalid_actions = [[] for _ in range(args.num_envs)]

            for step in range(args.episode_max_steps):
                # ---- 动作前崩飞检测 ----
                pre_exploded = False
                for env_idx in range(args.num_envs):
                    is_out, out_reason, is_exploded = \
                        env._check_out_of_bounds(env_idx, spawned_objects)
                    if is_exploded:
                        print(f"\n  💥 [动作前检测] Env {env_idx} "
                              f"物体已崩飞: {out_reason}")
                        pre_exploded = True
                        break

                if pre_exploded:
                    # 动作前崩飞 → 本次重试
                    should_retry = True
                    break

                # ---- 选择动作 (epsilon=0, 完全利用) ----
                actions = []
                for env_idx in range(args.num_envs):
                    state = states[env_idx:env_idx + 1]
                    action, _ = agent.select_action(
                        state, epsilon=0.0,
                        invalid_actions=invalid_actions[env_idx],
                        env_idx=env_idx
                    )
                    actions.append(action)

                torch.cuda.empty_cache()

                # ---- 执行动作 ----
                try:
                    next_states, rewards, dones, infos = \
                        env.step(actions, spawned_objects)
                except Exception as e:
                    print(f"\n  ❌ env.step() 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_fail_reason = f"执行异常: {str(e)}"
                    break

                episode_steps += 1

                # ---- 检查 IK 失败 ----
                ik_failed = False
                for env_idx in range(args.num_envs):
                    if infos[env_idx].get('ik_failed', False):
                        ik_failed = True
                        break

                if ik_failed:
                    print(f"\n  ⚠ IK解算失败，本次Episode将终止并重试...")
                    should_retry = True
                    break

                # ---- 检查崩飞 (来自 _check_exploded_objects) ----
                step_exploded = False
                for env_idx in range(args.num_envs):
                    if infos[env_idx].get('is_exploded', False):
                        step_exploded = True
                        episode_exploded = True
                        episode_fail_reason = "物体崩飞"

                # ---- 打印每步日志 ----
                for env_idx in range(args.num_envs):
                    info = infos[env_idx]

                    print(f"\n  Step {step + 1}/{args.episode_max_steps}")
                    print_step_log(
                        step + 1, args.episode_max_steps,
                        actions[env_idx], invalid_actions[env_idx], info,
                        is_exploded_step=info.get('is_exploded', False)
                    )

                    # 更新无效动作列表 (IAS)
                    if dones[env_idx]:
                        invalid_actions[env_idx] = []
                    elif info.get('empty_push', False):
                        if actions[env_idx] not in invalid_actions[env_idx]:
                            invalid_actions[env_idx].append(actions[env_idx])
                        episode_empty_push_count += 1
                    else:
                        invalid_actions[env_idx] = []

                    # 记录结果标记
                    if info.get('success', False):
                        episode_success = True
                    if info.get('out_of_bounds', False):
                        episode_out_of_bounds = True
                        if not episode_fail_reason:
                            episode_fail_reason = \
                                f"出界: {info.get('out_reason', 'unknown')}"

                # 崩飞后终止
                if step_exploded:
                    break

                # 释放旧状态
                del states
                torch.cuda.empty_cache()
                states = next_states

                # 所有环境结束
                if dones.all():
                    break

            # ---- 内层循环结束：判断是否需要重试 ----
            if should_retry:
                torch.cuda.empty_cache()
                gc.collect()
                continue  # 重试 while 循环
            else:
                episode_valid = True

        # ---- 重试用尽仍失败 ----
        if not episode_valid:
            print(f"\n  ⚠ Episode {episode + 1} 重试 {MAX_RETRIES} "
                  f"次后仍失败（仿真器问题），标记为失败")
            episode_fail_reason = ("仿真器问题 (IK失败/崩飞) "
                                   f"重试{MAX_RETRIES}次后仍失败")

        # ============================================================
        # Episode 结果输出
        # ============================================================
        print("\n" + "-" * 50)
        if episode_success:
            print(f"  ✓ Episode {episode + 1} 成功 "
                  f"(步数: {episode_steps})")
        else:
            fail_msg = episode_fail_reason if episode_fail_reason \
                else "超过最大步数"
            print(f"  ✗ Episode {episode + 1} 失败 "
                  f"(原因: {fail_msg}, 步数: {episode_steps})")
        print("-" * 50)

        # 更新统计
        if episode_success:
            total_success += 1
        if episode_exploded:
            total_exploded += 1
        if episode_out_of_bounds:
            total_out_of_bounds += 1
        total_empty_pushes += episode_empty_push_count
        total_steps_all += episode_steps

        episode_results.append({
            'episode': episode + 1,
            'success': episode_success,
            'exploded': episode_exploded,
            'out_of_bounds': episode_out_of_bounds,
            'empty_pushes': episode_empty_push_count,
            'steps': episode_steps,
            'fail_reason': episode_fail_reason if not episode_success else ""
        })

        # 显存清理
        states = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # 最终统计
    # ============================================================
    success_rate = (100.0 * total_success / args.n_episodes
                    if args.n_episodes > 0 else 0.0)
    avg_steps = (total_steps_all / args.n_episodes
                 if args.n_episodes > 0 else 0.0)

    print("\n" + "█" * 80)
    print("  评估完成 - 最终统计")
    print("█" * 80)
    print(f"  随机种子 (Seed): {args.seed}")
    print(f"  模型: {args.model_path}")
    print(f"  评估轮数: {args.n_episodes}")
    print(f"  ----------------------------------------")
    print(f"  成功率: {success_rate:.2f}% "
          f"({total_success}/{args.n_episodes})")
    print(f"  崩飞次数: {total_exploded}")
    print(f"  出界次数: {total_out_of_bounds}")
    print(f"  总空推次数: {total_empty_pushes}")
    print(f"  平均步数: {avg_steps:.2f}")
    print("█" * 80)

    # ============================================================
    # 保存 CSV 日志
    # 命名规则: <模型文件名>_seed<种子>_<开始时间>.csv
    # ============================================================
    model_filename = Path(args.model_path).stem
    log_filename = f"{model_filename}_seed{args.seed}_{start_time_str}.csv"

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = current_dir  # 默认保存到 eval/ 目录

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ---- 元信息 ----
        writer.writerow(['# 评估配置'])
        writer.writerow(['model_path', args.model_path])
        writer.writerow(['seed', args.seed])
        writer.writerow(['n_episodes', args.n_episodes])
        writer.writerow(['episode_max_steps', args.episode_max_steps])
        writer.writerow(['num_objects_range',
                         f'{args.num_objects_min}-{args.num_objects_max}'])
        writer.writerow(['start_time', start_time_str])
        writer.writerow(['success_rate', f'{success_rate:.2f}%'])
        writer.writerow(['total_exploded', total_exploded])
        writer.writerow(['total_out_of_bounds', total_out_of_bounds])
        writer.writerow(['total_empty_pushes', total_empty_pushes])
        writer.writerow(['avg_steps', f'{avg_steps:.2f}'])
        writer.writerow([])

        # ---- 每个 Episode 详情 ----
        writer.writerow(['episode', 'success', 'exploded',
                         'out_of_bounds', 'empty_pushes',
                         'steps', 'fail_reason'])
        for result in episode_results:
            writer.writerow([
                result['episode'],
                result['success'],
                result['exploded'],
                result['out_of_bounds'],
                result['empty_pushes'],
                result['steps'],
                result['fail_reason']
            ])

    print(f"\n✓ 评估日志已保存到: {log_path}")

    # 关闭仿真
    scene.simulation_app.close()


if __name__ == "__main__":
    main()

# # 基本用法
# python eval/eval.py --model_path /path/to/model.pth --n_episodes 100 --seed 42

# # 自定义参数
# python eval.py --model_path /home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/model_results/equi_obj_9/model_1000_9s.pth --n_episodes 50 --seed 123 --episode_max_steps 10 --num_objects_min 7 --num_objects_max 9