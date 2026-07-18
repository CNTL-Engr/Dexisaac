"""
训练工具函数模块

包含日志打印、epsilon计算、路径生成等辅助功能
"""
import os


def format_contact_action(info):
    """按统一格式展示一次动作的意图模型与 PhysX 首次实际接触模型。"""
    intended = info.get('contact_intended_model_id') or '未知'
    status = info.get('contact_status', 'unavailable')
    actual = info.get('contact_actual_model_id')

    if status == 'illegal':
        return f'准备推"{intended}"  实际碰撞"{actual or "未知"}"'

    if status in ('no_contact', 'unavailable'):
        actual = '未接触'
    else:
        actual = actual or '未知'
    return f'准备推"{intended}"  实际推"{actual}"'


def format_push_effectiveness(info):
    """展示“接触力 AND (质心位移 OR 旋转边缘位移)”空推判定。"""
    is_empty = info.get('empty_push', False)
    metrics = info.get('empty_metrics') or {}
    if not metrics:
        return f"推动判定: {'⚠ 空推' if is_empty else '跳过'} (当前优先级分支未执行空推检测)"

    displacement = float(metrics.get('displacement_m', 0.0))
    displacement_threshold = float(metrics.get('displacement_threshold_m', 0.01))
    rotation_arc = float(metrics.get('rotation_arc_m', 0.0))
    rotation_arc_threshold = float(metrics.get('rotation_arc_threshold_m', 0.01))
    rotation_theta = metrics.get('rotation_theta_rad')
    rotation_radius = metrics.get('rotation_radius_m')
    peak_force = float(metrics.get('peak_contact_force_n', 0.0))
    force_threshold = float(metrics.get('force_threshold_n', 1.0))
    displacement_mark = '✓' if metrics.get('displacement_ok', False) else '×'
    rotation_mark = '✓' if metrics.get('rotation_ok', False) else '×'
    force_mark = '✓' if metrics.get('force_ok', False) else '×'
    theta_text = f'{float(rotation_theta):.4f}rad' if rotation_theta is not None else '不可用'
    radius_text = f'{float(rotation_radius):.4f}m' if rotation_radius is not None else '不可用'
    reason = metrics.get('reason', 'unknown')
    return (
        f"推动判定: {'⚠ 空推' if is_empty else '✓ 有效'} "
        f"(峰值力={peak_force:.3f}N/{force_threshold:.3f}N {force_mark} AND "
        f"[质心XY位移={displacement:.4f}m/{displacement_threshold:.4f}m {displacement_mark} "
        f"OR 旋转S={rotation_arc:.4f}m/{rotation_arc_threshold:.4f}m {rotation_mark} "
        f"(theta={theta_text}, D={radius_text})], 原因={reason})"
    )


def compute_epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps):
    """
    [功能]: 计算当前的epsilon值（线性衰减）
    [输入]: 
        step: 当前训练步数
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay_steps: 衰减步数
    [输出]: float, 当前epsilon值
    """
    decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
    epsilon = epsilon_start - decay_rate * step
    return max(epsilon_end, epsilon)


def generate_checkpoint_dir(base_dir, use_equivariant, num_objects_min, num_objects_max, num_targets=1):
    """
    [功能]: 根据训练参数自动生成检查点目录名
    [输入]:
        base_dir: 基础目录路径
        use_equivariant: 是否使用等变网络
        num_objects_min: 最小总物体数
        num_objects_max: 最大总物体数
        num_targets: 保留兼容参数，目录名不再额外叠加
    [输出]: str, 完整的检查点目录路径

    [命名规则]:
        - 等变网络(UNet2): equi_obj_X 或 equi_obj_X_Y
        - FC精炼器(消融实验): fc_obj_X 或 fc_obj_X_Y
        - X, Y 为总物体数
    """
    prefix = "equi" if use_equivariant else "fc"

    if num_objects_min == num_objects_max:
        obj_suffix = f"obj_{num_objects_min}"
    else:
        obj_suffix = f"obj_{num_objects_min}_{num_objects_max}"

    dir_name = f"{prefix}_{obj_suffix}"

    return os.path.join(base_dir, dir_name)


def print_training_log(mode, **kwargs):
    """
    [功能]: 统一的训练日志打印函数
    [输入]: mode ('step'|'episode'|'progress'), **kwargs (相关参数)
    """
    if mode == 'step':
        # 模式1：打印每一步的详细信息
        step = kwargs['step']
        max_steps = kwargs['max_steps']
        infos = kwargs.get('infos', [])
        step_loss = kwargs.get('step_loss')
        epsilon = kwargs['epsilon']
        agent = kwargs['agent']
        min_buffer_size = kwargs['min_buffer_size']
        actions = kwargs.get('actions', [])
        invalid_actions_list = kwargs.get('invalid_actions_list', [])
        rewards = kwargs.get('rewards', [])
        dones = kwargs.get('dones', [])
        strategy_types = kwargs.get('strategy_types', [])  # 获取策略类型列表
        
        print(f"\nStep {step}/{max_steps}")
        print("-" * 15)
        
        # 1. 打印全局状态 (只打印一次)
        # 根据实际策略类型确定探索状态
        if strategy_types and len(strategy_types) > 0:
            num_exploit = sum(1 for s in strategy_types if s == 'exploit')
            num_explore = sum(1 for s in strategy_types if s == 'explore')
            if num_explore > num_exploit:
                explore_status = "随机探索"
            elif num_exploit > num_explore:
                explore_status = "趋势利用"
            else:
                explore_status = "混合策略"
        else:
            # Fallback: 基于 epsilon（如果没有策略类型信息）
            explore_status = "随机探索" if epsilon > 0.05 else "趋势利用"
            if epsilon == 0: explore_status = "完全利用"
        print(f"  探索状态: {explore_status} (Epsilon: {epsilon:.3f})")
        
        # 2. 遍历每个环境打印详细判定
        num_envs = len(infos) if infos else 1
        for i in range(num_envs):
            is_done = dones[i] if i < len(dones) else False
            status_str = " (已结束)" if is_done else " (未结束)"
            
            print("\n" + "=" * 50)
            print(f"Environment ：{i}{status_str}")
            print("=" * 50)
            
            # A. 动作与屏蔽信息
            if i < len(actions):
                action_idx = actions[i]  # 离散动作索引 0-7
                
                # 判断动作类型
                if action_idx <= 3:
                    action_type = "推目标"
                    direction_deg = action_idx * 90
                else:
                    action_type = "推障碍"
                    direction_deg = (action_idx - 4) * 90
                
                print(f"  动作选择: {action_type} (Index {action_idx}, 方向{direction_deg}°)")

                
                if i < len(invalid_actions_list) and invalid_actions_list[i]:
                    # invalid_actions_list现在存储动作索引(整数)，不是(u,v,d)元组
                    ias_details = ", ".join([f"Act{a}" for a in invalid_actions_list[i]])
                    print(f"  [IAS] 屏蔽动作: {ias_details}")
            
            print("-" * 10)
            
            # B. 判定细节
            if i < len(infos):
                env_info = infos[i]

                if not env_info.get('already_done', False):
                    print(format_contact_action(env_info))
                
                # 出界
                is_out = env_info.get('out_of_bounds', False)
                out_msg = f"✗ 出界 (原因: {env_info.get('out_reason', 'none')})" if is_out else "✓ 未出界"
                print(f"  出界判定: {out_msg}")
                
                # 成功
                success = env_info.get('success', False)
                sep = env_info.get('separation_metrics', {})
                detail_str = sep.get('threshold', '')  # detail_str stored in 'threshold' field
                print(f"  成功判定: {'✓ 成功' if success else '× 未成功'} ({detail_str})")
                
                # 空推
                # [旧深度图日志保留，已停用]
                # is_empty = env_info.get('empty_push', False)
                # emp = env_info.get('empty_metrics', {})
                # print(f"  推动判定: {'⚠ 空推' if is_empty else '✓ 有效'} "
                #       f"(变化: {int(emp.get('change_value', 0))}/"
                #       f"{emp.get('total_pixels', 1)} ({emp.get('change_ratio', 0.0):.2f}%))")
                print(f"  {format_push_effectiveness(env_info)}")
                
                print("-" * 8)
                
                # C. 奖励拆解
                if 'reward_breakdown' in env_info:
                    breakdown = env_info['reward_breakdown']
                    reward_parts = [f"{v:+.1f}({k})" for k, v in breakdown.items() if v != 0]
                    total_r = sum(breakdown.values())
                    print(f"  单步奖励: {' '.join(reward_parts) if reward_parts else '0.0'} = {total_r:.1f}")
            
            if i < len(rewards):
                pass # 已在拆解中打印

        # 3. 打印 Loss (每步一次)
        print("\n" + "-" * 15)
        if step_loss is not None:
            print(f"  全局Loss: {step_loss:.4f}")
        else:
            print(f"  全局Loss: N/A (缓冲区记录数: {len(agent.replay_buffer)})")
        print("-" * 15 + "\n")
    
    elif mode == 'episode':
        # 模式2：打印Episode总结
        episode = kwargs['episode']
        total_steps = kwargs['total_steps']
        max_steps = kwargs['max_steps']
        total_reward = kwargs['total_reward']
        buffer_size = kwargs['buffer_size']
        buffer_capacity = kwargs['buffer_capacity']
        num_envs = kwargs.get('num_envs', 1)
        env_rewards = kwargs.get('env_rewards', [])
        
        print("\n" + "=" * 40)
        print(f"  Episode {episode} 总结")
        print("=" * 40)
        print(f"  总步数: {total_steps}/{max_steps}")
        print(f"  总奖励: {total_reward:.2f}")
        
        # [新增] 分环境显示奖励
        if env_rewards and num_envs > 1:
            print(f"  各环境奖励:")
            for env_idx, reward in enumerate(env_rewards):
                print(f"    Env {env_idx}: {reward:.2f}")
        
        # [已移除] 动作选择统计（减少日志输出）
        # action_counts = kwargs.get('action_counts', [])
        # action_explore_counts = kwargs.get('action_explore_counts', [])
        # action_exploit_counts = kwargs.get('action_exploit_counts', [])
        # 
        # if action_counts:
        #     print(f"  动作选择统计:")
        #     for action_idx, count in enumerate(action_counts):
        #         if action_idx <= 3:
        #             action_desc = f"推目标{action_idx*90}°"
        #         else:
        #             action_desc = f"推障碍{(action_idx-4)*90}°"
        #         
        #         explore = action_explore_counts[action_idx] if action_explore_counts else 0
        #         exploit = action_exploit_counts[action_idx] if action_exploit_counts else 0
        #         print(f"    Index {action_idx} ({action_desc}): {count}次 (探索{explore} + 利用{exploit})")
        
        # [修改] 只显示最近100次环境任务的成功率
        recent_100_env_results = kwargs.get('recent_100_env_results', None)
        if recent_100_env_results and len(recent_100_env_results) > 0:
            total_success_in_100 = sum(1 for r in recent_100_env_results if r)
            total_attempts_in_100 = len(recent_100_env_results)
            success_rate_100 = 100.0 * total_success_in_100 / total_attempts_in_100
            print(f"  近{total_attempts_in_100}次环境成功率: {success_rate_100:.2f}% ({total_success_in_100}/{total_attempts_in_100})")
        
        print(f"  缓冲区: {buffer_size}/{buffer_capacity}")
    
    elif mode == 'progress':
        # 模式3：打印每10个episode的进度报告
        episode = kwargs['episode']
        total_episodes = kwargs['total_episodes']
        avg_reward_10 = kwargs['avg_reward_10']
        global_step = kwargs['global_step']
        
        print("\n" + "█" * 80)
        print(f"  【进度报告】Episode {episode}/{total_episodes}")
        print("█" * 80)
        print(f"  近10ep平均奖励: {avg_reward_10:.2f}")
        print(f"  总训练步数: {global_step}")
        print("█" * 80 + "\n")
