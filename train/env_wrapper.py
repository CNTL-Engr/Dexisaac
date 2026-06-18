"""
环境包装器：将 Isaac Sim 场景封装为强化学习环境

提供类似 OpenAI Gym 的接口
"""

import torch
import numpy as np
import sys
import os

from success_judge import SuccessSeparationMixin

# 添加 src 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_path)


class PushEnv(SuccessSeparationMixin):
    """
    推操作强化学习环境
    """
    def __init__(self, scene, args):
        """
        Args:
            scene: Scene 实例
            args: 参数
        """
        self.scene = scene
        self.device = args.device
        self.num_envs = scene.num_envs
        self.max_steps_per_episode = args.episode_max_steps

        # 环境状态
        self.current_step = 0
        self.env_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # 物体引用（用于清理）
        self.spawned_objects = None

        # 物体初始位置追踪（用于调试）
        self.initial_obj_positions = {}

        # 目标位置跟踪
        self.initial_target_pos = None
        self.previous_target_pos = None
        self.goal_region_center = torch.tensor([0.75, 0.0, 0.1], device=self.device)
        self.goal_radius = 0.15

        # 空推检测
        self.previous_depth_imgs = {}
        self.previous_actions = [None for _ in range(self.num_envs)]
        self.opposite_action_streaks = [0 for _ in range(self.num_envs)]
        self.previous_empty_pushes = [False for _ in range(self.num_envs)]
        self.num_objects_min = args.num_objects_min
        self.num_objects_max = args.num_objects_max

        # 空推深度图调试保存（默认关闭，由评估脚本通过命令行开关注入目录）
        # depth_debug_dir 为 None 时不保存任何图片
        self.depth_debug_dir = None
        self.depth_debug_episode = 0  # 当前 episode 序号（由评估循环递增，用于文件命名）
        # 每个环境的目标主体抓取模板；reset/场景生成后预计算，step 中只重建清空区域
        self.target_grasp_specs = [None for _ in range(self.num_envs)]
        # IK失败黑名单：记录失败环境，强制清零或制
        self.ik_failed_blacklist = set()  # 存储env_idx

    def reset(self, env_indices=None, force_task_config=None):
        '''
        [功能]: 重置环境
        [输入]: env_indices (list): 要重置的环境索引列表
                force_task_config (dict): MAML 的特定任务配置
        [输出]: states (torch.Tensor): 重置后的环境状态
        '''
        if env_indices is None:
            env_indices = range(self.num_envs)

        # 清理旧物体
        if self.spawned_objects is not None:
            self.scene._delete_objects(self.spawned_objects, env_ids_to_delete=None)  # 删除所有环境的物体
            self.spawned_objects = None

        # 重置 scene（清除缓存配置，确保每轮重新随机生成物体数量和布局）
        if force_task_config is None and hasattr(self.scene, '_global_spawn_config'):
            del self.scene._global_spawn_config
        spawned_objects = self.scene.create_clutter_environment(
            num_objects_range=(self.num_objects_min,self.num_objects_max),
            force_task_config=force_task_config
        )

        # 保存物体引用
        self.spawned_objects = spawned_objects

        # 保存初始位置（用于调试移动距离）
        self.initial_obj_positions = {}
        for obj in spawned_objects:
            obj_name = obj.cfg.prim_path.split('/')[-1]
            pos = obj.data.root_pos_w[0]
            self.initial_obj_positions[obj_name] = pos.clone()

        # 重置环境状态
        self.current_step = 0
        self.env_dones.fill_(False)
        self.env_steps.fill_(0)
        self.previous_actions = [None for _ in range(self.num_envs)]
        self.opposite_action_streaks = [0 for _ in range(self.num_envs)]
        self.previous_empty_pushes = [False for _ in range(self.num_envs)]

        # 清空IK失败黑名单
        self.ik_failed_blacklist.clear()

        # [修复] 清除所有robot的IK失败状态，确保新回合全部重新开始
        for robot in self.scene.robots:
            if hasattr(robot, 'ik_fail_indices'):
                robot.ik_fail_indices.clear()

        # 获取目标物体初始位置
        self.initial_target_pos = self._get_target_position(spawned_objects)
        self.previous_target_pos = self.initial_target_pos.clone()

        # 基于初始分割图预计算目标主体抓取区域和两侧清空区域模板
        self._precompute_target_grasp_specs(spawned_objects, env_ids=env_indices)

        # 获取初始状态
        states = self._get_observations(spawned_objects)

        return states, spawned_objects

    def step(self, actions, spawned_objects):
        """
        [功能]: 执行动作（支持多环境同步执行）
        [输入]: actions: List of (u, v, direction) tuples for each env
                spawned_objects: 当前场景的物体列表
        [输出]: next_states: (num_envs, 3, 320, 320)
                rewards: (num_envs,) tensor
                dones: (num_envs,) tensor (bool)
                infos: list of dict
        """

        # 更新每个环境的步数（用于动态步数惩罚）
        for env_idx in range(self.num_envs):
            if not self.env_dones[env_idx]:
                self.env_steps[env_idx] += 1

        # 保存推动前的掩膜（用于空推检测）
        self._save_previous_masks(spawned_objects)


        # 检查哪些环境IK已失败，直接标记为失败，不执行动作
        ik_failed_envs = []
        for env_idx, robot in enumerate(self.scene.robots):
            if hasattr(robot, 'ik_fail_indices') and env_idx in robot.ik_fail_indices:
                ik_failed_envs.append(env_idx)
                self.env_dones[env_idx] = True
                print(f"⚠️ [Env {env_idx}] IK已失败，跳过动作执行")

        # 只对非IK失败的环境执行推动动作
        active_envs = [i for i in range(self.num_envs) if i not in ik_failed_envs]

        if active_envs:
            # 同步执行推动动作（方法内部会检查env_dones跳过失败环境）
            self._execute_push_batch(actions, spawned_objects)

        # **强制清零黑名单环境的控制**
        self._enforce_blacklist_zero_control()

        # **额外执行几步物理模拟，让物体稳定并更新位置**
        for _ in range(25):  # 执行25步物理模拟，约0.25秒
            self.scene.step()
        # 更新所有物体的数据
        for obj in spawned_objects:
            obj.update(dt=0.01)

        # 崩飞检测：检查每个环境的物体是否飞出工作空间范围外
        exploded_envs = self._check_exploded_objects(spawned_objects)
        for env_idx in exploded_envs:
            if not self.env_dones[env_idx]:
                print(f"💥 [Env {env_idx}] 检测到物体崩飞，标记为失败并等待重置")
                self.env_dones[env_idx] = True

        # 获取新状态
        next_states = self._get_observations(spawned_objects)

        # 计算奖励
        rewards, infos = self._compute_rewards(spawned_objects, actions)

        # 将崩飞信息添加到infos中
        for env_idx in exploded_envs:
            infos[env_idx]['is_exploded'] = True
            infos[env_idx]['success'] = False
            infos[env_idx]['failed'] = False
            infos[env_idx]['out_of_bounds'] = False
            infos[env_idx]['out_reason'] = '物体崩飞'

        # [FailSafe] 检查 IK 失败状态
        # 如果机器人报告有 IK 失败，这些环境也视为 Done (并且由于惩罚已经给在 _compute_rewards 中)
        # 只要确保 dones 更新即可

        # 判断是否结束
        dones = self._check_dones(infos)

        # 4. 更新全局完成状态
        for env_idx in range(self.num_envs):
            if not self.env_dones[env_idx]:
                 if dones[env_idx]:
                     self.env_dones[env_idx] = True
                     # print(f"Environment {env_idx} finished.")

        # 5. 同步重置 (Synchronous Reset)
        # 只要有一个环境没结束，其他已结束的环境就等待
        # 只有当所有环境都结束时，才触发全局重置
        if all(self.env_dones):
            print(f"[Sync-Reset] 所有环境已完成，触发全局重置...")

            # 不再在此处调用 self.reset()，避免生成物体数量不符的场景。
            # train.py 在下一个 episode 开头会调用 env.reset(force_task_config=...) 生成正确的场景。
            # 这里只需要保证 next_states 不影响 Agent 决策即可（dones=True，不会被使用）。

        self.previous_target_pos = self._get_target_position(self.spawned_objects)

        return next_states, rewards, dones, infos

    def reset_idx(self, env_ids):
        """
        [功能]: 重置指定的环境 (部分重置)
        [输入]: env_ids (list): 需要重置的环境索引列表
        """
        if not env_ids:
            return

        # 1. 删除旧物体
        self.scene._delete_objects(self.spawned_objects, env_ids_to_delete=env_ids)

        # 2. 生成新物体
        new_objects = self.scene.create_clutter_environment(
            num_objects_range=[self.num_objects_min, self.num_objects_max],
            env_ids=env_ids
        )
        self.spawned_objects.extend(new_objects)

        # 3. 重置机器人状态
        for env_id in env_ids:
            self.scene.robots[env_id].reset()

        # 只为被部分重置的环境更新抓取模板
        self._precompute_target_grasp_specs(self.spawned_objects, env_ids=env_ids)

        # 4. 刷新状态对象 (主要是更新内部的物体引用 if needed? State get_state dynamically uses spawned_objects)
        # State 类是 stateless 的，但 _get_observations 依赖 self.spawned_objects 参数，已经更新。

    def _execute_push_batch(self, actions, spawned_objects):
        """
        [功能]: 批量执行推动动作（所有环境同步）
        [输入]: actions (List[int]): 每个环境的离散动作索引 (0-7)
                spawned_objects (list)
        """
        from action_primitive import compute_push_point_from_action

        push_points = []
        direction_indices = []
        active_envs = []

        for env_idx in range(self.num_envs):
            # 跳过已完成的环境
            if self.env_dones[env_idx]:
                continue

            action_idx = actions[env_idx]  # 离散动作索引 0-7

            try:
                # 计算推点和方向
                state = self.scene.states[env_idx]
                # 获取当前环境的物体列表
                # 获取当前环境的物体列表
                # [Fix] 健壮的物体过滤逻辑 (支持 List[List] 和 List[Obj])
                env_objects = []
                if spawned_objects and len(spawned_objects) > 0:
                    first_item = spawned_objects[0]

                    if isinstance(first_item, list):
                        # 已经是分组好的 List[List]
                        if env_idx < len(spawned_objects):
                            env_objects = spawned_objects[env_idx]
                    else:
                        # 扁平列表，根据 prim_path 过滤
                        target_path_segment = f"/Scene_{env_idx}/"
                        # 单环境特例
                        is_single_env_path = (self.num_envs == 1) and ("/Scene/" in first_item.cfg.prim_path and "/Scene_0/" not in first_item.cfg.prim_path)

                        for obj in spawned_objects:
                            path = obj.cfg.prim_path
                            if target_path_segment in path:
                                env_objects.append(obj)
                            elif is_single_env_path and "/Scene/" in path and "/Scene_" not in path:
                                # 处理 /World/Scene/Obj... 格式
                                env_objects.append(obj)

                push_point, direction_idx = compute_push_point_from_action(
                    action_idx, env_idx, state, self.scene, env_objects
                )

                push_points.append(push_point)
                direction_indices.append(direction_idx)
                active_envs.append(env_idx)

            except Exception as e:
                print(f"❌ [Env{env_idx}] 动作{action_idx}计算失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认推点（环境中心）
                push_point = torch.tensor([0.75, 0.0, 0.1], device=self.device)
                push_points.append(push_point)
                direction_indices.append(0)
                active_envs.append(env_idx)

        # 如果没有active环境，直接返回
        if not active_envs:
            return

        # 批量执行（所有机械臂同时动作）
        # [Sync Fix] 同步执行推操作
        # 1. 为所有 Active 环境生成计划
        env_plans = {}
        for env_idx in range(self.num_envs):
            if env_idx not in active_envs:
                continue

            robot = self.scene.robots[env_idx]
            # 获取路径规划 (segments list)
            plan = robot.get_push_plan(
                direction_index=direction_indices[active_envs.index(env_idx)],
                push_center=push_points[active_envs.index(env_idx)]
            )

            # 初始化状态
            # 计算起始位置 (参考 Robot.move_to 的 offset 逻辑)
            from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, quat_slerp

            offset_vec = torch.tensor([0.0, 0.0, 0.2333], device=self.device)
            ee_pos = robot.get_end_effector_pose(robot.ee_body_name)[0]
            ee_quat = robot.get_end_effector_pose(robot.ee_body_name)[1]

            base_pos = robot.articulation.data.root_pos_w
            base_quat = robot.articulation.data.root_quat_w

            # Start Pos (Local)
            start_pos = quat_apply(quat_inv(base_quat), (ee_pos - base_pos)) - offset_vec

            # Start Quat (Local)
            start_quat = quat_mul(quat_inv(base_quat), ee_quat)

            env_plans[env_idx] = {
                'plan': plan,
                'segment_idx': 0,
                'elapsed_time': 0.0,
                'start_pos': start_pos,
                'start_quat': start_quat,
                'done': False,
                'stable_steps': 0
            }

        # 2. 同步执行循环
        dt = 0.01  # Simulation dt
        all_finished = False
        max_iterations = 500  # 全局安全上限：500次循环 ≈ 5秒仿真时间
        iteration_count = 0

        from isaaclab.utils.math import quat_slerp

        while not all_finished and self.scene.is_app_running():
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"⚠ [_execute_push_batch] 超过最大循环次数 {max_iterations}，强制退出")
                for env_idx in active_envs:
                    if not env_plans[env_idx]['done']:
                        env_plans[env_idx]['done'] = True
                        print(f"  ⚠ Env {env_idx} 未完成，强制标记为完成")
                break

            all_finished = True

            # A. 为每个Robot设置命令
            for env_idx in active_envs:
                state = env_plans[env_idx]
                if state['done']:
                    continue


                all_finished = False # 只要有一个没做完，就还没结束

                robot = self.scene.robots[env_idx]

                # [FailSafe] 在发送命令前检查 IK 失败状态
                # 如果该环境已经失败，跳过该环境，其他环境继续
                if env_idx in robot.ik_fail_indices:
                    # 只在第一次失败时处理
                    if not state.get('ik_printed', False):
                        print(f"❌ [Env {env_idx}] IK解算失败，加入黑名单，强制清零控制")

                        # 加入黑名单
                        self.ik_failed_blacklist.add(env_idx)

                        # 立即清零该环境控制
                        self._zero_robot_control(env_idx)

                        state['ik_printed'] = True

                    # 标记该环境完成，后续循环会自动跳过
                    state['done'] = True
                    continue  # 跳过此环境，继续处理其他环境

                plan = state['plan']
                segment = plan[state['segment_idx']]

                target_pos = segment['target_pos']
                target_quat = segment['target_quat']
                speed = segment['speed']
                gripper_pos = segment.get('gripper_pos', 1.0)

                # 计算 Duration
                duration = torch.norm(target_pos - state['start_pos']) / speed
                duration = max(duration.item(), dt) # 避免除零

                # 插值
                alpha = min(state['elapsed_time'] / duration, 1.0)

                interp_pos = state['start_pos'] + (target_pos - state['start_pos']) * alpha

                # [Fix] 处理None姿态：当target_quat为None时，使用当前姿态（保持不变）
                if target_quat is None:
                    interp_quat = state['start_quat']
                else:
                    # [Fix] 压缩维度以适配 quat_slerp (需 1D Tensor), 然后还原
                    q1 = state['start_quat'].squeeze()
                    q2 = target_quat.squeeze()
                    interp_quat_1d = quat_slerp(q1, q2, alpha)
                    interp_quat = interp_quat_1d.unsqueeze(0) # (4) -> (1, 4)

                # 发送命令
                robot.move_ik(interp_pos, interp_quat)
                robot.move_gripper(gripper_pos)
                robot.write()

            # B. 物理步进 (所有Robot同时动)
            # [修改] 增加物理步数以提高插值密度，降低奇异点风险
            # 原来：1步/cycle，现在：3步/cycle
            self.scene.step()

            # [关键优化] 物理步进后立即检查IK失败，快速退出
            # 检查是否有任何环境IK失败，如果有则标记all_finished=True跳出主循环
            ik_failed_now = False
            for env_idx in active_envs:
                robot = self.scene.robots[env_idx]
                if hasattr(robot, 'ik_fail_indices') and env_idx in robot.ik_fail_indices:
                    ik_failed_now = True
                    # 立即标记为done
                    if env_idx in env_plans and not env_plans[env_idx].get('done', False):
                        env_plans[env_idx]['done'] = True
                        env_plans[env_idx]['elapsed_time'] = 999.0

            # 如果检测到IK失败，提前检查是否所有环境都完成
            if ik_failed_now:
                all_finished = all(env_plans[i]['done'] for i in active_envs if i in env_plans)
                if all_finished:
                    break  # 立即跳出while循环

            # C. 更新状态与检查结束
            for env_idx in active_envs:
                state = env_plans[env_idx]
                if state['done']:
                    continue

                robot = self.scene.robots[env_idx]
                # [修改] 由于每个循环执行3次物理步，elapsed_time也要相应增加
                robot.update(dt * 3)
                state['elapsed_time'] += dt * 3

                plan = state['plan']
                segment = plan[state['segment_idx']]
                target_pos = segment['target_pos']

                # 检查误差 & 时间
                # 重复 move_to 的误差检查逻辑
                offset_vec = torch.tensor([0.0, 0.0, 0.2333], device=self.device)

                # [FailSafe] 优先检查 IK 失败 - 在所有其他检查之前
                # 如果该环境已经失败，跳过该环境
                if env_idx in robot.ik_fail_indices:
                    # 只在第一次失败时打印
                    if not state.get('ik_printed', False):
                        print(f"❌ [IK FailSafe] Env {env_idx} IK 失败 (Singularity/Error)")
                        state['ik_printed'] = True

                    # [关键修复] 每次循环都强制停止该环境的机器人运动
                    # 设置关节position target为当前位置（冻结）
                    current_joint_pos = robot.articulation.data.joint_pos[0]
                    robot.articulation.set_joint_position_target(
                        current_joint_pos.unsqueeze(0),
                        joint_ids=None,
                        env_ids=[env_idx]
                    )
                    # 设置速度和力为0
                    robot.articulation.set_joint_velocity_target(
                        torch.zeros(robot.articulation.num_joints, device=self.device).unsqueeze(0),
                        joint_ids=None,
                        env_ids=[env_idx]
                    )
                    robot.articulation.set_joint_effort_target(
                        torch.zeros(robot.articulation.num_joints, device=self.device).unsqueeze(0),
                        joint_ids=None,
                        env_ids=[env_idx]
                    )

                    # [优化] 立即标记为超时，确保快速退出
                    state['elapsed_time'] = 999.0  # 设置为远大于8.0的值
                    state['done'] = True
                    continue  # 跳过此环境，继续处理其他环境

                ee_pos_w = robot.get_end_effector_pose(robot.ee_body_name)[0]

                if ee_pos_w is not None:
                    # Target Global
                    base_pos = robot.articulation.data.root_pos_w
                    base_quat = robot.articulation.data.root_quat_w
                    target_w = quat_apply(base_quat, target_pos + offset_vec) + base_pos

                    # 位置误差
                    pos_error = torch.norm(target_w - ee_pos_w)
                    pos_threshold = 0.02

                    # [新增] 姿态误差检查
                    quat_error = 0.0
                    quat_threshold = 0.1  # 弧度
                    if target_quat is not None:
                        ee_quat = robot.get_end_effector_pose(robot.ee_body_name)[1]
                        target_quat_global = quat_mul(base_quat, target_quat)
                        dot_product = torch.abs(torch.sum(ee_quat * target_quat_global))
                        dot_product = torch.clamp(dot_product, 0, 1)
                        quat_error = 2 * torch.acos(dot_product)

                    # 判定条件：时间到了 且 误差足够小 (或者超时保护)
                    # 计算 Duration (需要重新计算或存储)
                    duration = torch.norm(target_pos - state['start_pos']) / segment['speed']
                    duration = max(duration.item(), dt)

                    # [修改] 收敛条件：对waypoint 0需要姿态也收敛
                    is_converged = False
                    if state['segment_idx'] == 0 and target_quat is not None:
                        # Waypoint 0: 需要位置和姿态都收敛
                        if (state['stable_steps'] > 10 and state['elapsed_time'] >= duration
                            and pos_error < pos_threshold and quat_error < quat_threshold) or (state['elapsed_time'] > 1.0):
                            is_converged = True
                    else:
                        if (state['stable_steps'] > 5 and state['elapsed_time'] >= duration) or (state['elapsed_time'] > 0.5):
                            is_converged = True

                    # [修改] 稳定计数器：waypoint 0检查姿态
                    if state['segment_idx'] == 0 and target_quat is not None:
                        if pos_error < pos_threshold and quat_error < quat_threshold:
                            state['stable_steps'] += 1
                        else:
                            state['stable_steps'] = 0
                    else:
                        if pos_error < pos_threshold:
                            state['stable_steps'] += 1
                        else:
                            state['stable_steps'] = 0

                    if is_converged:
                        # 当前段结束，进入下一段
                        state['segment_idx'] += 1
                        if state['segment_idx'] >= len(plan):
                            state['done'] = True
                        else:
                            # 准备下一段
                            state['elapsed_time'] = 0.0
                            state['stable_steps'] = 0


                            # Re-capture Local Pose
                            curr_ee_pos = robot.get_end_effector_pose(robot.ee_body_name)[0]
                            curr_ee_quat = robot.get_end_effector_pose(robot.ee_body_name)[1]
                            # Localize
                            state['start_pos'] = quat_apply(quat_inv(base_quat), (curr_ee_pos - base_pos)) - offset_vec
                            state['start_quat'] = quat_mul(quat_inv(base_quat), curr_ee_quat)


    def _get_observations(self, spawned_objects):
        """
        [功能]: 获取观测（状态）
        [输入]: spawned_objects: 物体列表
        [输出]: states: (num_envs, 3, 320, 320) tensor
        """
        states = []

        for env_idx in range(self.num_envs):
            # 从 state 获取输入 tensor
            state_obj = self.scene.states[env_idx]
            state_tensor = state_obj.get_state(spawned_objects)

            if state_tensor is not None:
                states.append(state_tensor)
            else:
                # Fallback: 全零状态 (uint8)
                states.append(torch.zeros(1, 3, 320, 320, dtype=torch.uint8, device=self.device))

        return torch.cat(states, dim=0)  # (num_envs, 3, 320, 320)

    def _get_target_position(self, spawned_objects):
        """
        [功能]: 获取目标物体的当前位置
        [输入]: spawned_objects: 物体列表
        [输出]: positions: (num_envs, 3) tensor
        """
        positions = []

        for obj in spawned_objects:
            obj_name = obj.cfg.prim_path.split('/')[-1]
            if "Target_" in obj_name:
                # 获取目标物体位置
                pos = obj.data.root_pos_w[0]  # (3,)
                positions.append(pos)
                break

        if not positions:
            # 没找到目标物体，返回默认位置
            return torch.tensor([[0.75, 0.0, 0.1]], device=self.device).repeat(self.num_envs, 1)

        # 扩展到所有环境
        return positions[0].unsqueeze(0).repeat(self.num_envs, 1)

    def _compute_rewards(self, spawned_objects, actions):
        """
        [功能]: 计算离散奖励（严格优先级）
        [输入]: spawned_objects (list), actions (list)
        [输出]: (rewards: tensor, infos: list) - infos包含奖励组成详情

        优先级顺序（从高到低）：
        1. 出界惩罚（最高优先级）
        2. 成功奖励（次高优先级）- 成功时不检测空推
        3. 反向动作惩罚 - 仅在未成功时检测
        4. 空推惩罚（最低优先级）- 仅在未出界且未成功时检测
        """
        rewards = torch.zeros(self.num_envs, device=self.device)
        infos = []

        for env_idx in range(self.num_envs):
            # 已完成环境跳过奖励和检测
            if self.env_dones[env_idx]:
                infos.append({
                    'out_of_bounds': False,
                    'failed': False,
                    'empty_push': False,
                    'reward_breakdown': {},
                    'total_reward': 0.0,
                    'already_done': True  # 标记此环境已完成
                })
                rewards[env_idx] = 0.0
                continue  # 跳过奖励计算

            reward = 0.0
            reward_breakdown = {}  # 奖励组成
            info = {}
            info['opposite_action'] = False
            info['opposite_action_streak'] = 0

            # 0. 步数惩罚（固定-1）
            current_step = self.env_steps[env_idx].item()
            step_penalty = -1.0  # 固定-1
            reward += step_penalty
            reward_breakdown['步数惩罚'] = step_penalty

            # [FailSafe] 优先检查 IK 失败 (最高优先级: System Failure)
            robot = self.scene.robots[env_idx]
            if env_idx in robot.ik_fail_indices:
                ik_penalty = -10.0
                reward += ik_penalty
                reward_breakdown['IK求解失败'] = ik_penalty
                info['failed'] = True
                info['success'] = False
                info['out_of_bounds'] = False # Not necessarily out of bounds, just stuck
                info['ik_failed'] = True

                # 立即标记为 Done
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_empty_pushes[env_idx] = False
                continue


            # 1. 检查是否出界（最高优先级）
            is_out, out_reason, is_exploded = self._check_out_of_bounds(env_idx, spawned_objects)
            info['out_of_bounds'] = is_out
            info['out_reason'] = out_reason
            info['is_exploded'] = is_exploded  # [新增] 崩飞标记（物体飞出边界20cm以外）

            if is_out and is_exploded:
                # 崩飞视为仿真异常：触发外层重试，不给奖励/惩罚，也不计为普通出界
                reward = 0.0
                reward_breakdown = {}
                info['out_of_bounds'] = False
                info['failed'] = False
                info['success'] = False
                info['empty_push'] = False
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                continue

            if is_out:
                # 出界惩罚 - 最高优先级，跳过其他检测
                out_penalty = -10.0
                reward += out_penalty
                reward_breakdown['出界惩罚'] = out_penalty
                opposite_penalty = self._get_opposite_action_penalty(
                    env_idx, actions[env_idx], is_empty=False
                )
                if opposite_penalty != 0.0:
                    reward += opposite_penalty
                    reward_breakdown['反向动作惩罚'] = opposite_penalty
                    info['opposite_action'] = True
                    info['opposite_action_streak'] = self.opposite_action_streaks[env_idx]
                info['failed'] = True
                info['success'] = False
                info['empty_push'] = False  # 出界时不检测空推

                # 保存并立即返回当前环境的结果
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = False
                continue  # 跳过后续检测

            # 2. 检查成功（次高优先级）
            success, separation_sim, separation_threshold = self._check_successful_separation(env_idx, spawned_objects) # type: ignore
            info['success'] = success
            info['failed'] = False
            info['separation_metrics'] = {'similarity': separation_sim, 'threshold': separation_threshold}

            if success:
                # 成功奖励 - 次高优先级，不检测空推
                success_reward = 10.0
                reward += success_reward
                reward_breakdown['成功分离'] = success_reward
                info['empty_push'] = False  # 成功时不检测空推

                # 保存并立即返回当前环境的结果
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = False
                continue  # 跳过空推检测

            # 3. 检查空推（在超时检测之前，确保每步都有空推数据用于日志）
            is_empty, empty_value, empty_total, empty_ratio, empty_threshold = self._check_empty_push(env_idx)
            info['empty_push'] = is_empty
            info['empty_metrics'] = {
                'change_value': empty_value,
                'total_pixels': empty_total,
                'change_ratio': empty_ratio,
                'threshold': empty_threshold
            }

            if is_empty:
                # 空推惩罚
                empty_penalty = -5.0
                reward += empty_penalty
                reward_breakdown['空推惩罚'] = empty_penalty

            # 4. 检查是否与上一步反方向推动（空推相关动作豁免该惩罚）
            opposite_penalty = self._get_opposite_action_penalty(
                env_idx, actions[env_idx], is_empty=is_empty
            )
            if opposite_penalty != 0.0:
                reward += opposite_penalty
                reward_breakdown['反向动作惩罚'] = opposite_penalty
                info['opposite_action'] = True
                info['opposite_action_streak'] = self.opposite_action_streaks[env_idx]

            # 5. 检查是否超过最大步数（仅在未出界且未成功时）
            if current_step >= self.max_steps_per_episode:
                max_steps_penalty = -10.0
                reward += max_steps_penalty
                reward_breakdown['超过最大步数'] = max_steps_penalty
                info['failed'] = True
                info['success'] = False
                info['out_of_bounds'] = False
                info['max_steps_exceeded'] = True

            # 保存奖励信息
            info['reward_breakdown'] = reward_breakdown
            info['total_reward'] = reward
            rewards[env_idx] = reward
            infos.append(info)
            self.previous_actions[env_idx] = actions[env_idx]
            self.previous_empty_pushes[env_idx] = bool(is_empty)

        # 奖励归一化：降低Q值和Target方差，提高训练稳定性
        # 将奖励范围从[-10, 10]缩放到[-1, 1]
        rewards = rewards  / 10

        return rewards, infos

    def _is_opposite_repeat_action(self, env_idx, current_action):
        """
        检查当前动作是否与上一步为反方向动作。
        动作0-3为推目标，4-7为推障碍；方向索引均为 action % 4。
        """
        previous_action = self.previous_actions[env_idx]
        if previous_action is None:
            return False

        previous_action = int(previous_action)
        current_action = int(current_action)

        opposite_direction = (previous_action % 4 - current_action % 4) % 4 == 2

        return opposite_direction

    def _get_opposite_action_penalty(self, env_idx, current_action, is_empty=False):
        """
        根据连续反方向推动次数返回递增惩罚。
        第1次=-2，第2次及之后=-5。
        非反向、上一step空推或当前step空推都会清零连续计数。
        """
        if not self._is_opposite_repeat_action(env_idx, current_action):
            self.opposite_action_streaks[env_idx] = 0
            return 0.0

        if self.previous_empty_pushes[env_idx] or is_empty:
            self.opposite_action_streaks[env_idx] = 0
            return 0.0

        self.opposite_action_streaks[env_idx] += 1
        streak = self.opposite_action_streaks[env_idx]

        if streak == 1:
            return -2.0
        return -5.0

    def _save_previous_masks(self, spawned_objects):
        """
        [功能]: 保存推动前的深度图（用于空推检测）
        [输入]: spawned_objects (list)

        使用 state 的 320x320 裁剪深度图
        """
        import numpy as np
        for env_idx in range(self.num_envs):
            state = self.scene.states[env_idx]
            # 使用 state 的方法获取处理后的 320x320 图像
            # 旧逻辑: 使用 normalize_depth=True 的 0-255 归一化深度图
            # images = state.get_img(hide_robot=True)
            images = state.get_img(hide_robot=True, normalize_depth=False)
            if images is not None:
                _, depth_320, _ = images
                # depth_320 已经是 (320, 320) 的 numpy 数组
                # 旧逻辑: 保留原始 dtype，通常是 uint8 归一化深度
                # self.previous_depth_imgs[env_idx] = depth_320.copy()
                self.previous_depth_imgs[env_idx] = depth_320.astype(np.float32).copy()

    def _check_empty_push(self, env_idx, change_threshold=450):
        """
        [功能]: 检查是否为空推（推动前后深度图变化像素数 < 阈值）
        [输入]: env_idx (int), change_threshold (int): 变化像素数阈值，默认200
        [输出]: bool

        检测逻辑：
        1. 计算推动前后深度图的绝对差异
        2. 过滤 NaN 值、过大值(>1.0m)、过小变化(<5mm)
        3. 统计显著变化的像素数量
        4. 如果变化像素数 < 阈值，判定为空推
        """
        import numpy as np

        if env_idx not in self.previous_depth_imgs:
            return False, 0, 0, 0.0, change_threshold

        state = self.scene.states[env_idx]
        # 获取处理后的 320x320 深度图
        # 旧逻辑: 使用 normalize_depth=True 的 0-255 归一化深度图
        # images = state.get_img(hide_robot=True)
        images = state.get_img(hide_robot=True, normalize_depth=False)
        if images is None:
            return False, 0, 0, 0.0, change_threshold

        _, current_depth, _ = images  # (320, 320) numpy array
        # 旧逻辑: 直接使用缓存 dtype
        # previous_depth = self.previous_depth_imgs[env_idx]  # (320, 320) numpy array
        current_depth = current_depth.astype(np.float32)
        previous_depth = self.previous_depth_imgs[env_idx].astype(np.float32)  # (320, 320) numpy array

        # 调试：打印深度图的范围


        # 计算深度差异
        depth_diff = np.abs(previous_depth - current_depth)

        # 旧逻辑: 根据深度范围自适应处理 0-255 归一化深度或米制深度
        # depth_max = max(previous_depth.max(), current_depth.max())
        #
        # if depth_max > 10:  # 深度图在0-255范围
        #     # 对于0-255范围的深度图
        #     depth_diff[np.isnan(depth_diff)] = 0  # 过滤 NaN
        #     depth_diff[depth_diff > 50] = 0       # 过滤过大值（可能是噪声）
        #     depth_diff[depth_diff < 2] = 0        # 过滤过小变化（<2灰度级，不显著）
        # else:  # 深度图在0-1范围（归一化的实际深度）
        #     # 对于0-1范围的深度图
        #     depth_diff[np.isnan(depth_diff)] = 0  # 过滤 NaN
        #     depth_diff[depth_diff > 1.0] = 0      # 过滤过大值（>1m，可能是噪声）
        #     depth_diff[depth_diff < 0.005] = 0    # 过滤过小变化（<5mm，不显著）

        # 新逻辑: 空推检测固定使用 normalize_depth=False 的米制深度
        depth_diff = np.nan_to_num(depth_diff, nan=0.0, posinf=0.0, neginf=0.0)
        depth_diff[depth_diff > 0.3] = 0      # 过滤过大值（>0.3m，可能是噪声）
        depth_diff[depth_diff < 0.015] = 0    # 过滤过小变化（<0.015m，不显著）

        # 将剩余的显著变化标记为1
        depth_diff[depth_diff > 0] = 1

        # 统计变化像素数量
        change_value = np.sum(depth_diff)
        total_pixels = depth_diff.size
        change_ratio = (change_value / total_pixels) * 100

        is_empty = change_value < change_threshold

        # 调试：保存前一帧、后一帧深度图与二值变化掩码
        if self.depth_debug_dir is not None:
            self._save_depth_debug(
                env_idx, previous_depth, current_depth, depth_diff,
                change_value, is_empty
            )

        # 变化像素数 < 阈值 → 空推
        return is_empty, change_value, total_pixels, change_ratio, change_threshold

    def _save_depth_debug(self, env_idx, previous_depth, current_depth,
                          change_mask, change_value, is_empty):
        """
        [功能]: 保存空推检测的调试图片（前一帧深度图、后一帧深度图、二值变化掩码）
        [输入]:
          env_idx (int): 环境索引
          previous_depth (np.ndarray): 推动前深度图 (320,320)
          current_depth (np.ndarray): 推动后深度图 (320,320)
          change_mask (np.ndarray): 二值变化掩码 (320,320)，值为 0/1
          change_value (int): 变化像素数
          is_empty (bool): 是否判定为空推

        命名: ep{episode}_env{env_idx}_step{step}_chg{change_value}_{empty/valid}_{prev/curr/diff}.png
        所有图片均为 PNG 可视化。
        """
        import cv2
        import numpy as np

        save_dir = self.depth_debug_dir
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)

        # step 序号取当前环境步数（_save_previous_masks 在 step 自增后调用）
        step = int(self.env_steps[env_idx].item()) if hasattr(self.env_steps, 'item') \
            else int(self.env_steps[env_idx])
        ep = int(self.depth_debug_episode)
        tag = "empty" if is_empty else "valid"
        prefix = f"ep{ep}_env{env_idx}_step{step}_chg{int(change_value)}_{tag}"

        # 米制深度图：PNG 仅用于同尺度可视化。
        # 旧逻辑: 直接转 uint8，适用于 normalize_depth=True 的 0-255 深度图
        # prev_vis = np.nan_to_num(previous_depth).astype(np.uint8)
        # curr_vis = np.nan_to_num(current_depth).astype(np.uint8)
        prev_depth = np.nan_to_num(previous_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        curr_depth = np.nan_to_num(current_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        valid_depth = np.concatenate([
            prev_depth[prev_depth > 0].reshape(-1),
            curr_depth[curr_depth > 0].reshape(-1),
        ])
        if valid_depth.size > 0:
            depth_min = float(np.percentile(valid_depth, 1))
            depth_max = float(np.percentile(valid_depth, 99))
            if depth_max - depth_min < 1e-6:
                depth_max = depth_min + 1e-6
        else:
            depth_min, depth_max = 0.0, 1.0

        def depth_to_vis(depth):
            clipped = np.clip(depth, depth_min, depth_max)
            vis = (1.0 - (clipped - depth_min) / (depth_max - depth_min)) * 255.0
            vis[depth <= 0] = 0
            return vis.astype(np.uint8)

        prev_vis = depth_to_vis(prev_depth)
        curr_vis = depth_to_vis(curr_depth)
        # 二值变化掩码：0/1 → 0/255，白色像素=发生变化
        diff_vis = (np.nan_to_num(change_mask) > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(save_dir, f"{prefix}_prev.png"), prev_vis)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_curr.png"), curr_vis)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_diff.png"), diff_vis)

    def _check_exploded_objects(self, spawned_objects):
        """
        [功能]: 检查每个环境是否有物体崩飞（飞出工作空间范围超过20cm）
        [输入]: spawned_objects (list)
        [输出]: list[int] - 崩飞的环境索引列表
        """
        import math

        exploded_envs = []
        explode_threshold = 0.18  # 18cm阈值

        # 工作空间限制
        workspace_limits = torch.tensor([
            [0.25, -0.50, 0.02],  # min [x, y, z]
            [1.25, 0.50, 0.4]     # max [x, y, z]
        ], device=self.device)

        # 计算每个环境的偏移量
        env_offsets = {}
        if self.num_envs > 1:
            grid_width = int(math.ceil(math.sqrt(self.num_envs)))
            for env_idx in range(self.num_envs):
                row = env_idx // grid_width
                col = env_idx % grid_width
                env_offsets[env_idx] = (row * self.scene.env_spacing, col * self.scene.env_spacing)
        else:
            env_offsets[0] = (0.0, 0.0)

        # 先更新所有物体数据
        for obj in spawned_objects:
            try:
                obj.update(dt=0.01)
            except:
                pass

        # 检查每个物体
        for obj in spawned_objects:
            try:
                # 获取物体所属环境
                obj_env_id = self.scene._get_env_id_from_prim_path(obj.cfg.prim_path)

                # 获取物体位置
                pos = obj.data.root_pos_w[0]

                # 获取环境偏移
                x_offset, y_offset = env_offsets.get(obj_env_id, (0.0, 0.0))

                # 转换为本地坐标
                local_x = pos[0].item() - x_offset
                local_y = pos[1].item() - y_offset
                local_z = pos[2].item()

                # 计算超出边界的距离
                x_exceed = max(workspace_limits[0, 0].item() - local_x, local_x - workspace_limits[1, 0].item(), 0)
                y_exceed = max(workspace_limits[0, 1].item() - local_y, local_y - workspace_limits[1, 1].item(), 0)
                z_exceed = max(workspace_limits[0, 2].item() - local_z, 0)  # 只检查掉落

                max_exceed = max(x_exceed, y_exceed, z_exceed)

                # 如果超出阈值，标记该环境为崩飞
                if max_exceed > explode_threshold:
                    if obj_env_id not in exploded_envs:
                        obj_name = obj.cfg.prim_path.split('/')[-1]
                        print(f"  💥 [Env {obj_env_id}] 物体 {obj_name} 崩飞！超出边界 {max_exceed:.2f}m")
                        exploded_envs.append(obj_env_id)

            except Exception as e:
                pass  # 静默处理异常

        return exploded_envs

    def _settle_after_out_of_bounds(self, spawned_objects, steps=40):
        """
        [功能]: 出界后额外推进物理仿真，让潜在崩飞充分显现。
        """
        for _ in range(steps):
            self.scene.step()

        for obj in spawned_objects:
            try:
                obj.update(dt=0.01)
            except Exception:
                pass

    def _check_delayed_explosion_after_out_of_bounds(self, env_idx, spawned_objects):
        self._settle_after_out_of_bounds(spawned_objects, steps=40)
        exploded_envs = self._check_exploded_objects(spawned_objects)
        return env_idx in exploded_envs

    def _check_out_of_bounds(self, env_idx, spawned_objects):
        """
        [功能]: 检查是否有物体出界（双重检测，检测所有物体）
        [输入]: env_idx (int), spawned_objects (list)
        [输出]: bool

        使用两种检测方法（或逻辑）：
        1. 物体中心坐标检测（快速检测，防止物体崩飞）- 检测所有物体的中心
        2. 深度图掩膜检测（精确边界检测）

        任一检测方法返回True则判定为出界
        """
        import torch
        import math

        # 工作空间限制（与scene.py保持一致）
        workspace_limits = torch.tensor([
            [0.25, -0.50, 0.02],  # min [x, y, z]
            [1.25, 0.50, 0.4]     # max [x, y, z]
        ], device=self.device)

        # 计算环境偏移量
        if self.num_envs > 1:
            grid_width = int(math.ceil(math.sqrt(self.num_envs)))
            row = env_idx // grid_width
            col = env_idx % grid_width
            x_offset = row * self.scene.env_spacing
            y_offset = col * self.scene.env_spacing
        else:
            x_offset, y_offset = 0.0, 0.0

        # 检测1：所有物体的中心坐标检测（防止崩飞）
        # print(f'\n[出界检测调试] Env {env_idx}:')
        # print(f'  工作空间限制: X[{workspace_limits[0,0]:.2f}, {workspace_limits[1,0]:.2f}], Y[{workspace_limits[0,1]:.2f}, {workspace_limits[1,1]:.2f}], Z>={workspace_limits[0,2]:.2f}')

        # **先更新所有物体的数据，获取最新位置**
        for obj in spawned_objects:
            obj.update(dt=0.01)  # 从物理引擎同步最新状态，使用正确的dt

        for obj in spawned_objects:
            # 只检查属于当前环境的物体
            obj_env_id = self.scene._get_env_id_from_prim_path(obj.cfg.prim_path)
            if obj_env_id != env_idx:
                continue

            pos = obj.data.root_pos_w[0]  # 全局坐标 [x, y, z]
            obj_name = obj.cfg.prim_path.split('/')[-1]

            # 转换为本地坐标
            local_x = pos[0].item() - x_offset
            local_y = pos[1].item() - y_offset
            local_z = pos[2].item()

            # 检查并打印
            x_in = workspace_limits[0, 0] <= local_x <= workspace_limits[1, 0]
            y_in = workspace_limits[0, 1] <= local_y <= workspace_limits[1, 1]
            z_in = local_z >= 0.02

            # 计算移动距离
            if obj_name in self.initial_obj_positions:
                init_pos = self.initial_obj_positions[obj_name]
                move_dist = torch.norm(pos - init_pos).item()
                # status = "✓" if (x_in and y_in and z_in) else "✗"
                # print(f'  {status} {obj_name}: 本地坐标({local_x:.3f}, {local_y:.3f}, {local_z:.3f}) | 移动距离:{move_dist:.3f}m | X:{x_in} Y:{y_in} Z:{z_in}')
            else:
                pass
                # status = "✓" if (x_in and y_in and z_in) else "✗"
                # print(f'  {status} {obj_name}: 本地坐标({local_x:.3f}, {local_y:.3f}, {local_z:.3f}) | X:{x_in} Y:{y_in} Z:{z_in}')

            # 检查XY是否在工作空间内，并计算超出距离
            if not x_in:
                # 计算超出边界的距离
                x_exceed = max(workspace_limits[0, 0].item() - local_x, local_x - workspace_limits[1, 0].item())
                is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                if env_idx >= 2:
                    print(f"  [DEBUG Env {env_idx}] ✗ OOB X: Obj={obj_name}, LocalX={local_x:.2f}, Exceed={x_exceed:.2f}m {'[崩飞]' if is_exploded else ''}")
                return True, f"物体 {obj_name} X轴出界", is_exploded
            if not y_in:
                # 计算超出边界的距离
                y_exceed = max(workspace_limits[0, 1].item() - local_y, local_y - workspace_limits[1, 1].item())
                is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                if env_idx >= 2:
                    print(f"  [DEBUG Env {env_idx}] ✗ OOB Y: Obj={obj_name}, LocalY={local_y:.2f}, Exceed={y_exceed:.2f}m {'[崩飞]' if is_exploded else ''}")
                return True, f"物体 {obj_name} Y轴出界", is_exploded
            # Z轴检查（是否掉落到桌面以下）
            if not z_in:
                is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                return True, f"物体 {obj_name} 掉落", is_exploded

        # 检测2：深度图掩膜检测（精确边界）
        state = self.scene.states[env_idx]
        out_of_bounds, check_info = state.check_out_of_bounds(verbose=False)

        out_reason = check_info.get("reason", "unknown") if out_of_bounds else "none"

        if out_of_bounds:
            is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
            return True, out_reason, is_exploded

        return False, out_reason, False

    def _check_collision(self, env_idx):
        """
        检查是否发生碰撞或出界（使用掩膜检测）

        Args:
            env_idx: 环境索引

        Returns:
            collision: bool (True = 碰撞或出界)
        """
        # 使用基于掩膜的出界检测
        state = self.scene.states[env_idx]
        out_of_bounds, info = state.check_out_of_bounds(verbose=False)

        # 如果出界，视为碰撞
        if out_of_bounds:
            return True

        # TODO: 可以添加额外的碰撞检测逻辑
        # 例如检查机器人和物体的接触力

        return False

    def _check_dones(self, infos):
        """
        [功能]: 检查哪些环境已结束（成功或失败提前结束）
        [输入]: infos (list of dict)
        [输出]: dones (num_envs,) bool tensor
        """
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_idx, info in enumerate(infos):
            # 提前结束条件：
            # 1. 成功分离
            # 2. 出界（失败）
            # 3. 崩飞异常（外层重试，不计普通失败）
            # 4. 达到最大步数
            if (
                info.get('success', False)
                or info.get('out_of_bounds', False)
                or info.get('failed', False)
                or info.get('is_exploded', False)
                or self.current_step >= self.max_steps_per_episode
            ):
                dones[env_idx] = True

        return dones

    def _zero_robot_control(self, env_idx):
        """
        [功能]: 强制清零指定环境的机械臂控制（瞬间完全静止）
        [输入]: env_idx (int): 环境索引
        [说明]: 使用最强制的方法，直接设置物理状态使机械臂立即停止
        """
        try:
            robot = self.scene.robots[env_idx]
            articulation = robot.articulation

            # 获取当前关节位置
            current_joint_pos = articulation.data.joint_pos.clone()

            # 1. 设置关节位置目标为当前位置（控制层面）
            articulation.set_joint_position_target(current_joint_pos)

            # 2. 清零速度目标（控制层面）
            zero_vel = torch.zeros_like(current_joint_pos)
            if hasattr(articulation, 'set_joint_velocity_target'):
                articulation.set_joint_velocity_target(zero_vel)

            # 3. 【强制】直接清零关节速度状态（物理层面）
            if hasattr(articulation.data, 'joint_vel'):
                articulation.data.joint_vel[:] = 0.0

            # 4. 【强制】清零根节点速度（防止整体移动）
            if hasattr(articulation.data, 'root_lin_vel_w'):
                articulation.data.root_lin_vel_w[:] = 0.0
            if hasattr(articulation.data, 'root_ang_vel_w'):
                articulation.data.root_ang_vel_w[:] = 0.0

            # 5. 写入仿真（包括位置和速度状态）
            articulation.write_data_to_sim()

            # 6. 【额外】尝试直接写入关节状态（如果支持）
            if hasattr(articulation, 'write_joint_state_to_sim'):
                articulation.write_joint_state_to_sim(
                    joint_pos=current_joint_pos,
                    joint_vel=zero_vel
                )

        except Exception as e:
            print(f"  [警告] Env {env_idx} 清零机械臂控制时出错: {e}")

    def _enforce_blacklist_zero_control(self):
        """
        [功能]: 强制清零黑名单中所有环境的机械臂控制
        [说明]: 在每次物理步进后调用，确保黑名单环境的机械臂不会乱飞
        """
        if not self.ik_failed_blacklist:
            return

        for env_idx in self.ik_failed_blacklist:
            # 复用已增强的_zero_robot_control函数
            self._zero_robot_control(env_idx)
