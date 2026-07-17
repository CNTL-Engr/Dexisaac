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
        self.empty_push_displacement_threshold = float(
            getattr(args, 'empty_push_displacement_threshold', 0.03)
        )
        self.empty_push_force_threshold = float(
            getattr(args, 'empty_push_force_threshold', 1.0)
        )
        # 动力学崩飞检测：只使用物体的三维总线速度和三维总线加速度。
        # 阈值由所有训练/评估入口显式传入；getattr 默认值兼容诊断脚本的简化 Args。
        self.explosion_linear_speed_threshold = float(
            getattr(args, 'explosion_linear_speed_threshold', 1.0)
        )
        self.explosion_linear_acceleration_threshold = float(
            getattr(args, 'explosion_linear_acceleration_threshold', 50.0)
        )
        # 连续速度异常步数，以及加速度异常步与速度异常区间允许的最大步距。
        self.explosion_abnormal_steps_threshold = int(
            getattr(args, 'explosion_abnormal_steps_threshold', 5)
        )
        self.explosion_acceleration_speed_step_window = int(
            getattr(args, 'explosion_acceleration_speed_step_window', 5)
        )
        if self.explosion_linear_speed_threshold <= 0.0:
            raise ValueError('explosion_linear_speed_threshold 必须大于 0')
        if self.explosion_linear_acceleration_threshold <= 0.0:
            raise ValueError('explosion_linear_acceleration_threshold 必须大于 0')
        if self.explosion_abnormal_steps_threshold <= 0:
            raise ValueError('explosion_abnormal_steps_threshold 必须大于 0')
        if self.explosion_acceleration_speed_step_window < 0:
            raise ValueError('explosion_acceleration_speed_step_window 必须大于等于 0')
        print(
            "[动力学崩飞监控] "
            f"三维|v|>{self.explosion_linear_speed_threshold:.3f}m/s 连续"
            f">={self.explosion_abnormal_steps_threshold}步，且存在 "
            f"三维|a|>{self.explosion_linear_acceleration_threshold:.3f}m/s² 的物理步，"
            f"与速度异常区间步距<={self.explosion_acceleration_speed_step_window}"
        )
        # 每个 env 首次触发后锁存到 reset，防止瞬时崩飞在后续物理步恢复后漏报。
        self.dynamics_explosion_envs = {}
        self.dynamics_explosion_peaks = {}
        self.dynamics_explosion_monitor_step = 0
        self.dynamics_explosion_speed_streaks = {}
        self.dynamics_explosion_speed_streak_starts = {}
        self.dynamics_explosion_max_speed_streaks = {}
        self.dynamics_explosion_speed_intervals = {}
        self.dynamics_explosion_acceleration_events = {}
        # 本次动作的物理测量。位移来自 PhysX 物理质心，接触力复用现有 tracker。
        self.action_push_measurements = {}
        self.previous_actions = [None for _ in range(self.num_envs)]
        self.opposite_action_streaks = [0 for _ in range(self.num_envs)]
        self.previous_empty_pushes = [False for _ in range(self.num_envs)]
        self.num_objects_min = args.num_objects_min
        self.num_objects_max = args.num_objects_max

        # 非法接触检测（PhysX GPU 成对接触力；兼容 Isaac Lab 2.1.1）。
        # 监控左右 inner/outer 四指；接触双方由接触视图的刚体行/实体列精确识别。
        # 每个 env 一个 dict: {'illegal': bool, 'first_hit': str, ...}, reset/每步开始清空。
        # 由 _execute_push_batch 里对 scene.contact_tracker 的判定置位；
        # tracker 为 None (关闭/未启用) 时此集合恒空 → 非法接触检查是无害的 no-op。
        self.illegal_contact_envs = {}  # env_idx -> info dict
        # 每步动作的统一接触结果，供训练/评估输出“准备推谁、实际碰到谁”。
        # status: no_contact | legal | illegal | unavailable
        self.action_contact_results = {}  # env_idx -> info dict

        # [点1-调试] 接触事件原始记录 (仅供 inspect_sim 等调试脚本消费, 训练时默认关闭)。
        # env_wrapper 只负责"检测 + 记录原始 prim_path", 不做文件夹序号解析、不打印;
        # 展示逻辑 (prim_path→模型文件夹序号、打印) 全部放在 inspect_sim.py。
        self.contact_debug = False       # inspect_sim 置 True 后才记录事件
        self._contact_events = []        # 每次 _execute_push_batch 开始清空

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

        # [点1] 清空非法接触记录
        self.illegal_contact_envs.clear()
        self.action_contact_results.clear()
        self.action_push_measurements.clear()
        self.dynamics_explosion_envs.clear()
        self.dynamics_explosion_peaks.clear()
        self.dynamics_explosion_monitor_step = 0
        self.dynamics_explosion_speed_streaks.clear()
        self.dynamics_explosion_speed_streak_starts.clear()
        self.dynamics_explosion_max_speed_streaks.clear()
        self.dynamics_explosion_speed_intervals.clear()
        self.dynamics_explosion_acceleration_events.clear()

        # [修复] 清除所有 robot 的 IK 失败状态，确保新回合全部重新开始。
        for robot in self.scene.robots:
            robot.reset_ik_status()

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

        # 本 step 的接触输出从空状态开始；即使所有环境因 IK 被跳过也不会沿用旧结果。
        self.action_contact_results.clear()
        self.action_push_measurements.clear()

        # 更新每个环境的步数（用于动态步数惩罚）
        for env_idx in range(self.num_envs):
            if not self.env_dones[env_idx]:
                self.env_steps[env_idx] += 1

        # [旧空推逻辑停用] 原逻辑在动作前保存深度图，再用图像差分判定空推。
        # 现改为在目标解析完成后直接保存 PhysX 物理质心，不再依赖图像。
        # self._save_previous_masks(spawned_objects)


        # 检查哪些环境IK已失败，直接标记为失败，不执行动作
        ik_failed_envs = []
        for env_idx, robot in enumerate(self.scene.robots):
            if robot.ik_failed:
                ik_failed_envs.append(env_idx)
                # [旧 IK 处理，已停用] 提前写 env_dones 会让奖励层只得到 already_done，
                # 丢失 ik_failed 异常信息。
                # self.env_dones[env_idx] = True
                print(f"⚠️ [Env {env_idx}] IK已失败，跳过动作执行")

        # 只对非IK失败的环境执行推动动作
        active_envs = [i for i in range(self.num_envs) if i not in ik_failed_envs]

        if active_envs:
            # 同步执行推动动作（方法内部会检查env_dones跳过失败环境）
            self._execute_push_batch(actions, spawned_objects)

        # **强制清零黑名单环境的控制**
        self._enforce_blacklist_zero_control()

        # **额外执行几步物理模拟，让物体稳定并逐物理步监控动力学崩飞**
        simulation_invalid_now = bool(self.dynamics_explosion_envs) or any(
            robot.ik_failed for robot in self.scene.robots
        )
        if not simulation_invalid_now:
            for _ in range(25):  # 执行25步物理模拟，约0.25秒
                self.scene.step()
                newly_exploded = self._monitor_dynamics_explosion(
                    spawned_objects, phase='post_action_settle'
                )
                if newly_exploded:
                    break
        # 更新所有物体的数据
        for obj in spawned_objects:
            obj.update(dt=0.01)

        # [旧崩飞检测，已停用] 原逻辑只在动作结束后检查最终位置是否超界 18 cm。
        # exploded_envs = self._check_exploded_objects(spawned_objects)
        # for env_idx in exploded_envs:
        #     if not self.env_dones[env_idx]:
        #         print(f"💥 [Env {env_idx}] 检测到物体崩飞，标记为失败并等待重置")
        #         self.env_dones[env_idx] = True
        exploded_envs = sorted(self.dynamics_explosion_envs)

        # 获取新状态
        next_states = self._get_observations(spawned_objects)

        # 计算奖励。旧奖励流程保留在 _compute_rewards_legacy_disabled 中。
        # rewards, infos = self._compute_rewards_legacy_disabled(spawned_objects, actions)
        rewards, infos = self._compute_rewards(spawned_objects, actions)

        # 动力学崩飞信息已由奖励层完整写入；这里仅作防御性补全。
        for env_idx in exploded_envs:
            infos[env_idx]['is_exploded'] = True
            infos[env_idx]['success'] = False
            infos[env_idx]['failed'] = False
            infos[env_idx]['out_of_bounds'] = False
            infos[env_idx]['simulation_invalid'] = True
            infos[env_idx]['out_reason'] = '物体动力学崩飞'

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

        # [点1-调试] 本次推动批次的接触事件从零开始记录 (仅 contact_debug 时会被填充)。
        if self.contact_debug:
            self._contact_events = []

        push_points = []
        direction_indices = []
        contact_specs = []  # [点1] 每个 active env 的接触判定信息 (与 push_points 平行)
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

                push_point, direction_idx, contact_spec = compute_push_point_from_action(
                    action_idx, env_idx, state, self.scene, env_objects
                )

                push_points.append(push_point)
                direction_indices.append(direction_idx)
                contact_specs.append(contact_spec)
                active_envs.append(env_idx)

            except Exception as e:
                print(f"❌ [Env{env_idx}] 动作{action_idx}计算失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认推点（环境中心）
                push_point = torch.tensor([0.75, 0.0, 0.1], device=self.device)
                push_points.append(push_point)
                direction_indices.append(0)
                contact_specs.append(None)  # 计算失败: 跳过接触判定
                active_envs.append(env_idx)

        # 如果没有active环境，直接返回
        if not active_envs:
            return

        tracker = getattr(self.scene, 'contact_tracker', None)
        if tracker is not None and tracker.is_ready:
            tracker.begin_action(active_envs)

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

            _spec = contact_specs[active_envs.index(env_idx)]
            intended_path = _spec.get('intended_prim_path') if _spec is not None else None
            com_before = self._get_object_com_position(intended_path, spawned_objects)
            tracker_ready = tracker is not None and tracker.is_ready
            self.action_push_measurements[env_idx] = {
                'intended_prim_path': intended_path,
                'intended_model_id': (
                    _spec.get('intended_model_id') if _spec is not None else None
                ),
                'com_before_w': com_before,
                'peak_contact_force_n': 0.0,
                'contact_available': tracker_ready,
            }
            if _spec is None:
                self.action_contact_results[env_idx] = {
                    'status': 'unavailable',
                    'intended_model_id': None,
                    'intended_prim_path': None,
                    'actual_model_id': None,
                    'actual_prim_path': None,
                }
            else:
                self.action_contact_results[env_idx] = {
                    'status': 'no_contact',
                    'intended_model_id': _spec.get('intended_model_id'),
                    'intended_prim_path': _spec.get('intended_prim_path'),
                    'actual_model_id': None,
                    'actual_prim_path': None,
                }
            env_plans[env_idx] = {
                'plan': plan,
                'segment_idx': 0,
                'elapsed_time': 0.0,
                'start_pos': start_pos,
                'start_quat': start_quat,
                'done': False,
                'stable_steps': 0,
                # [点1] 接触判定信息 + 阶段状态。
                'contact_spec': _spec,
                # 推动段(segment==2)"第一次接触已判为合法" → 后续不再检测 (锁定)。
                'push_contact_resolved': False,
                'push_phase_started': False,
                # [旧下降规则保留字段] 过去允许下降末端接触意图物体；当前下降
                # 阶段任意接触均非法，因此该字段正常情况下始终为 None。
                'intended_contact_at_descent_end': None,
            }

            # [点1-调试] push 开始记录"这一下意图推谁"的原始事件 (prim_path, 不解析文件夹/不打印);
            #            inspect_sim 消费时再把 prim_path 翻成模型文件夹序号并打印。
            if self.contact_debug and _spec is not None:
                self._contact_events.append({
                    'type': 'push_start',
                    'env_idx': env_idx,
                    'kind': _spec['kind'],
                    'intended_prim_path': _spec.get('intended_prim_path'),
                    'intended_model_id': _spec.get('intended_model_id'),
                    'direction': int(direction_indices[active_envs.index(env_idx)]),
                })

        # 2. 同步执行循环
        # 轨迹插值、asset 数据更新和 PhysX 必须共享同一个真实步长。
        # Scene.step() 每轮只推进一个物理步。
        dt = float(self.scene.sim.get_physics_dt())
        if dt <= 0.0:
            raise RuntimeError(f"无效的 PhysX 物理步长: {dt}")
        all_finished = False
        max_iterations = max(1, int(np.ceil(5.0 / dt)))  # 全局安全上限：5秒仿真时间
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
                if robot.ik_failed:
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

            # B. 物理步进：每轮严格推进一个 PhysX 步。
            self.scene.step()

            # 每个物理步直接读取 PhysX 刚体真值，锁存异常三维总线速度/加速度。
            newly_exploded = self._monitor_dynamics_explosion(
                spawned_objects, phase='push_execution'
            )
            for env_idx in newly_exploded:
                state = env_plans.get(env_idx)
                if state is not None:
                    state['done'] = True
                    state['elapsed_time'] = 999.0
                self._zero_robot_control(env_idx)

            # 用 PhysX 当前物理步的真实成对接触力按环境、按阶段判定。
            tracker = getattr(self.scene, 'contact_tracker', None)
            if tracker is not None and tracker.is_ready:
                contacts_by_env = tracker.poll_contacts(self.scene.sim.get_physics_dt())
                for env_idx in active_envs:
                    ps = env_plans.get(env_idx)
                    if ps is None or ps['done']:
                        continue
                    contacts = contacts_by_env.get(
                        env_idx, {'active': [], 'new': [], 'ended': []}
                    )
                    # 旁路累计“任一手指-意图物体”的动作内峰值力；不改变原有
                    # 首次碰撞、非法碰撞以及接触阈值状态机。
                    self._accumulate_intended_contact_force(env_idx, contacts)
                    self._judge_contacts(
                        env_idx,
                        ps,
                        contacts,
                    )

            # [关键优化] 物理步进后立即检查IK失败，快速退出
            # 检查是否有任何环境IK失败，如果有则标记all_finished=True跳出主循环
            ik_failed_now = False
            for env_idx in active_envs:
                robot = self.scene.robots[env_idx]
                if robot.ik_failed:
                    ik_failed_now = True
                    # 立即标记为done
                    if env_idx in env_plans and not env_plans[env_idx].get('done', False):
                        env_plans[env_idx]['done'] = True
                        env_plans[env_idx]['elapsed_time'] = 999.0
                        self.ik_failed_blacklist.add(env_idx)
                        self._zero_robot_control(env_idx)

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
                # Robot asset 已由 Scene.step() 用同一 physics_dt 更新。
                state['elapsed_time'] += dt

                plan = state['plan']
                segment = plan[state['segment_idx']]
                target_pos = segment['target_pos']

                # 检查误差 & 时间
                # 重复 move_to 的误差检查逻辑
                offset_vec = torch.tensor([0.0, 0.0, 0.2333], device=self.device)

                # [FailSafe] 优先检查 IK 失败 - 在所有其他检查之前
                # 如果该环境已经失败，跳过该环境
                if robot.ik_failed:
                    # 只在第一次失败时打印
                    if not state.get('ik_printed', False):
                        print(f"❌ [IK FailSafe] Env {env_idx} IK 失败 (Singularity/Error)")
                        state['ik_printed'] = True

                    # robot 已由全局 env_idx 选中；其 articulation 局部实例只有 0。
                    # 统一冻结整个单实例 articulation，不再传全局环境索引。
                    self.ik_failed_blacklist.add(env_idx)
                    self._zero_robot_control(env_idx)

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


    def _judge_contacts(self, env_idx, plan_state, contacts):
        """
        使用 PhysX 成对接触数据形成的 active/new 集合执行阶段状态机。

        segment 0（到推点上方）和 segment 3（抬起）不判定；
        segment 1（下降）持续检查，碰到任何注册物体或桌面都失败；
        segment 2（推动）只检查第一次新接触，合法后永久锁定。
        若同一物理步第一次同时出现合法和非法实体，按用户要求判非法。

        推障碍动作未解析到明确目标时（kind=obstacle 且 intended=None）：
        同样要求下降段完全无接触；推动段第一次接触非目标障碍物即合法，
        第一次接触目标物体或桌面仍失败。同一步同时含目标/桌面时保守判失败。
        """
        spec = plan_state.get('contact_spec')
        if not spec:
            return  # 计算失败的动作: 不判接触

        seg = plan_state.get('segment_idx', 0)
        intended = spec.get('intended_prim_path')
        unknown_obstacle = spec.get('kind') == 'obstacle' and intended is None
        active = self._deduplicate_contact_entities(contacts.get('active', []))
        new_hits = self._deduplicate_contact_entities(contacts.get('new', []))

        if seg == 1:
            # 下降阶段零接触容忍：active 覆盖持续受力接触，new_hits 覆盖
            # 本物理步刚产生（包括随后立即分开）的受力接触。只要有一条就非法。
            descent_hits = self._deduplicate_contact_entities(active + new_hits)
            if descent_hits:
                self._record_illegal(
                    env_idx,
                    plan_state,
                    spec,
                    'descent_contact',
                    descent_hits[0],
                    intended,
                    phase='descent',
                )
            return

        if seg != 2 or plan_state.get('push_contact_resolved'):
            return

        # 推动阶段第一次进入时，先处理下降结束瞬间已存在的合法接触。
        if not plan_state.get('push_phase_started'):
            plan_state['push_phase_started'] = True
            boundary_hit = plan_state.get('intended_contact_at_descent_end')
            if boundary_hit is not None and boundary_hit['entity_key'] == intended:
                self._record_legal_first_contact(env_idx, plan_state, spec, boundary_hit, intended)
                return

        if not new_hits:
            return

        # 推障碍但视觉侧没有选出唯一 intended：下降无接触的前提下，推动时
        # 第一下碰到任意非 target 障碍物即可视为合法。桌面沿用全局非法规则；
        # 若同一物理步同时出现障碍物与 target/table，无法确定亚步顺序，判失败。
        if unknown_obstacle:
            target_hit = next((hit for hit in new_hits if hit.get('role') == 'target'), None)
            table_hit = next((hit for hit in new_hits if hit.get('kind') == 'table'), None)
            illegal_hit = target_hit or table_hit
            if illegal_hit is not None:
                reason = (
                    'push_unknown_obstacle_hit_target'
                    if illegal_hit.get('role') == 'target'
                    else 'push_table'
                )
                self._record_illegal(
                    env_idx, plan_state, spec, reason, illegal_hit, intended, phase='push'
                )
            else:
                self._record_legal_first_contact(
                    env_idx,
                    plan_state,
                    spec,
                    new_hits[0],
                    intended,
                    first_hit='non_target_obstacle',
                )
            return

        intended_hit = next((hit for hit in new_hits if hit['entity_key'] == intended), None)
        illegal_hits = [hit for hit in new_hits if hit['entity_key'] != intended]

        # 同一物理步首次同时出现 intended 和非法实体，也必须判非法。
        if illegal_hits:
            illegal_hit = illegal_hits[0]
            if intended_hit is not None:
                reason = 'push_simultaneous_with_illegal'
            else:
                reason = 'push_table' if illegal_hit['kind'] == 'table' else 'push_wrong_object'
            self._record_illegal(
                env_idx, plan_state, spec, reason, illegal_hit, intended, phase='push'
            )
        elif intended_hit is not None:
            self._record_legal_first_contact(env_idx, plan_state, spec, intended_hit, intended)

    @staticmethod
    def _deduplicate_contact_entities(hits):
        """同一物体可能同时碰多根手指；判定层按实体去重，保留力较大的记录。"""
        result = {}
        for hit in hits or []:
            key = hit.get('entity_key')
            if key is None:
                continue
            if key not in result or hit.get('force', 0.0) > result[key].get('force', 0.0):
                result[key] = hit
        return list(result.values())

    def _record_legal_first_contact(
        self, env_idx, plan_state, spec, hit, intended, first_hit='intended'
    ):
        plan_state['push_contact_resolved'] = True
        self.action_contact_results[env_idx] = {
            'status': 'legal',
            'intended_model_id': spec.get('intended_model_id'),
            'intended_prim_path': intended,
            'actual_model_id': hit.get('model_id'),
            'actual_prim_path': hit.get('root_path'),
        }
        if self.contact_debug:
            self._contact_events.append({
                'type': 'first_contact',
                'env_idx': env_idx,
                'kind': spec['kind'],
                'illegal': False,
                'first_hit': first_hit,
                'phase': 'push',
                'intended_prim_path': intended,
                'intended_model_id': spec.get('intended_model_id'),
                'hit_prim_path': hit['root_path'],
                'hit_model_id': hit.get('model_id'),
                'finger_prim_path': hit.get('finger_path'),
                'actor0': hit.get('actor0'),
                'actor1': hit.get('actor1'),
                'collider0': hit.get('collider0'),
                'collider1': hit.get('collider1'),
                'force': hit.get('force', 0.0),
                'step': hit.get('step'),
            })

    def _record_illegal(self, env_idx, plan_state, spec, first_hit, hit, intended, phase):
        """记录一次非法接触: 置 illegal_contact_envs (功能) + _contact_events (调试), done + 冻结。"""
        hit_path = 'table' if hit.get('kind') == 'table' else hit.get('root_path')
        hit_model_id = 'TABLE' if hit.get('kind') == 'table' else hit.get('model_id')
        self.action_contact_results[env_idx] = {
            'status': 'illegal',
            'intended_model_id': spec.get('intended_model_id'),
            'intended_prim_path': intended,
            'actual_model_id': hit_model_id,
            'actual_prim_path': hit_path,
        }
        if self.contact_debug:
            self._contact_events.append({
                'type': 'first_contact', 'env_idx': env_idx, 'kind': spec['kind'],
                'illegal': True, 'first_hit': first_hit, 'phase': phase,
                'intended_prim_path': intended, 'hit_prim_path': hit_path,
                'intended_model_id': spec.get('intended_model_id'),
                'hit_model_id': hit.get('model_id'),
                'finger_prim_path': hit.get('finger_path'),
                'actor0': hit.get('actor0'), 'actor1': hit.get('actor1'),
                'collider0': hit.get('collider0'), 'collider1': hit.get('collider1'),
                'force': hit.get('force', 0.0), 'step': hit.get('step'),
            })
        # 功能字段: _compute_rewards 只读 'illegal' 与 'first_hit'。
        self.illegal_contact_envs[env_idx] = {
            'illegal': True,
            'first_hit': first_hit,
            'intended_prim_path': intended,
            'intended_model_id': spec.get('intended_model_id'),
            'hit_prim_path': hit_path,
            'hit_model_id': hit.get('model_id'),
            'finger_prim_path': hit.get('finger_path'),
            'actor0': hit.get('actor0'),
            'actor1': hit.get('actor1'),
            'collider0': hit.get('collider0'),
            'collider1': hit.get('collider1'),
            'force': hit.get('force', 0.0),
            'kind': spec['kind'],
        }
        plan_state['done'] = True
        self._zero_robot_control(env_idx)

    def _base_step_info(self, env_idx):
        """把物理接触结果复制到公开 info；所有训练/评估入口统一消费这些字段。"""
        result = self.action_contact_results.get(env_idx, {})
        return {
            'contact_status': result.get('status', 'unavailable'),
            'contact_intended_model_id': result.get('intended_model_id'),
            'contact_intended_prim_path': result.get('intended_prim_path'),
            'contact_actual_model_id': result.get('actual_model_id'),
            'contact_actual_prim_path': result.get('actual_prim_path'),
        }

    def _monitor_dynamics_explosion(self, spawned_objects, phase):
        """检测连续速度异常，并要求附近存在加速度异常物理步。"""
        dt = float(self.scene.sim.get_physics_dt())
        self.dynamics_explosion_monitor_step += 1
        physics_step = self.dynamics_explosion_monitor_step
        newly_exploded = []
        speed_abnormal_by_env = {}
        acceleration_abnormal_by_env = {}
        observed_envs = set()

        for obj in self._iter_spawned_objects(spawned_objects):
            try:
                obj.update(dt=dt)
                env_idx = int(self.scene._get_env_id_from_prim_path(obj.cfg.prim_path))
                observed_envs.add(env_idx)
                obj_name = obj.cfg.prim_path.split('/')[-1]

                linear_velocity = obj.data.root_lin_vel_w[0]
                linear_acceleration = obj.data.body_lin_acc_w[0, 0]
                velocity_finite = bool(torch.isfinite(linear_velocity).all())
                acceleration_finite = bool(torch.isfinite(linear_acceleration).all())
                speed = (
                    float(torch.linalg.vector_norm(linear_velocity).item())
                    if velocity_finite else float('inf')
                )
                acceleration = (
                    float(torch.linalg.vector_norm(linear_acceleration).item())
                    if acceleration_finite else float('inf')
                )

                peaks = self.dynamics_explosion_peaks.setdefault(env_idx, {
                    'peak_linear_speed_m_s': 0.0,
                    'peak_linear_acceleration_m_s2': 0.0,
                })
                peaks['peak_linear_speed_m_s'] = max(
                    float(peaks['peak_linear_speed_m_s']), speed
                )
                peaks['peak_linear_acceleration_m_s2'] = max(
                    float(peaks['peak_linear_acceleration_m_s2']), acceleration
                )

                speed_exceeded = speed > self.explosion_linear_speed_threshold
                acceleration_exceeded = (
                    acceleration > self.explosion_linear_acceleration_threshold
                )
                base_record = {
                    'object_name': obj_name,
                    'object_prim_path': obj.cfg.prim_path,
                    'phase': phase,
                    'linear_speed_m_s': speed,
                    'linear_acceleration_m_s2': acceleration,
                }
                if speed_exceeded:
                    previous = speed_abnormal_by_env.get(env_idx)
                    if previous is None or speed > previous['linear_speed_m_s']:
                        speed_abnormal_by_env[env_idx] = dict(base_record)
                if acceleration_exceeded:
                    previous = acceleration_abnormal_by_env.get(env_idx)
                    if (
                        previous is None
                        or acceleration > previous['linear_acceleration_m_s2']
                    ):
                        acceleration_abnormal_by_env[env_idx] = dict(base_record)
            except Exception as exc:
                # 监控器不可静默失效；保留一次可定位的警告，但不把读取异常伪装成崩飞。
                print(f"⚠️ [崩飞监控] 读取物体动力学失败: {exc}")

        # 每个 env 在一次调用中至多更新一次状态，因此“步数”是真实物理步而非物体数。
        for env_idx in observed_envs:
            speed_record = speed_abnormal_by_env.get(env_idx)
            if speed_record is None:
                self.dynamics_explosion_speed_streaks[env_idx] = 0
                self.dynamics_explosion_speed_streak_starts.pop(env_idx, None)
            else:
                previous_streak = self.dynamics_explosion_speed_streaks.get(env_idx, 0)
                speed_streak = previous_streak + 1
                self.dynamics_explosion_speed_streaks[env_idx] = speed_streak
                if previous_streak == 0:
                    self.dynamics_explosion_speed_streak_starts[env_idx] = physics_step
                streak_start = self.dynamics_explosion_speed_streak_starts[env_idx]
                self.dynamics_explosion_max_speed_streaks[env_idx] = max(
                    self.dynamics_explosion_max_speed_streaks.get(env_idx, 0),
                    speed_streak,
                )

                if speed_streak == self.explosion_abnormal_steps_threshold:
                    self.dynamics_explosion_speed_intervals.setdefault(env_idx, []).append({
                        'start_step': streak_start,
                        'end_step': physics_step,
                        'speed_record': speed_record,
                    })
                elif speed_streak > self.explosion_abnormal_steps_threshold:
                    interval = self.dynamics_explosion_speed_intervals[env_idx][-1]
                    interval['end_step'] = physics_step
                    if (
                        speed_record['linear_speed_m_s']
                        > interval['speed_record']['linear_speed_m_s']
                    ):
                        interval['speed_record'] = speed_record

            acceleration_record = acceleration_abnormal_by_env.get(env_idx)
            if acceleration_record is not None:
                self.dynamics_explosion_acceleration_events.setdefault(env_idx, []).append({
                    'step': physics_step,
                    'record': acceleration_record,
                })

        for env_idx in observed_envs:
            if env_idx in self.dynamics_explosion_envs:
                continue
            intervals = self.dynamics_explosion_speed_intervals.get(env_idx, [])
            acceleration_events = self.dynamics_explosion_acceleration_events.get(
                env_idx, []
            )
            best_match = None
            for interval in intervals:
                for acceleration_event in acceleration_events:
                    acceleration_step = acceleration_event['step']
                    if acceleration_step < interval['start_step']:
                        step_distance = interval['start_step'] - acceleration_step
                    elif acceleration_step > interval['end_step']:
                        step_distance = acceleration_step - interval['end_step']
                    else:
                        step_distance = 0
                    if step_distance <= self.explosion_acceleration_speed_step_window:
                        if best_match is None or step_distance < best_match['step_distance']:
                            best_match = {
                                'interval': interval,
                                'acceleration_event': acceleration_event,
                                'step_distance': step_distance,
                            }

            if best_match is None:
                continue

            interval = best_match['interval']
            speed_record = interval['speed_record']
            acceleration_event = best_match['acceleration_event']
            acceleration_record = acceleration_event['record']
            speed_streak_length = interval['end_step'] - interval['start_step'] + 1
            record = {
                'object_name': speed_record['object_name'],
                'object_prim_path': speed_record['object_prim_path'],
                'acceleration_object_name': acceleration_record['object_name'],
                'acceleration_object_prim_path': acceleration_record['object_prim_path'],
                'phase': phase,
                'reason': 'consecutive_speed_with_nearby_acceleration',
                'linear_speed_m_s': speed_record['linear_speed_m_s'],
                'linear_acceleration_m_s2': acceleration_record[
                    'linear_acceleration_m_s2'
                ],
                'speed_streak_start_step': interval['start_step'],
                'speed_streak_end_step': interval['end_step'],
                'speed_streak_length': speed_streak_length,
                'speed_streak_threshold': self.explosion_abnormal_steps_threshold,
                'acceleration_abnormal_step': acceleration_event['step'],
                'acceleration_speed_step_distance': best_match['step_distance'],
                'acceleration_speed_step_window': (
                    self.explosion_acceleration_speed_step_window
                ),
                'monitor_step': physics_step,
            }
            self.dynamics_explosion_envs[env_idx] = record
            newly_exploded.append(env_idx)
            print(
                f"💥 [Env {env_idx}] 物体 {record['object_name']} 动力学崩飞: "
                f"|v|={record['linear_speed_m_s']:.3f}/"
                f"{self.explosion_linear_speed_threshold:.3f}m/s, "
                f"连续速度异常步={speed_streak_length}/"
                f"{self.explosion_abnormal_steps_threshold} "
                f"(step {interval['start_step']}-{interval['end_step']}); "
                f"|a|={record['linear_acceleration_m_s2']:.3f}/"
                f"{self.explosion_linear_acceleration_threshold:.3f}m/s² "
                f"(step {acceleration_event['step']}), 步距={best_match['step_distance']}/"
                f"{self.explosion_acceleration_speed_step_window}, phase={phase}"
            )

        return newly_exploded

    @staticmethod
    def _iter_spawned_objects(spawned_objects):
        """兼容扁平 List[Obj] 和 List[List[Obj]] 两种物体容器。"""
        for item in spawned_objects or []:
            if isinstance(item, list):
                yield from item
            else:
                yield item

    def _get_object_com_position(self, prim_path, spawned_objects=None):
        """直接从 PhysX tensor 读取指定模型的世界坐标物理质心。"""
        if not prim_path:
            return None
        objects = self.spawned_objects if spawned_objects is None else spawned_objects
        normalized_path = str(prim_path).rstrip('/')
        for obj in self._iter_spawned_objects(objects):
            try:
                if obj.cfg.prim_path.rstrip('/') != normalized_path:
                    continue
                # Isaac Lab 2.1.1 已提供 root_com_pos_w；它由 PhysX 的
                # actor pose 与 get_coms() 组合得到，不是模型 USD 原点。
                com = obj.data.root_com_pos_w[0]
                if not bool(torch.isfinite(com).all()):
                    return None
                return com.detach().clone()
            except Exception:
                return None
        return None

    def _accumulate_intended_contact_force(self, env_idx, contacts):
        """累计当前动作中任一受监控手指对意图物体的最大接触力。"""
        measurement = self.action_push_measurements.get(env_idx)
        if measurement is None:
            return
        intended = measurement.get('intended_prim_path')
        if not intended:
            return

        peak = float(measurement.get('peak_contact_force_n', 0.0))
        for hit in list(contacts.get('active', [])) + list(contacts.get('new', [])):
            if hit.get('entity_key') != intended:
                continue
            force = float(hit.get('force', 0.0))
            if np.isfinite(force):
                peak = max(peak, force)
        measurement['peak_contact_force_n'] = peak

    @staticmethod
    def _evaluate_push_effectiveness(
        displacement_m,
        peak_force_n,
        displacement_threshold_m=0.01,
        force_threshold_n=1.0,
        target_resolved=True,
        com_available=True,
        contact_available=True,
    ):
        """返回 ``(is_empty, displacement_ok, force_ok, reason)``。"""
        if not target_resolved:
            return True, False, False, 'unresolved_target'
        if not com_available:
            return True, False, False, 'com_unavailable'
        if not contact_available:
            return True, displacement_m > displacement_threshold_m, False, 'contact_unavailable'

        displacement_ok = displacement_m > displacement_threshold_m
        force_ok = peak_force_n > force_threshold_n
        if displacement_ok and force_ok:
            reason = 'valid'
        elif not displacement_ok and not force_ok:
            reason = 'displacement_and_force_below_threshold'
        elif not displacement_ok:
            reason = 'displacement_below_threshold'
        else:
            reason = 'force_below_threshold'
        return not (displacement_ok and force_ok), displacement_ok, force_ok, reason

    def _check_empty_push(self, env_idx):
        """用“物理质心 XY 位移 AND 意图物体峰值接触力”判定空推。"""
        measurement = self.action_push_measurements.get(env_idx, {})
        intended_path = measurement.get('intended_prim_path')
        com_before = measurement.get('com_before_w')
        com_after = self._get_object_com_position(intended_path)
        target_resolved = bool(intended_path)
        com_available = com_before is not None and com_after is not None

        displacement_m = 0.0
        if com_available:
            displacement_m = float(torch.linalg.vector_norm(com_after[:2] - com_before[:2]).item())
        peak_force_n = float(measurement.get('peak_contact_force_n', 0.0))
        contact_available = bool(measurement.get('contact_available', False))

        is_empty, displacement_ok, force_ok, reason = self._evaluate_push_effectiveness(
            displacement_m=displacement_m,
            peak_force_n=peak_force_n,
            displacement_threshold_m=self.empty_push_displacement_threshold,
            force_threshold_n=self.empty_push_force_threshold,
            target_resolved=target_resolved,
            com_available=com_available,
            contact_available=contact_available,
        )
        metrics = {
            'target_prim_path': intended_path,
            'target_model_id': measurement.get('intended_model_id'),
            'center_type': 'physx_center_of_mass',
            'axes': 'xy',
            'com_before_w': com_before.detach().cpu().tolist() if com_before is not None else None,
            'com_after_w': com_after.detach().cpu().tolist() if com_after is not None else None,
            'displacement_m': displacement_m,
            'displacement_threshold_m': self.empty_push_displacement_threshold,
            'peak_contact_force_n': peak_force_n,
            'force_threshold_n': self.empty_push_force_threshold,
            'target_resolved': target_resolved,
            'com_available': com_available,
            'contact_available': contact_available,
            'displacement_ok': displacement_ok,
            'force_ok': force_ok,
            'reason': reason,
        }
        return is_empty, metrics


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

    '''
    def _compute_rewards_legacy_disabled(self, spawned_objects, actions):
        """
        [旧奖励流程，已停用]: 原“出界→成功→空推”流程完整保留，不再被 step 调用。
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
            info = self._base_step_info(env_idx)
            info['opposite_action'] = False
            info['opposite_action_streak'] = 0

            # 0. 步数惩罚（固定-1.5）
            current_step = self.env_steps[env_idx].item()
            step_penalty = -1.5  # 固定-1.5
            reward += step_penalty
            reward_breakdown['步数惩罚'] = step_penalty

            # [FailSafe] 优先检查 IK 失败 (最高优先级: System Failure)
            robot = self.scene.robots[env_idx]
            if robot.ik_failed:
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

            # [点1] 非法接触检查 (优先级紧跟 IK, 高于出界/成功)
            # "推X的第一次接触碰到的不是X" → 非法即失败: -10 + done + return。
            # 关键: 必须在成功判定之前 return, 切断"歪打正着的成功"错误标签。
            # illegal_contact_envs 由 _execute_push_batch 的 PhysX 接触追踪逻辑置位;
            # 未接入前恒空, 此分支为无害 no-op。
            illegal_info = self.illegal_contact_envs.get(env_idx)
            if illegal_info and illegal_info.get('illegal', False):
                illegal_penalty = -10.0
                reward += illegal_penalty
                reward_breakdown['非法接触'] = illegal_penalty
                info['failed'] = True
                info['success'] = False
                info['out_of_bounds'] = False
                info['illegal_contact'] = True
                info['illegal_first_hit'] = illegal_info.get('first_hit', 'unknown')
                # 保留 PhysX 接触双方刚体及三位模型 ID，供训练/评估日志精确追溯。
                info['intended_model_id'] = illegal_info.get('intended_model_id')
                info['intended_prim_path'] = illegal_info.get('intended_prim_path')
                info['illegal_hit_model_id'] = illegal_info.get('hit_model_id')
                info['illegal_hit_prim_path'] = illegal_info.get('hit_prim_path')
                info['illegal_finger_prim_path'] = illegal_info.get('finger_prim_path')
                info['illegal_actor0'] = illegal_info.get('actor0')
                info['illegal_actor1'] = illegal_info.get('actor1')
                info['illegal_collider0'] = illegal_info.get('collider0')
                info['illegal_collider1'] = illegal_info.get('collider1')
                info['illegal_contact_force'] = illegal_info.get('force', 0.0)

                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
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
                # [点2] 反向动作惩罚暂时注释掉 (保留代码, 暂不删除)
                # opposite_penalty = self._get_opposite_action_penalty(
                #     env_idx, actions[env_idx], is_empty=False
                # )
                # if opposite_penalty != 0.0:
                #     reward += opposite_penalty
                #     reward_breakdown['反向动作惩罚'] = opposite_penalty
                #     info['opposite_action'] = True
                #     info['opposite_action_streak'] = self.opposite_action_streaks[env_idx]
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
            # [旧深度图空推判定保留，已停用]
            # is_empty, empty_value, empty_total, empty_ratio, empty_threshold = \
            #     self._check_empty_push_depth_legacy(env_idx)
            # info['empty_push'] = is_empty
            # info['empty_metrics'] = {
            #     'change_value': empty_value,
            #     'total_pixels': empty_total,
            #     'change_ratio': empty_ratio,
            #     'threshold': empty_threshold
            # }
            is_empty, empty_metrics = self._check_empty_push(env_idx)
            info['empty_push'] = is_empty
            info['empty_metrics'] = empty_metrics

            if is_empty:
                # 空推惩罚
                empty_penalty = -5.0
                reward += empty_penalty
                reward_breakdown['空推惩罚'] = empty_penalty

            # 4. [点2] 反向动作惩罚暂时注释掉 (保留代码, 暂不删除)
            # opposite_penalty = self._get_opposite_action_penalty(
            #     env_idx, actions[env_idx], is_empty=is_empty
            # )
            # if opposite_penalty != 0.0:
            #     reward += opposite_penalty
            #     reward_breakdown['反向动作惩罚'] = opposite_penalty
            #     info['opposite_action'] = True
            #     info['opposite_action_streak'] = self.opposite_action_streaks[env_idx]

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
    '''
    def _compute_rewards(self, spawned_objects, actions):
        """按严格短路顺序计算奖励：碰撞→空推→出界→成功→最大步数。"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        infos = []

        for env_idx in range(self.num_envs):
            if self.env_dones[env_idx]:
                infos.append({
                    'out_of_bounds': False,
                    'failed': False,
                    'success': False,
                    'empty_push': False,
                    'reward_breakdown': {},
                    'total_reward': 0.0,
                    'already_done': True,
                })
                continue

            # 每次实际执行的动作固定产生 -1 原始步数惩罚；最终统一除以 10。
            reward = -1.0
            reward_breakdown = {'步数惩罚': -1.0}
            current_step = int(self.env_steps[env_idx].item())
            info = self._base_step_info(env_idx)
            info.update({
                'failed': False,
                'success': False,
                'empty_push': False,
                'out_of_bounds': False,
                'is_exploded': False,
                'simulation_invalid': False,
                'opposite_action': False,
                'opposite_action_streak': 0,
            })
            dynamics_peaks = self.dynamics_explosion_peaks.get(env_idx, {})
            info['dynamics_metrics'] = {
                'peak_linear_speed_m_s': dynamics_peaks.get(
                    'peak_linear_speed_m_s', 0.0
                ),
                'linear_speed_threshold_m_s': self.explosion_linear_speed_threshold,
                'peak_linear_acceleration_m_s2': dynamics_peaks.get(
                    'peak_linear_acceleration_m_s2', 0.0
                ),
                'linear_acceleration_threshold_m_s2': (
                    self.explosion_linear_acceleration_threshold
                ),
                'monitor_step': self.dynamics_explosion_monitor_step,
                'current_speed_streak': self.dynamics_explosion_speed_streaks.get(
                    env_idx, 0
                ),
                'max_speed_streak': self.dynamics_explosion_max_speed_streaks.get(
                    env_idx, 0
                ),
                'speed_streak_threshold': self.explosion_abnormal_steps_threshold,
                'latest_acceleration_abnormal_step': (
                    self.dynamics_explosion_acceleration_events.get(
                        env_idx, [{}]
                    )[-1].get('step')
                    if self.dynamics_explosion_acceleration_events.get(env_idx)
                    else None
                ),
                'acceleration_speed_step_window': (
                    self.explosion_acceleration_speed_step_window
                ),
            }

            robot = self.scene.robots[env_idx]
            ik_failed = robot.ik_failed
            explosion = self.dynamics_explosion_envs.get(env_idx)

            # IK 与动力学崩飞都是无效仿真尝试：终止并由外层丢弃整段轨迹、最多重试5次。
            if ik_failed or explosion is not None:
                info['ik_failed'] = bool(ik_failed)
                info['is_exploded'] = explosion is not None
                info['simulation_invalid'] = True
                info['failed'] = False
                info['success'] = False
                info['out_of_bounds'] = False
                if explosion is not None:
                    peaks = self.dynamics_explosion_peaks.get(env_idx, {})
                    info.update({
                        'out_reason': '物体动力学崩飞',
                        'explosion_reason': explosion.get('reason'),
                        'explosion_phase': explosion.get('phase'),
                        'exploded_object': explosion.get('object_name'),
                        'exploded_object_prim_path': explosion.get('object_prim_path'),
                        'explosion_acceleration_object': explosion.get(
                            'acceleration_object_name'
                        ),
                        'explosion_acceleration_object_prim_path': explosion.get(
                            'acceleration_object_prim_path'
                        ),
                        'explosion_linear_speed_m_s': explosion.get('linear_speed_m_s'),
                        'explosion_linear_acceleration_m_s2': explosion.get(
                            'linear_acceleration_m_s2'
                        ),
                        'peak_linear_speed_m_s': peaks.get('peak_linear_speed_m_s', 0.0),
                        'peak_linear_acceleration_m_s2': peaks.get(
                            'peak_linear_acceleration_m_s2', 0.0
                        ),
                        'explosion_linear_speed_threshold_m_s': (
                            self.explosion_linear_speed_threshold
                        ),
                        'explosion_linear_acceleration_threshold_m_s2': (
                            self.explosion_linear_acceleration_threshold
                        ),
                        'explosion_speed_streak_start_step': explosion.get(
                            'speed_streak_start_step'
                        ),
                        'explosion_speed_streak_end_step': explosion.get(
                            'speed_streak_end_step'
                        ),
                        'explosion_speed_streak_length': explosion.get(
                            'speed_streak_length', 0
                        ),
                        'explosion_acceleration_abnormal_step': explosion.get(
                            'acceleration_abnormal_step'
                        ),
                        'explosion_acceleration_speed_step_distance': explosion.get(
                            'acceleration_speed_step_distance'
                        ),
                        'explosion_acceleration_speed_step_window': (
                            self.explosion_acceleration_speed_step_window
                        ),
                    })
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_empty_pushes[env_idx] = False
                continue

            # 1. 非法碰撞：最高普通失败优先级；与空推/出界绝不叠加。
            illegal_info = self.illegal_contact_envs.get(env_idx)
            if illegal_info and illegal_info.get('illegal', False):
                reward += -10.0
                reward_breakdown['非法接触'] = -10.0
                info.update({
                    'failed': True,
                    'illegal_contact': True,
                    'illegal_first_hit': illegal_info.get('first_hit', 'unknown'),
                    'intended_model_id': illegal_info.get('intended_model_id'),
                    'intended_prim_path': illegal_info.get('intended_prim_path'),
                    'illegal_hit_model_id': illegal_info.get('hit_model_id'),
                    'illegal_hit_prim_path': illegal_info.get('hit_prim_path'),
                    'illegal_finger_prim_path': illegal_info.get('finger_prim_path'),
                    'illegal_actor0': illegal_info.get('actor0'),
                    'illegal_actor1': illegal_info.get('actor1'),
                    'illegal_collider0': illegal_info.get('collider0'),
                    'illegal_collider1': illegal_info.get('collider1'),
                    'illegal_contact_force': illegal_info.get('force', 0.0),
                })
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = False
                continue

            # 2. 空推：在出界/成功之前判断；失败后立即短路，惩罚不与其他失败叠加。
            is_empty, empty_metrics = self._check_empty_push(env_idx)
            info['empty_push'] = is_empty
            info['empty_metrics'] = empty_metrics
            if is_empty:
                # [旧空推惩罚，已停用]
                # empty_penalty = -5.0
                empty_penalty = -10.0
                reward += empty_penalty
                reward_breakdown['空推惩罚'] = empty_penalty
                info['failed'] = True
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = True
                continue

            # 3. 普通出界；40 个追加物理步也持续监控动力学崩飞。
            is_out, out_reason, is_exploded = self._check_out_of_bounds(
                env_idx, spawned_objects
            )
            info['out_of_bounds'] = is_out
            info['out_reason'] = out_reason
            if is_exploded:
                explosion = self.dynamics_explosion_envs.get(env_idx, {})
                peaks = self.dynamics_explosion_peaks.get(env_idx, {})
                info.update({
                    'is_exploded': True,
                    'simulation_invalid': True,
                    'failed': False,
                    'success': False,
                    'out_of_bounds': False,
                    'out_reason': '物体动力学崩飞',
                    'explosion_reason': explosion.get('reason'),
                    'explosion_phase': explosion.get('phase'),
                    'exploded_object': explosion.get('object_name'),
                    'exploded_object_prim_path': explosion.get('object_prim_path'),
                    'explosion_acceleration_object': explosion.get(
                        'acceleration_object_name'
                    ),
                    'explosion_acceleration_object_prim_path': explosion.get(
                        'acceleration_object_prim_path'
                    ),
                    'explosion_linear_speed_m_s': explosion.get('linear_speed_m_s'),
                    'explosion_linear_acceleration_m_s2': explosion.get(
                        'linear_acceleration_m_s2'
                    ),
                    'peak_linear_speed_m_s': peaks.get('peak_linear_speed_m_s', 0.0),
                    'peak_linear_acceleration_m_s2': peaks.get(
                        'peak_linear_acceleration_m_s2', 0.0
                    ),
                    'explosion_speed_streak_start_step': explosion.get(
                        'speed_streak_start_step'
                    ),
                    'explosion_speed_streak_end_step': explosion.get(
                        'speed_streak_end_step'
                    ),
                    'explosion_speed_streak_length': explosion.get(
                        'speed_streak_length', 0
                    ),
                    'explosion_acceleration_abnormal_step': explosion.get(
                        'acceleration_abnormal_step'
                    ),
                    'explosion_acceleration_speed_step_distance': explosion.get(
                        'acceleration_speed_step_distance'
                    ),
                    'explosion_acceleration_speed_step_window': (
                        self.explosion_acceleration_speed_step_window
                    ),
                })
                info['dynamics_metrics']['monitor_step'] = (
                    self.dynamics_explosion_monitor_step
                )
                info['dynamics_metrics']['current_speed_streak'] = (
                    self.dynamics_explosion_speed_streaks.get(env_idx, 0)
                )
                info['dynamics_metrics']['max_speed_streak'] = (
                    self.dynamics_explosion_max_speed_streaks.get(env_idx, 0)
                )
                # 清除普通出界候选，不附加 -10；外层会丢弃整段轨迹。
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_empty_pushes[env_idx] = False
                continue
            if is_out:
                reward += -10.0
                reward_breakdown['出界惩罚'] = -10.0
                info['failed'] = True
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = False
                continue

            # 4. 成功：事件奖励 +11，与当步 -1 相加后为 +10，归一化后为 +1。
            success, separation_sim, separation_threshold = \
                self._check_successful_separation(env_idx, spawned_objects)  # type: ignore
            info['success'] = success
            info['separation_metrics'] = {
                'similarity': separation_sim,
                'threshold': separation_threshold,
            }
            if success:
                # [旧成功奖励，已停用]
                # success_reward = 10.0
                success_reward = 11.0
                reward += success_reward
                reward_breakdown['成功分离'] = success_reward
                info['reward_breakdown'] = reward_breakdown
                info['total_reward'] = reward
                rewards[env_idx] = reward
                infos.append(info)
                self.previous_actions[env_idx] = actions[env_idx]
                self.previous_empty_pushes[env_idx] = False
                continue

            # 5. 未成功且达到最大步数：唯一终止失败，不与前面的失败分支叠加。
            if current_step >= self.max_steps_per_episode:
                reward += -10.0
                reward_breakdown['超过最大步数'] = -10.0
                info['failed'] = True
                info['max_steps_exceeded'] = True

            info['reward_breakdown'] = reward_breakdown
            info['total_reward'] = reward
            rewards[env_idx] = reward
            infos.append(info)
            self.previous_actions[env_idx] = actions[env_idx]
            self.previous_empty_pushes[env_idx] = False

        # 所有原始奖励统一缩放：中间步 -0.1，成功 +1.0，普通失败 -1.1。
        return rewards / 10.0, infos

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
    '''
    def _check_empty_push_depth_legacy(self, env_idx, change_threshold=450):
        """
        [旧逻辑，保留但不再调用]: 用推动前后深度图变化像素数检查空推。
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
        depth_diff[depth_diff > 0.2] = 0      # 过滤过大值（>0.3m，可能是噪声）
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
    '''
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
    '''
    def _check_exploded_objects(self, spawned_objects):
        """
        [旧崩飞检测，已停用]: 检查物体最终位置是否飞出工作空间范围超过18cm。
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
        [旧崩飞检测，已停用]: 出界后额外推进物理仿真，再检查最终位置。
        """
        for _ in range(steps):
            self.scene.step()

        for obj in spawned_objects:
            try:
                obj.update(dt=0.01)
            except Exception:
                pass

    def _check_delayed_explosion_after_out_of_bounds(self, env_idx, spawned_objects):
        """[旧崩飞检测，已停用] 40 步后按最终越界距离判断。"""
        self._settle_after_out_of_bounds(spawned_objects, steps=40)
        exploded_envs = self._check_exploded_objects(spawned_objects)
        return env_idx in exploded_envs
    '''
    def _monitor_after_out_of_bounds(self, env_idx, spawned_objects, steps=40):
        """出界后继续逐物理步监控动力学崩飞，返回指定 env 是否已锁存。"""
        for _ in range(steps):
            self.scene.step()
            newly_exploded = self._monitor_dynamics_explosion(
                spawned_objects, phase='out_of_bounds_settle'
            )
            if env_idx in self.dynamics_explosion_envs:
                break
            # 其他 env 崩飞不影响当前 env 的 OOB 复查，但状态已经分别锁存。
            if newly_exploded:
                continue
        return env_idx in self.dynamics_explosion_envs

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
                # [旧崩飞检测，已停用]
                # is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                is_exploded = self._monitor_after_out_of_bounds(env_idx, spawned_objects)
                if env_idx >= 2:
                    print(f"  [DEBUG Env {env_idx}] ✗ OOB X: Obj={obj_name}, LocalX={local_x:.2f}, Exceed={x_exceed:.2f}m {'[崩飞]' if is_exploded else ''}")
                return True, f"物体 {obj_name} X轴出界", is_exploded
            if not y_in:
                # 计算超出边界的距离
                y_exceed = max(workspace_limits[0, 1].item() - local_y, local_y - workspace_limits[1, 1].item())
                # [旧崩飞检测，已停用]
                # is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                is_exploded = self._monitor_after_out_of_bounds(env_idx, spawned_objects)
                if env_idx >= 2:
                    print(f"  [DEBUG Env {env_idx}] ✗ OOB Y: Obj={obj_name}, LocalY={local_y:.2f}, Exceed={y_exceed:.2f}m {'[崩飞]' if is_exploded else ''}")
                return True, f"物体 {obj_name} Y轴出界", is_exploded
            # Z轴检查（是否掉落到桌面以下）
            if not z_in:
                # [旧崩飞检测，已停用]
                # is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
                is_exploded = self._monitor_after_out_of_bounds(env_idx, spawned_objects)
                return True, f"物体 {obj_name} 掉落", is_exploded

        # 检测2：深度图掩膜检测（精确边界）
        state = self.scene.states[env_idx]
        out_of_bounds, check_info = state.check_out_of_bounds(verbose=False)

        out_reason = check_info.get("reason", "unknown") if out_of_bounds else "none"

        if out_of_bounds:
            # [旧崩飞检测，已停用]
            # is_exploded = self._check_delayed_explosion_after_out_of_bounds(env_idx, spawned_objects)
            is_exploded = self._monitor_after_out_of_bounds(env_idx, spawned_objects)
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
                or info.get('simulation_invalid', False)
                # [旧全局步数判断，已停用] current_step 未按 env 更新；最大步数由
                # _compute_rewards 使用 env_steps 设置 failed=True。
                # or self.current_step >= self.max_steps_per_episode
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
            # [修复] 该版本 API 的关键字是 position/velocity, 不是 joint_pos/joint_vel
            if hasattr(articulation, 'write_joint_state_to_sim'):
                articulation.write_joint_state_to_sim(
                    position=current_joint_pos,
                    velocity=zero_vel
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
