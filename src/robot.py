import torch
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_inv

class PushActionPrimitive:
    def __init__(self, device="cuda:0", push_length=0.1, lift_height=0.1):
        self.device = device
        self.push_length = push_length
        self.lift_height = lift_height

    def get_waypoints(self, push_point, theta_index, base_quat=None):
        """
        [功能]: 根据推点和方向索引生成动作路径点
        [输入]: push_point (Tensor N*3), theta_index (Tensor N), base_quat (Tensor N*4)
        [输出]: waypoints (List[Tuple[Tensor, Tensor]])
        """
        # 确保输入是 Tensor
        if not isinstance(push_point, torch.Tensor):
            push_point = torch.tensor(push_point, device=self.device, dtype=torch.float32)
        if not isinstance(theta_index, torch.Tensor):
            theta_index = torch.tensor(theta_index, device=self.device, dtype=torch.float32)
            
        # 确保维度正确 (处理单个输入的情况)
        if push_point.dim() == 1:
            push_point = push_point.unsqueeze(0)
        if theta_index.dim() == 0:
            theta_index = theta_index.unsqueeze(0)

        # 计算角度 (弧度)
        # [输入]: theta_index (Tensor [N]): 推动角度索引 [0-7] (对应 C4 的 8 个方向)
        # 角度 = theta_index * 45° = theta_index * (π / 4)
        # push_center (Tensor        # 角度计算改为 90 度步长 (C4: 4个方向)
        # theta_index: 0 → 0°, 1 → 90°, 2 → 180°, 3 → 270°
        pi = 3.1415926535
        angles = theta_index * (pi / 2.0)  # 90度 = π/2
        
        # 归一化到 [-pi, pi]
        angles = torch.remainder(angles + pi, 2 * pi) - pi
        
        direction = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1)
        
        target_quats = None
        if base_quat is not None:
             half_angles = angles / 2.0
             z_rot_quat = torch.stack([torch.cos(half_angles), torch.zeros_like(angles), torch.zeros_like(angles), torch.sin(half_angles)], dim=-1)
             target_quats = quat_mul(z_rot_quat, base_quat)

        # 返回 (Pos, Quat) 元组列表
        # [Fix] 为路径点2-3设置None，确保推动过程中姿态保持固定
        offsets = [
            # 0: 预推点上方 - 旋转到目标推动方向
            (push_point.clone().add_(torch.tensor([0,0,self.lift_height], device=self.device)), target_quats),
            # 1: 推点 - 保持当前姿态（不再旋转）
            (push_point.clone(), None),
            # 2: 推动终点 - 保持当前姿态（不再旋转）
            (push_point + direction * self.push_length, None),
            # 3: 上提点 - 保持当前姿态（不再旋转）
            (push_point + direction * self.push_length + torch.tensor([0,0,self.lift_height], device=self.device), None)
        ]
        return offsets

class Robot:
    def __init__(self, prim_path, device="cuda:0"):
        self.prim_path = prim_path
        self.device = device
        
        self.cfg = ArticulationCfg(
            prim_path=prim_path,
            actuators={
                "ur10": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    stiffness=30000.0,  # [Standard] 高刚度以确保精确跟踪
                    damping=1000.0,      # [Standard] 标准阻尼 (Ratio 100:1)
                ),
                "2f140": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "finger_joint", 
                        "right_outer_knuckle_joint", 
                        "left_outer_finger_joint", 
                        "right_outer_finger_joint", 
                        "left_inner_finger_joint", 
                        "right_inner_finger_joint",
                        "left_inner_finger_pad_joint",   # 正确的关节名
                        "right_inner_finger_pad_joint",  # 正确的关节名
                    ],
                    stiffness=40000.0,
                    damping=400.0,
                ),
                "prismatic": ImplicitActuatorCfg(
                    joint_names_expr=["PrismaticJoint"],
                    stiffness=30000.0,
                    damping=1000.0,
                ),
            },
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "PrismaticJoint": 0.0,
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.57,
                    "elbow_joint": 1.57,
                    "wrist_1_joint": -1.57,
                    "wrist_2_joint": -1.57,
                    "wrist_3_joint": 0.0,
                }
            ),
        )
        
        self.articulation = Articulation(self.cfg)
        self.articulation._device = self.device

        self.ur10_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_joint_names = ["finger_joint", "right_outer_knuckle_joint", "left_outer_finger_joint", "right_outer_finger_joint", "left_inner_finger_joint", "right_inner_finger_joint", "left_inner_finger_pad_joint", "right_inner_finger_pad_joint"]

        self.ur10_dof_indices = []
        self.gripper_dof_indices = []
        
        self.ik_controller = None
        self.ee_body_name = None
        self.ee_body_idx = None
        self.fixed_ee_orientation = None

        self.workspace_limits = torch.tensor([
            [0.375, -0.375, 0.02], # Min
            [1.125, 0.375, 0.4]    # Max
        ], device=self.device)
        
        self.push_primitive = PushActionPrimitive(device=self.device, lift_height=0.1)
        
        # [FailSafe] Track environments where IK failed
        self.ik_fail_indices = set()
        
    def reset_ik_status(self):
        """
        重置 IK 失败状态。在每一轮 step 开始时调用。
        """
        self.ik_fail_indices.clear()

    def initialize(self):
        """
        初始化关节索引。必须在仿真重置之后调用。
        """
        self.articulation._initialize_impl() if not self.articulation.is_initialized else None
        
        def get_indices(names):
            indices = []
            for name in names:
                for i, jn in enumerate(self.articulation.joint_names):
                    if name in jn:
                        indices.append(i)
                        break
            return indices

        self.ur10_dof_indices = get_indices(self.ur10_joint_names)
        self.gripper_dof_indices = get_indices(self.gripper_joint_names)
        
        # 调试信息：打印关节配置
        print(f"\n[Robot] 关节配置信息：")
        print(f"  总关节数: {len(self.articulation.joint_names)}")
        print(f"  关节列表: {self.articulation.joint_names}")
        print(f"  UR10 DOF索引: {self.ur10_dof_indices} (应为6个)")
        print(f"  夹爪 DOF索引: {self.gripper_dof_indices} (应为8个)")
        print(f"  UR10 DOF数量: {len(self.ur10_dof_indices)}")
        
        if len(self.ur10_dof_indices) != 6:
            print(f"  ⚠ 警告：UR10 DOF数量不正确！预期6个，实际{len(self.ur10_dof_indices)}个")
        
        # 设置PrismaticJoint参数（不通过Articulation配置）
        # [Fix] Increase stiffness to prevent base recoil/jitter
        self.set_prismatic_joint_params(stiffness=30000.0, damping=1000.0)
        
        # Capture default state and initialize current targets
        if self.articulation.data.joint_pos is not None:
             self.default_joint_pos = self.articulation.data.joint_pos.clone()
             self.current_joint_targets = self.default_joint_pos.clone()
        
        self.reset_joints()
        self.setup_ik(ee_link_name="ee_link")
        self.move_gripper(1.0)
        self.write()
        
    def set_prismatic_joint_params(self, stiffness=10000.0, damping=100.0, env_id=None):
        """
        直接设置PrismaticJoint的stiffness和damping参数(支持多环境)
        
        参数:
            stiffness: 刚度值
            damping: 阻尼值
            env_id: 环境ID(可选)
                   - 如果为None，设置所有环境的PrismaticJoint
                   - 如果指定，只设置该环境的PrismaticJoint
        """
        import omni.usd
        from pxr import UsdPhysics
        
        # 获取USD stage
        stage = omni.usd.get_context().get_stage()
        
        # 确定需要设置的环境ID列表
        if env_id is not None:
            env_ids = [env_id]
        else:
            num_envs = self.articulation.num_instances
            env_ids = list(range(num_envs))
        
        success_count = 0
        
        for eid in env_ids:
            # 1. 确定精确的 joint 路径 (基于机器人自身的 prim_path 推导，以支持单/多环境)
            # 例如: /World/Scene/UR10 -> /World/Scene/R_2F_140/PrismaticJoint
            # 或: /World/Scene_0/Scene/UR10 -> /World/Scene_0/Scene/R_2F_140/PrismaticJoint
            if "/UR10" in self.prim_path:
                joint_path = self.prim_path.replace("/UR10", "/R_2F_140/PrismaticJoint")
            else:
                # 兜底转换
                if num_envs == 1:
                    joint_path = "/World/Scene/R_2F_140/PrismaticJoint"
                else:
                    joint_path = f"/World/Scene_{eid}/Scene/R_2F_140/PrismaticJoint"
            
            joint_prim = stage.GetPrimAtPath(joint_path)
            
            if not joint_prim.IsValid():
                print(f"[Robot] ⚠ Env {eid}: 找不到 PrismaticJoint at {joint_path}")
                continue
            
            # 设置参数
            try:
                # 尝试获取或应用 DriveAPI
                drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
                if not drive_api:
                    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
                
                if drive_api:
                    drive_api.GetStiffnessAttr().Set(stiffness)
                    drive_api.GetDampingAttr().Set(damping)
                    # 还需要设置目标位置，否则默认为0可能也是对的，但要确保它是position control
                    # drive_api.GetTypeAttr().Set("force") # 不，应该是默认的 position?
                    # 通常 Isaac Sim 默认是 Position drive.
                    success_count += 1
                else:
                    print(f"[Robot] ⚠ Env {eid}: 无法应用DriveAPI at {joint_path}")
                    
            except Exception as e:
                print(f"[Robot] ⚠ Env {eid}: 设置参数时出错: {e}")
        
        # 打印总结
        if success_count > 0:
            print(f"[Robot] ✓ PrismaticJoint参数已更新 ({success_count}/{len(env_ids)} 环境):")
            print(f"  Stiffness: {stiffness}")
            print(f"  Damping: {damping}")
        else:
            print(f"[Robot] ⚠ 未能更新任何环境的PrismaticJoint参数")
        
    def reset(self):
        if hasattr(self, 'default_joint_pos'):
            self.current_joint_targets = self.default_joint_pos.clone()
            self.articulation.set_joint_position_target(self.current_joint_targets)
            self.articulation.write_data_to_sim()
        
    def reset_joints(self):
        self.articulation.reset()
        if self.articulation.data.joint_pos is not None:
            self.articulation.set_joint_position_target(self.articulation.data.joint_pos)
            self.articulation.write_data_to_sim()

    def setup_ik(self, ee_link_name="ee_link"):
        self.ee_body_name = ee_link_name
        body_ids, body_names = self.articulation.find_bodies(ee_link_name)
        if not body_ids:
            return
        self.ee_body_idx = body_ids[0]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},   # [Standard] 标准DLS阻尼
        )
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=self.articulation.num_instances, device=self.device)

        self._capture_initial_pose()

    def _capture_initial_pose(self):
        ee_pos_w, ee_quat_w = self.get_end_effector_pose(ee_link_name=self.ee_body_name)
        
        if ee_quat_w is None:
            return
        
        root_quat_w = self.articulation.data.root_quat_w
        
        ref_ee_quat_w = ee_quat_w[0].unsqueeze(0)
        ref_root_quat_w = root_quat_w[0].unsqueeze(0)
        
        self.fixed_ee_orientation = quat_mul(quat_inv(ref_root_quat_w), ref_ee_quat_w).squeeze(0)

    # ==================================================================================================
    # 控制方法 (Control Methods)
    # ==================================================================================================

    def move_ik(self, target_pos, target_quat=None):
        """
        使用 IK 移动到指定位置，同时保持初始姿态。
        输入的目标位置为夹爪末端 (Tip) 的位置。

        Args:
            target_pos (torch.Tensor): 夹爪末端目标位置 (N, 3) (相对于基座)
            target_quat (torch.Tensor, optional): 目标姿态四元数 (N, 4) (相对于基座)。如果为 None，则使用初始姿态。
        """
        if self.fixed_ee_orientation is None: self._capture_initial_pose()
        if self.fixed_ee_orientation is None: return print("错误: 无法执行 move_ik，因为没有参考姿态。")

        target_quat = target_quat if target_quat is not None else self.fixed_ee_orientation.repeat(target_pos.shape[0], 1)
        wrist_target_pos = target_pos + torch.tensor([0.0, 0.0, 0.2333], device=self.device)
        
        if (targets := self.compute_ik(wrist_target_pos, target_quat)) is not None:
             self.apply_joint_actions(ur10_actions=targets)

    def move_gripper(self, pos):
        """
        控制夹爪开合。

        Args:
            pos (float): 夹爪位置，0.0 表示完全张开，1.0 表示完全闭合 (映射到 0.0 - 0.72 弧度)。
            force_instant (bool): 是否强制立即设置关节位置（跳过物理过程）。
        """
        joint_pos = max(0.0, min(1.0, pos)) * 0.784
        # 8个关节: finger_joint, right_outer_knuckle, left_outer_finger, right_outer_finger, left_inner_finger, right_inner_finger, left_inner_finger_pad, right_inner_finger_pad
        # pad_joint跟随inner_finger运动
        multipliers = torch.tensor([1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0], device=self.device)
        self.apply_joint_actions(gripper_actions=(multipliers * joint_pos).repeat(self.articulation.num_instances, 1))
        
    

    def compute_ik(self, target_pos, target_quat):
        """
        [功能]: 计算目标位姿的关节位置
        [输入]: target_pos (Tensor N*3), target_quat (Tensor N*4)
        [输出]: ur10_joint_targets (Tensor N*6)
        """
        if self.ik_controller is None: return None
        
        # Transform target to world frame
        root_quat_w = self.articulation.data.root_quat_w
        target_pos_w = quat_apply(root_quat_w, target_pos) + self.articulation.data.root_pos_w
        target_quat_w = quat_mul(root_quat_w, target_quat)
        
        # Get EE state
        ee_pose_w = self.articulation.data.body_state_w[:, self.ee_body_idx, :7]
        ee_quat = ee_pose_w[:, 3:7]

        # Force Gripper Down (z-axis > 0)
        z_axis = quat_apply(target_quat_w, torch.tensor([0., 0., 1.], device=self.device).repeat(len(target_pos), 1))
        if (mask_up := z_axis[:, 2] > 0).any():
             target_quat_w[mask_up] = quat_mul(target_quat_w[mask_up], torch.tensor([0., 1., 0., 0.], device=self.device).repeat(mask_up.sum(), 1))

        # Check Symmetry (180 deg z-rot)
        target_quat_alt = quat_mul(torch.tensor([0., 0., 0., 1.], device=self.device).repeat(len(target_pos), 1), target_quat_w)
        if (mask_alt := torch.abs((target_quat_w * ee_quat).sum(1)) < torch.abs((target_quat_alt * ee_quat).sum(1))).any():
             target_quat_w[mask_alt] = target_quat_alt[mask_alt]
             
        # Shortest path interpolation
        if (mask_neg := (target_quat_w * ee_quat).sum(1) < 0).any(): target_quat_w[mask_neg] *= -1

        self.ik_controller.set_command(torch.cat([target_pos_w, target_quat_w], dim=-1))
        
        jacobian = self.articulation.root_physx_view.get_jacobians()[:, self.ee_body_idx, :, :]
        
        
        # [关键修复] 捕获 IK 奇异点异常 (Batch FailSafe)
        try:
            # 1. 尝试批量求解 (Batch Compute)
            # 注意: 如果有一个环境奇异，这里的 batch compute 可能会抛出异常
            ur10_joint_targets = self.ik_controller.compute(ee_pose_w[:, :3], ee_quat, jacobian[:, :, self.ur10_dof_indices], self.articulation.data.joint_pos[:, self.ur10_dof_indices])
            
            # 2. 如果之前有 fail indices, 需要覆盖这些环境的 target 为当前位置 (Freeze)
            if self.ik_fail_indices:
                current_pos = self.articulation.data.joint_pos[:, self.ur10_dof_indices]
                for idx in self.ik_fail_indices:
                    ur10_joint_targets[idx] = current_pos[idx]
                    
            return ur10_joint_targets
            
        except (torch._C._LinAlgError, RuntimeError) as e:
            # print(f"⚠ [IK] 批量求解失败 ({e})，切换为逐环境求解...")
            
            # 3. Fallback: 逐个环境求解 (Sequential Fallback)
            num_envs = self.articulation.num_instances
            ur10_joint_targets = self.articulation.data.joint_pos[:, self.ur10_dof_indices].clone() # 默认保持当前位置
            
            for i in range(num_envs):
                # 如果已经标记为失败，跳过 (保持当前位置)
                if i in self.ik_fail_indices:
                    # print(f"⚠ [IK] 环境 {i} IK 失败 (Singularity/Error). 继续...")
                    continue
                    
                try:
                    # 提取单个环境的数据
                    # 注意: DifferentialIKController.compute 需要 batch 维度
                    ee_pos_i = ee_pose_w[i, :3].unsqueeze(0)
                    ee_quat_i = ee_quat[i].unsqueeze(0)
                    jacobian_i = jacobian[i, :, self.ur10_dof_indices].unsqueeze(0)
                    joint_pos_i = self.articulation.data.joint_pos[i, self.ur10_dof_indices].unsqueeze(0)
                    
                    # 求解
                    target_i = self.ik_controller.compute(ee_pos_i, ee_quat_i, jacobian_i, joint_pos_i)
                    ur10_joint_targets[i] = target_i[0]
                    
                except (torch._C._LinAlgError, RuntimeError):
                    if i not in self.ik_fail_indices:
                        print(f"❌ [IK FailSafe] Env {i} IK 失败 (Singularity/Error)")
                        self.ik_fail_indices.add(i)
                    # 保持 ur10_joint_targets[i] 为当前位置 (Freeze)
                    
            return ur10_joint_targets



    def apply_joint_actions(self, ur10_actions=None, gripper_actions=None):
        """
        应用关节动作到机器人。

        Args:
            ur10_actions (torch.Tensor): UR10 的关节位置目标 (N, 6)
            gripper_actions (torch.Tensor): 夹爪的关节位置目标 (N, 6)
        """
        if ur10_actions is not None:
            self.current_joint_targets[:, self.ur10_dof_indices] = ur10_actions
            
        if gripper_actions is not None:
            self.current_joint_targets[:, self.gripper_dof_indices] = gripper_actions
            
        # Apply the full set of targets to ensure persistence
        self.articulation.set_joint_position_target(self.current_joint_targets)

    def update(self, dt):
        """
        更新 Articulation 状态。

        Args:
            dt (float): 时间步长。
        """
        self.articulation.update(dt)

    def write(self):
        """
        将命令写入仿真。
        """
        self.articulation.write_data_to_sim()

    # ==================================================================================================
    # 状态获取方法 (State Retrieval Methods)
    # ==================================================================================================

    def get_ur10_state(self):
        """
        读取 UR10 机械臂的当前状态。

        Returns:
            dict: 包含关节位置和速度的字典。
        """
        if not self.ur10_dof_indices:
            return None
            
        full_pos = self.articulation.data.joint_pos
        full_vel = self.articulation.data.joint_vel
        
        return {
            "joint_pos": full_pos[:, self.ur10_dof_indices],
            "joint_vel": full_vel[:, self.ur10_dof_indices]
        }

    def get_2f140_state(self):
        """
        读取 2F140 夹爪的当前状态。

        Returns:
            dict: 包含关节位置和速度的字典。
        """
        if not self.gripper_dof_indices:
            return None

        full_pos = self.articulation.data.joint_pos
        full_vel = self.articulation.data.joint_vel
        
        return {
            "joint_pos": full_pos[:, self.gripper_dof_indices],
            "joint_vel": full_vel[:, self.gripper_dof_indices]
        }

    def get_end_effector_pose(self, ee_link_name="ee_link"):
        """
        获取末端执行器的当前位姿。

        Args:
            ee_link_name (str): 末端执行器 Link 名称。

        Returns:
            tuple: (position, orientation)，其中 position 为 (N, 3)，orientation 为 (N, 4)
        """
        # 确保已获取 Body 索引
        if self.ee_body_idx is None or self.ee_body_name != ee_link_name:
            body_ids, _ = self.articulation.find_bodies(ee_link_name)
            if not body_ids:
                print(f"警告: 未找到名称为 {ee_link_name} 的 Body。")
                return None, None
            self.ee_body_idx = body_ids[0]
            self.ee_body_name = ee_link_name

        ee_pose_w = self.articulation.data.body_state_w[:, self.ee_body_idx, :7]
        return ee_pose_w[:, 0:3], ee_pose_w[:, 3:7] # position, orientation 3 + 4 





    def execute_push(self, scene, direction_index, push_center, dt=0.01):
        """
        [功能]: 执行推操作序列
        [输入]: scene (Scene), direction_index (int/Tensor), push_center (List/Tensor)
        [输出]: None
        """
        # 确保输入是 Tensor 并扩展到 batch size
        num_envs = self.articulation.num_instances
        
        if not isinstance(direction_index, torch.Tensor):
            direction_index = torch.tensor([direction_index], device=self.device)
        if direction_index.dim() == 0:
            direction_index = direction_index.unsqueeze(0)
        if direction_index.shape[0] == 1 and num_envs > 1:
            direction_index = direction_index.repeat(num_envs)
            
        if not isinstance(push_center, torch.Tensor):
            push_center = torch.tensor(push_center, device=self.device)
        if push_center.dim() == 1:
            push_center = push_center.unsqueeze(0)
        if push_center.shape[0] == 1 and num_envs > 1:
            push_center = push_center.repeat(num_envs, 1)

        # [关键修正] 坐标系转换 World -> Local
        # move_to 和 ik_controller 期望的是相对于机器人基座的局部坐标
        # 而输入 push_center 是全局坐标
        from isaaclab.utils.math import quat_apply, quat_inv
        
        base_pos = self.articulation.data.root_pos_w
        base_quat = self.articulation.data.root_quat_w
        
        # Local = Inv(BaseRot) * (World - BasePos)
        push_center_local = quat_apply(quat_inv(base_quat), push_center - base_pos)

        # 使用初始姿态作为基准姿态，以便计算旋转
        base_quat_local = None
        if self.fixed_ee_orientation is not None:
             base_quat_local = self.fixed_ee_orientation.repeat(num_envs, 1)

        # 获取路径点 (使用局部坐标)
        waypoints = self.push_primitive.get_waypoints(push_center_local, direction_index, base_quat=base_quat_local)
        
        # 执行路径点
        # 0: Pre-push (Approach) -> Speed 1.5 (Reduced from 3.0 for better rotation stability)
        # 1: Pre-push -> Start (Descent) -> Speed 0.5 (Increased from 0.1 to avoid slow drift)
        # 2: Start -> End (Push) -> Speed 1.0
        # 3: End -> Lift (Lift) -> Speed 2.0
        speeds = [1.5, 0.5, 1.0, 2.0]
        
        for i, (target_pos, target_quat) in enumerate(waypoints):
            # Determine speed for this segment
            current_speed = speeds[i] if i < len(speeds) else 1.0
            
            # 默认使用闭合夹爪 (1.0)
            self.move_to(scene, target_pos, target_quat, dt=dt, gripper_pos=1.0, speed=current_speed)

    def return_to_origin(self, scene, origin_point, dt=0.01, gripper_pos=1.0):
        """
        [功能]: 控制机器人回到原点
        [输入]: scene (Scene), origin_point (Tensor N*3), dt (float), gripper_pos (float)
        [输出]: None
        """
       
        # 确保高度为 0.1
        target_pos = origin_point.clone()
        
        if self.fixed_ee_orientation is None:
            print("Warning: Robot orientation not initialized.")
            return

        target_quat = self.fixed_ee_orientation.repeat(target_pos.shape[0], 1)
        
        # 复用 move_to 的逻辑
    def get_push_plan(self, direction_index, push_center):
        """
        [功能]: 生成推操作的路径规划 (不执行)
        [输入]: direction_index (int/Tensor), push_center (List/Tensor)
        [输出]: list of dict [{'target_pos':, 'target_quat':, 'speed':}, ...]
        """
        # 确保输入是 Tensor 并扩展到 batch size
        num_envs = self.articulation.num_instances
        
        if not isinstance(direction_index, torch.Tensor):
            direction_index = torch.tensor([direction_index], device=self.device)
        if direction_index.dim() == 0:
            direction_index = direction_index.unsqueeze(0)
        if direction_index.shape[0] == 1 and num_envs > 1:
            direction_index = direction_index.repeat(num_envs)
            
        if not isinstance(push_center, torch.Tensor):
            push_center = torch.tensor(push_center, device=self.device)
        if push_center.dim() == 1:
            push_center = push_center.unsqueeze(0)
        if push_center.shape[0] == 1 and num_envs > 1:
            push_center = push_center.repeat(num_envs, 1)

        # 坐标系转换 World -> Local
        from isaaclab.utils.math import quat_apply, quat_inv
        
        base_pos = self.articulation.data.root_pos_w
        base_quat = self.articulation.data.root_quat_w
        
        # Local = Inv(BaseRot) * (World - BasePos)
        push_center_local = quat_apply(quat_inv(base_quat), push_center - base_pos)

        # 使用初始姿态作为基准姿态
        base_quat_local = None
        if self.fixed_ee_orientation is not None:
             base_quat_local = self.fixed_ee_orientation.repeat(num_envs, 1)

        # 获取路径点 (使用局部坐标)
        waypoints = self.push_primitive.get_waypoints(push_center_local, direction_index, base_quat=base_quat_local)
        
        # 定义每个阶段的速度（降低速度以避免奇异点）
        # 0: Approach, 1: Descent, 2: Push, 3: Lift
        # 原值: [4.0, 0.5, 1.0, 4.0] → 新值: [2.0, 0.3, 0.6, 2.0]
        speeds = [1.0, 0.5, 0.2, 1.0]
        
        plan = []
        for i, (target_pos, target_quat) in enumerate(waypoints):
            current_speed = speeds[i] if i < len(speeds) else 1.0
            plan.append({
                'target_pos': target_pos,
                'target_quat': target_quat,
                'speed': current_speed,
                'gripper_pos': 1.0
            })
            
        return plan

    def move_to(self, scene, target_pos, target_quat, dt=0.01, gripper_pos=1.0, threshold=0.02, timeout=10.0, speed=1):
        """
        [功能]: 控制机器人移动到指定位姿（带姿态插值）
        [输入]: scene (Scene), target_pos (Tensor N*3), target_quat (Tensor N*4)
        [输出]: None
        """
        from isaaclab.utils.math import quat_slerp, quat_apply, quat_inv, quat_mul  # 四元数球面线性插值
        
        offset_vec = torch.tensor([0.0, 0.0, 0.2333], device=self.device)
        ee_pos = self.get_end_effector_pose(self.ee_body_name)[0]
        ee_quat = self.get_end_effector_pose(self.ee_body_name)[1]  # 获取当前姿态
        
        start_pos = quat_apply(quat_inv(self.articulation.data.root_quat_w), 
                             (ee_pos if ee_pos is not None else torch.zeros(3, device=self.device)) - self.articulation.data.root_pos_w) - offset_vec
        
        # [DEBUG] 打印机器人基座位置与目标点
        # if "Scene_1" in self.prim_path or "Scene_0" in self.prim_path:
        #     base_pos = self.articulation.data.root_pos_w
        #     print(f"\n[DEBUG Robot {self.prim_path}] Base W: {base_pos}, Target L: {target_pos}")

        # 获取起始姿态（相对于base）
        if ee_quat is not None:
            start_quat = quat_mul(quat_inv(self.articulation.data.root_quat_w), ee_quat)
        else:
            start_quat = target_quat if target_quat is not None else torch.zeros((start_pos.shape[0], 4), device=self.device)
        
        # [Fix] 如果target_quat为None，使用当前姿态（保持姿态不变）
        if target_quat is None:
            target_quat = start_quat
                             
        # [Fix] Ensure target quaternion is in the same hemisphere as start quaternion (shortest path)
        dot = torch.sum(start_quat * target_quat, dim=1)
        mask_neg = dot < 0
        if mask_neg.any():
            target_quat = target_quat.clone()
            target_quat[mask_neg] *= -1
            dot[mask_neg] *= -1 # Updated dot for angle calculation below

        # 计算位移和旋转量
        dist = torch.norm(target_pos - start_pos, dim=1)
        
        # 计算旋转角度 (2 * acos(|dot|))
        angle = 2 * torch.acos(dot.clamp(min=-1.0, max=1.0))
        
        # 估算持续时间: Max(位移时间, 旋转时间)
        # 假设旋转速度: pi/2 rad/s (90度/秒) * speed
        rot_speed = (3.14159 / 2.0) * speed 
        t_pos = dist / speed
        t_rot = angle / rot_speed
        
        durations = torch.max(t_pos, t_rot)
        
        elapsed_time = 0
        stable_steps = 0
        
        while elapsed_time < timeout and scene.is_app_running():
            alpha = torch.clamp(elapsed_time / torch.clamp(durations, min=dt), 0.0, 1.0).unsqueeze(1)
            
            # 位置线性插值
            interp_pos = start_pos + (target_pos - start_pos) * alpha
            
            # 姿态球面线性插值（SLERP），避免姿态突变
            # 处理batch维度
            if start_quat.dim() == 1:
                # 单环境：直接插值
                interp_quat = quat_slerp(start_quat, target_quat, alpha.squeeze())
            else:
                # 多环境：对每个环境分别插值
                interp_quat = torch.stack([
                    quat_slerp(start_quat[i], target_quat[i], alpha[i].squeeze())
                    for i in range(start_quat.shape[0])
                ])
            
            self.move_ik(interp_pos, interp_quat)
            self.move_gripper(gripper_pos)
            self.write()
            scene.step()
            self.update(dt)
            
            elapsed_time += dt
            
            # Check error (Position + Orientation)
            if (ee_pose_w := self.get_end_effector_pose(self.ee_body_name)) is not None:
                ee_pos_w, ee_quat_w = ee_pose_w
                target_w = quat_apply(self.articulation.data.root_quat_w, target_pos + offset_vec) + self.articulation.data.root_pos_w
                target_quat_w = quat_mul(self.articulation.data.root_quat_w, target_quat)
                
                pos_err = torch.norm(target_w - ee_pos_w, dim=1).mean().item()
                
                # Orientation error (1 - |dot(q1, q2)|)
                # q1 dot q2
                dot_err = torch.sum(ee_quat_w * target_quat_w, dim=1).abs()
                rot_err = (1.0 - dot_err).mean().item()
                
                # Consider converged if position < threshold AND orientation strictly aligned
                # 0.001 approx 2.5 degrees error. Strict alignment required.
                if pos_err < threshold and rot_err < 0.001: 
                     if (stable_steps := stable_steps + 1) > 10 and elapsed_time >= durations.max(): break
                else: stable_steps = 0



