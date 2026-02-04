import argparse
from typing import Optional, TYPE_CHECKING
from config import initialize_app, configure_simulation

if TYPE_CHECKING:
    from robot import Robot

class Scene:
    """
    管理仿真场景和应用程序生命周期的类。
    """
    def __init__(self, description="Isaac Lab Scene", num_envs=1, env_spacing=2.0):
        """
        [功能]: 初始化 Scene 类。
        [输入]: description (str): 应用程序的描述信息。
                num_envs (int): 并行环境数量。
                env_spacing (float): 环境间距 (米)。
        """
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.usd_path = "/home/k/Projects/equi/mesh/env.usd"
        
        # 记录关键路径
        self.env_paths = {
            "UR10": "Scene/UR10",
            "Ground": "Scene/Ground",
            "Table": "Scene/Table",
            "R_2F_140": "Scene/R_2F_140"
        }
        
        # 确定机器人的 Prim 路径
        # env.usd内部结构是 Scene/UR10, Scene/R_2F_140 等
        if self.num_envs > 1:
            # 多环境约定：/World/Scene_{id}/Scene/xxx
            # 使用通配符用于后续查找，或固定基础路径
            self.robot_prim_path = "/World/Scene_.*/Scene/UR10"
            self.gripper_prim_path = "/World/Scene_.*/Scene/R_2F_140"
        else:
            # 单环境约定：/World/Scene/xxx
            self.robot_prim_path = "/World/Scene/UR10"
            self.gripper_prim_path = "/World/Scene/R_2F_140"

        # 启动应用程序
        def add_args(parser):
            pass

        self.simulation_app, self.app_launcher, self.args = initialize_app(description, add_args)
        
        # 导入依赖模块 (必须在应用启动后)
        import isaacsim.core.utils.stage as stage_utils
        
        self.stage_utils = stage_utils
        
        self.sim = None
        self.robot: Optional["Robot"] = None
        self.load()

    def load_stage(self):
        """
        [功能]: 阶段1：加载 USD 舞台（不启动物理引擎）
        
        """
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
        import math

        # 如果只有一个环境，直接打开舞台
        if self.num_envs <= 1:
            self.stage_utils.open_stage(self.usd_path)
        else:
            self.stage_utils.create_new_stage()
            grid_width = int(math.ceil(math.sqrt(self.num_envs)))
            stage = omni.usd.get_context().get_stage()
            
            for i in range(self.num_envs):
                x, y = (i // grid_width) * self.env_spacing, (i % grid_width) * self.env_spacing
                env_path = f"/World/Scene_{i}"
                UsdGeom.Xformable(stage.DefinePrim(env_path, "Xform")).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
                stage.GetPrimAtPath(env_path).GetReferences().AddReference(self.usd_path)
                
                if i > 0 and (g := stage.GetPrimAtPath(f"{env_path}/{self.env_paths['Ground']}")).IsValid():
                    g.SetActive(False)

        # 配置仿真上下文（但不启动）
        self.sim = configure_simulation(self.app_launcher)
        
        print("[Scene] Stage loaded. Ready for camera creation.")

    def start_simulation(self):
        """
        阶段2：启动物理引擎和初始化机器人
        支持多场景：为每个场景创建独立的Camera和State
        """
        if not self.sim:
            raise RuntimeError("Must call load_stage() before start_simulation()")
        
        # === 为每个场景创建独立相机 ===
        from camera import Camera
        exclude_paths = [self.robot_prim_path, self.gripper_prim_path]
        
        self.cameras = []
        for i in range(self.num_envs):
            if self.num_envs > 1:
                cam_path = f"/World/Scene_{i}/Scene/Camera/CameraSensor"
                # 为每个场景传递具体的robot路径（不使用通配符）
                env_exclude_paths = [
                    f"/World/Scene_{i}/Scene/UR10",
                    f"/World/Scene_{i}/Scene/R_2F_140"
                ]
            else:
                cam_path = "/World/Scene/Camera/CameraSensor"
                env_exclude_paths = exclude_paths
            
            camera = Camera(prim_path=cam_path, exclude_prim_paths=env_exclude_paths, height=480, width=640)
            camera.initialize()
            self.cameras.append(camera)
        
        print(f"[Scene] Created {len(self.cameras)} camera(s)")
        
        # 启动物理引擎
        self.sim.reset()
        
        # === 为每个场景创建独立Robot ===
        from robot import Robot
        self.robots = []
        for i in range(self.num_envs):
            if self.num_envs > 1:
                robot_path = f"/World/Scene_{i}/Scene/UR10"
            else:
                robot_path = "/World/Scene/UR10"
            
            robot = Robot(prim_path=robot_path)
            robot.initialize()
            self.robots.append(robot)
        
        # 单场景时提供便捷访问
        if self.num_envs == 1:
            self.robot = self.robots[0]
        
        print(f"[Scene] Created {len(self.robots)} robot(s)")
        
        # 刷新所有相机的exclusions
        for camera in self.cameras:
            camera.setup_exclusions()
        
        # === 为每个场景创建State ===
        from state import State
        # === 为每个场景创建State ===
        from state import State
        import math
        
        self.states = []
        grid_width = int(math.ceil(math.sqrt(self.num_envs)))
        
        for i, camera in enumerate(self.cameras):
            # Calculate env origin
            if self.num_envs > 1:
                row = i // grid_width
                col = i % grid_width
                x = row * self.env_spacing
                y = col * self.env_spacing
                origin = (x, y)
            else:
                origin = (0.0, 0.0)
                
            state = State(camera=camera, env_idx=i, env_origin=origin)
            self.states.append(state)
        
        # 单场景时提供便捷访问
        if self.num_envs == 1:
            self.state = self.states[0]
        
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/visualization/enable", False)
        settings.set_bool("/physics/visualization/showJoints", False)
        
        print(f"[Scene] Simulation started with {self.num_envs} environment(s).")
    
    def update_cameras(self, dt):
        """
        [功能]: 批量更新所有相机
        [输入]: dt (float): 时间步长
        """
        for camera in self.cameras:
            camera.update(dt)
    
    def reset_cameras(self):
        """
        [功能]: 批量reset所有相机
        """
        for camera in self.cameras:
            camera.reset()


    def load(self):
        """
        [功能]: 直接调用 load_stage() + start_simulation()
        [说明]: 保留用于向后兼容。
        """
        self.load_stage()
        self.start_simulation()

    def step(self):
        """
        执行一步仿真。
        """
        if self.robot:
            self.robot.write()
            
        if self.sim:
            self.sim.step()
            
        if self.robot and self.sim:
            self.robot.update(self.sim.get_physics_dt())

    def is_playing(self):
        """
        [功能]: 检查仿真是否正在播放。
        [输出]: bool: 如果正在播放返回 True，否则返回 False。
        """
        if self.sim:
            return self.sim.is_playing()
        return False

    def is_app_running(self):
        """
        检查应用程序是否正在运行。
        
        Returns:
            bool: 如果正在运行返回 True，否则返回 False。
        """
        return self.simulation_app.is_running()

    def load_usd_object(self, usd_path, init_pos, init_rot=None, name="object", prim_path_pattern=None):
        """
        [功能]: 从 USD 文件加载刚体对象到模拟器中。
        [输入]: usd_path (str): USD 文件路径.
                init_pos (list or torch.Tensor): 对象初始位置 [x, y, z].
                init_rot (list or torch.Tensor, optional): 对象初始旋转四元数 [w, x, y, z]. 默认为 None，使用 [1,0,0,0]。
                name (str, optional): prim_path_pattern 为 None 时，对象的名称用于构建默认路径. 默认为 "object"。
                prim_path_pattern (str, optional): USD 场景图中的路径模式,支持通配符 ".*" 用于多环境.
                                               如果为 None,将根据 self.num_envs 自动推断:
                                               - 单环境: /World/Scene/{name}
                                               - 多环境: /World/Scene_.*/{name}
        [输出]: RigidObject: 创建的刚体对象实例。
        """
        import torch
        from isaaclab.assets import RigidObject, RigidObjectCfg
        import isaaclab.sim as sim_utils

        # 自动推断 prim_path_pattern 如果未提供
        if prim_path_pattern is None:
            if self.num_envs > 1:
                # 多环境模式
                prim_path_pattern = f"/World/Scene_.*/{name}"
            else:
                # 单环境模式
                prim_path_pattern = f"/World/Scene/{name}"
        
        if isinstance(init_pos, torch.Tensor):
            init_pos = init_pos.tolist()
        
        if init_rot is not None and isinstance(init_rot, torch.Tensor):
            init_rot = init_rot.tolist()
        
        # 检查是否需要启用接触传感器
        enable_contact_sensors = getattr(self, '_enable_contact_sensors', False)
            
        obj_cfg = RigidObjectCfg(
            prim_path=prim_path_pattern,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(0.01, 0.01, 0.01),  # 缩小100倍
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                ),
                activate_contact_sensors=enable_contact_sensors,  # 启用接触报告API（ContactSensor需要）
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot if init_rot is not None else [1.0, 0.0, 0.0, 0.0]
            ),
        )
        
        obj = RigidObject(obj_cfg)
        
        return obj

    def find_safe_positions(self, objects, candidate_radius=0.05, workspace=[[0.55, 0.95], [-0.2, 0.2], [0.1, 0.15]], min_dist=0.07,
                            max_dist=0.1,
                            num_positions=1, max_attempts=10000):
        """
        [功能]: 在工作空间中找到安全的位置。
        [输入]: objects (list of tuples): 已有对象的位置和半径。
                candidate_radius (float, optional): 候选位置的半径。默认为 0.05。
                workspace (list of lists, optional): 工作空间的边界。默认为 [[0.55, 0.95], [-0.2, 0.2], [0.1, 0.15]]。
                min_dist (float, optional): 最小距离。默认为 0.07。
                max_dist (float, optional): 最大距离。默认为 0.1。
                num_positions (int, optional): 要找到的位置数量。默认为 1。
                max_attempts (int, optional): 最大尝试次数。默认为 10000。
        [输出]: list of tuples: 找到的安全位置。
        """
        import math
        import random
        
        def is_valid(c):
            # Check workspace bounds
            if not all(workspace[i][0] <= c[i] <= workspace[i][1] for i in range(3)): return False
            if not objects: return True
            
            # Check object distances
            surf_dists = [math.dist(c, obj[0]) - candidate_radius - obj[1] for obj in objects]
            return all(d >= min_dist for d in surf_dists) and (min(surf_dists) <= max_dist)

        for _ in range(max_attempts):
            c = [round(random.uniform(*workspace[i]), 4) for i in range(3)]
            if is_valid(c): return c
        return []

    def _spawn_object(self, name, usd_path, pos, quat, prim_path):
        """Spawn object from USD file - uses USD's native materials"""
        obj = self.load_usd_object(usd_path, pos, quat, name=name, prim_path_pattern=prim_path)
        return obj
    
    def _get_or_create_contact_sensor(self, prim_path):
        """
        [功能]: 懒加载方式获取或创建 ContactSensor（避免每次重置都重新创建）
        """
        if not hasattr(self, '_contact_sensors'):
            self._contact_sensors = {}
        
        # 如果已存在且有效，直接返回
        if prim_path in self._contact_sensors:
            return self._contact_sensors[prim_path]
        
        # 首次使用时创建（懒加载）
        try:
            from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
            
            sensor_cfg = ContactSensorCfg(
                prim_path=prim_path,
                debug_vis=False,
            )
            sensor = ContactSensor(cfg=sensor_cfg)
            
            # 关键：手动初始化传感器（因为仿真已经在运行，PLAY 事件已过）
            # 这会设置 _timestamp, _device, _physics_sim_view 等必要属性
            if not sensor.is_initialized:
                sensor._initialize_impl()
                sensor._is_initialized = True
                
            self._contact_sensors[prim_path] = sensor
            return sensor
        except Exception as e:
            print(f"  ⚠️ [ContactSensor] 创建传感器失败: {e}")
            return None
    
    def clear_contact_sensors(self):
        """
        [功能]: 场景重置时清理传感器缓存（因为物体 prim 已重建）
        """
        if hasattr(self, '_contact_sensors'):
            self._contact_sensors.clear()
    
    def update_contact_sensors(self, dt=0.01):
        """
        [功能]: 更新所有接触传感器数据
        [输入]: dt - 时间步长
        """
        if not hasattr(self, '_contact_sensors'):
            return
        
        for sensor in self._contact_sensors.values():
            try:
                sensor.update(dt, force_recompute=True)
            except Exception as e:
                pass  # 静默处理
    
    def get_contact_force_info(self, prim_path):
        """
        [功能]: 获取指定物体的接触力信息（如果可用）
        [输入]: prim_path - 物体的 prim 路径
        [输出]: dict - 包含 net_forces_w 等信息
        [注意]: ContactSensor 在 Isaac Lab 中需要特定初始化流程，动态创建可能失败
        """
        import torch
        
        # 检查是否启用接触传感器
        if not getattr(self, '_enable_contact_sensors', False):
            return {'error': '接触传感器未启用', 'net_force': None, 'net_force_magnitude': None}
        
        result = {
            'error': None,
            'net_force': None,
            'net_force_magnitude': None
        }
        
        try:
            # 检查是否有已创建的传感器
            if not hasattr(self, '_contact_sensors') or prim_path not in self._contact_sensors:
                # 尝试创建传感器（可能因初始化问题失败）
                sensor = self._get_or_create_contact_sensor(prim_path)
                if sensor is None:
                    result['error'] = 'ContactSensor 创建失败'
                    return result
            else:
                sensor = self._contact_sensors[prim_path]
            
            # 尝试更新和读取传感器数据
            # 注意：如果传感器未正确初始化，这里可能会失败
            sensor.update(dt=0.01, force_recompute=True)
            
            # 获取净接触力 (N, B, 3)
            if sensor.data.net_forces_w is not None:
                net_forces = sensor.data.net_forces_w
                result['net_force'] = net_forces.cpu().tolist()
                result['net_force_magnitude'] = torch.norm(net_forces).item()
                
        except AttributeError as e:
            # 传感器未完全初始化（缺少 _timestamp 等属性）
            result['error'] = 'ContactSensor 未初始化完成'
        except Exception as e:
            result['error'] = str(e)
        
        return result

    def _gather_objects(self, spawned_objects, reset_sim=True):
        """
        [功能]: 聚集所有已生成的物体。
        [输入]: spawned_objects (list): 已生成的物体列表。
                reset_sim (bool, optional): 是否重置模拟器。默认为 True。
        """
        import torch
        if not spawned_objects or self.sim is None: 
            return
        
        if reset_sim:
            self.sim.reset()
            if self.robots:
                for robot in self.robots:
                    robot.reset()
            
            for _ in range(50):
                self.step()
        else:
            print(f"  [初始化] 让新物体稳定 10 步...")
            for _ in range(10):
                self.step()

        # 按环境分组物体
        env_objs = {}
        for obj in spawned_objects:
            if "Scene_" in obj.cfg.prim_path:
                eid = int(obj.cfg.prim_path.split("Scene_")[1].split("/")[0])
            else:
                eid = 0
            env_objs.setdefault(eid, []).append(obj)
        
        # [修改] 聚拢方向改为朝着目标物体中心，而不是所有物体的几何中心
        env_target_pos = {}
        for eid, objs in env_objs.items():
            target_obj = None
            for obj in objs:
                # 查找目标物体（名称包含 "Target"）
                if "Target" in obj.cfg.prim_path:
                    target_obj = obj
                    break
            if target_obj is not None:
                env_target_pos[eid] = target_obj.data.root_pos_w[0].clone()
            elif objs:
                # 备用：如果找不到目标物体，使用几何中心
                positions = torch.stack([obj.data.root_pos_w[0] for obj in objs])
                env_target_pos[eid] = torch.mean(positions, dim=0)
        
        gather_steps = 50
        gather_strength = 8.0  # [修改] 降低聚拢力，避免物体挤压后崩飞
        
        for step in range(gather_steps):
            dt = self.sim.get_physics_dt()
            
            # 更新所有物体状态
            for obj in spawned_objects:
                obj.update(dt)
            
            # 为每个环境应用聚拢力
            for eid, objs in env_objs.items():
                if eid not in env_target_pos:
                    continue
                target_pos = env_target_pos[eid]
                for obj in objs:
                    # 目标物体不需要聚拢（已经是中心）
                    if "Target" in obj.cfg.prim_path:
                        continue
                    current_pos = obj.data.root_pos_w[0]
                    direction = target_pos - current_pos
                    # 只在XY平面聚拢，Z轴不施加力(或微弱向下)
                    direction[2] = 0.0
                    
                    distance = torch.norm(direction)
                    
                    if distance > 1e-3:
                        direction = direction / distance  # Normalize
                    
                    # 距离越远力越大，距离极近时不施力以避免穿模震荡
                    if distance > 0.05:
                        strength = gather_strength
                    else:
                         # 靠近中心后减小力
                        strength = gather_strength * (distance / 0.05)
                    
                    force = direction * strength
                    # [修改] 只在XY平面施加力，Z方向不施加力
                    force[2] = 0.0
                    

                    if hasattr(obj, "root_physx_view"):
                        obj.root_physx_view.apply_forces(
                            force.unsqueeze(0), 
                            indices=torch.tensor([0], device=obj.device), 
                            is_global=True
                        )
            
            self.step()
        
    
        
        # print(f"  [缓存位置] 保存聚拢后的物体位置...")
        if not hasattr(self, '_object_position_cache'):
            self._object_position_cache = {}
        
        cached_count = 0
        for obj in spawned_objects:
            obj_prim_path = obj.cfg.prim_path
            if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                self._object_position_cache[obj_prim_path] = {
                    'pos': obj.data.root_pos_w[0].clone().cpu(),
                    'quat': obj.data.root_quat_w[0].clone().cpu(),
                }
                cached_count += 1
        
        # print(f"  ✓ 已缓存 {cached_count} 个物体的位置")

    def _restore_cached_positions(self, objects, env_ids_to_restore):
        """
        [功能]: 恢复指定环境的物体到缓存的位置
        [输入]: objects (list), env_ids_to_restore (list)
        """
        import torch
        
        if not hasattr(self, '_object_position_cache'):
            print("  ⚠ 警告：没有缓存的位置可恢复")
            return
        
        # print(f"  [恢复位置] 恢复环境 {env_ids_to_restore} 的物体到聚拢后位置...")
        restored_count = 0
        
        for obj in objects:
            obj_env_id = self._get_env_id_from_prim_path(obj.cfg.prim_path)
            if obj_env_id in env_ids_to_restore:
                obj_prim_path = obj.cfg.prim_path
                if obj_prim_path in self._object_position_cache:
                    cached = self._object_position_cache[obj_prim_path]
                    
                    if hasattr(obj, 'write_root_pose_to_sim'):
                        device = obj.device if hasattr(obj, 'device') else 'cuda:0'
                        pos_tensor = cached['pos'].to(device)
                        quat_tensor = cached['quat'].to(device)
                        obj.write_root_pose_to_sim(
                            torch.cat([pos_tensor, quat_tensor]).unsqueeze(0)
                        )
                        restored_count += 1
        
        # print(f"  ✓ 已恢复 {restored_count} 个物体的位置")

    def _gentle_gather_for_compensation(self, objects, env_ids_to_compensate):
        """
        [功能]: 对成功环境进行温和补偿聚拢，修正 sim.reset() 导致的位置偏移
        [输入]: objects (list), env_ids_to_compensate (list)
        """
        import torch
        
        if not env_ids_to_compensate:
            return
        
        # print(f"  [补偿聚拢] 对成功环境 {env_ids_to_compensate} 进行温和聚拢...")
        
        # 按环境分组
        env_objs = {}
        for obj in objects:
            obj_env_id = self._get_env_id_from_prim_path(obj.cfg.prim_path)
            if obj_env_id in env_ids_to_compensate:
                env_objs.setdefault(obj_env_id, []).append(obj)
        
        # 计算每个环境的重心
        env_centroids = {}
        for eid, objs in env_objs.items():
            if objs:
                positions = torch.stack([obj.data.root_pos_w[0] for obj in objs])
                centroid = torch.mean(positions, dim=0)
                env_centroids[eid] = centroid
        
        gather_steps = 25
        gather_strength = 10.0
        
        for step in range(gather_steps):
            dt = self.sim.get_physics_dt()
            
            for objs in env_objs.values():
                for obj in objs:
                    obj.update(dt)
            
            # 为每个环境应用温和聚拢力
            for eid, objs in env_objs.items():
                if eid not in env_centroids:
                    continue
                centroid = env_centroids[eid]
                for obj in objs:
                    current_pos = obj.data.root_pos_w[0]
                    direction = centroid - current_pos
                    direction[2] = current_pos[2]
                    distance = torch.norm(direction)
                    
                    if distance > 0.01:
                        force = direction * gather_strength * 0.5
                        if hasattr(obj, "root_physx_view"):
                            obj.root_physx_view.apply_forces(
                                force.unsqueeze(0), 
                                indices=torch.tensor([0], device=obj.device), 
                                is_global=True
                            )
            
            self.step()
        
        # print(f"  ✓ 补偿聚拢完成")

    def _get_env_id_from_prim_path(self, prim_path):
        """
        [功能]: 从prim路径提取环境ID
        [输入]: prim_path (str)
        [输出]: 环境ID (int)
        """
        if "Scene_" in prim_path:
            return int(prim_path.split("Scene_")[1].split("/")[0])
        return 0  # 单环境默认为0

    def _check_objects_in_workspace(self, spawned_objects, workspace_limits):
        """
        [功能]: 向量化检查多个环境中的物体是否在工作空间内
        [输入]: spawned_objects (list), workspace_limits (Tensor [[min_x,min_y,min_z],[max_x,max_y,max_z]])
        [输出]: 每个环境的验证结果 (dict {env_id: bool})
        """
        import torch
        import math
        
        # 1. 按环境分组物体
        env_objs = {}
        for obj in spawned_objects:
            env_id = self._get_env_id_from_prim_path(obj.cfg.prim_path)
            env_objs.setdefault(env_id, []).append(obj)
        
        # 2. 计算每个环境的全局偏移
        grid_width = int(math.ceil(math.sqrt(self.num_envs)))
        
        def get_env_offset(env_id):
            """计算环境的全局偏移量"""
            if self.num_envs <= 1:
                return 0.0, 0.0
            row = env_id // grid_width
            col = env_id % grid_width
            x_offset = row * self.env_spacing
            y_offset = col * self.env_spacing
            return x_offset, y_offset
        
        # 3. 检查每个环境
        dt = self.sim.get_physics_dt()
        env_results = {}
        
        # print(f"\n[调试] 环境数量: {self.num_envs}, 间距: {self.env_spacing}m, Grid宽度: {grid_width}")
        
        for env_id, objs in env_objs.items():
            x_offset, y_offset = get_env_offset(env_id)
            
            # print(f"\n  [Env {env_id}] 偏移量: X={x_offset:.2f}m, Y={y_offset:.2f}m")
            # print(f"  [Env {env_id}] 全局工作空间: X=[{workspace_limits[0,0]+x_offset:.2f}, {workspace_limits[1,0]+x_offset:.2f}], "
            #       f"Y=[{workspace_limits[0,1]+y_offset:.2f}, {workspace_limits[1,1]+y_offset:.2f}], "
            #       f"Z=[{workspace_limits[0,2]:.2f}, ∞]")
            
            # 更新物体状态
            for obj in objs:
                obj.update(dt)
            
            # 检查1：物体中心位置（快速检查，防止物体被崩飞）
            objects_out_of_workspace = False
            for obj in objs:
                pos = obj.data.root_pos_w[0]  # 全局坐标 [x, y, z]
                obj_name = obj.cfg.prim_path.split('/')[-1]
                
                # 转换为本地坐标
                local_x = pos[0] - x_offset
                local_y = pos[1] - y_offset
                local_z = pos[2]
                
                # 检查XY是否在工作空间内（使用中心点，不考虑半径）
                if not (workspace_limits[0, 0] <= local_x <= workspace_limits[1, 0]):
                    # print(f"  ✗ Env {env_id}: 物体 {obj_name} 中心超出X边界")
                    # print(f"    中心X: {local_x:.3f}, 工作空间: [{workspace_limits[0,0]:.2f}, {workspace_limits[1,0]:.2f}]")
                    objects_out_of_workspace = True
                    break
                if not (workspace_limits[0, 1] <= local_y <= workspace_limits[1, 1]):
                    # print(f"  ✗ Env {env_id}: 物体 {obj_name} 中心超出Y边界")
                    # print(f"    中心Y: {local_y:.3f}, 工作空间: [{workspace_limits[0,1]:.2f}, {workspace_limits[1,1]:.2f}]")
                    objects_out_of_workspace = True
                    break
                # Z轴检查
                if local_z < 0.02:
                    # print(f"  ✗ Env {env_id}: 物体 {obj_name} 底部低于桌面")
                    # print(f"    底部Z坐标: {local_z:.3f}, 阈值: 0.02")
                    objects_out_of_workspace = True
                    break
            
            if objects_out_of_workspace:
                env_results[env_id] = False
                continue  # 跳过掩膜检测
            
            # 检查2：使用基于掩膜的出界检测
            state = self.states[env_id]
            
            # 检查2：使用基于掩膜的出界检测
            out_of_bounds, info = state.check_out_of_bounds(verbose=False)
            
            if out_of_bounds:
                print(f"  ✗ Env {env_id}: 物体出界")
                print(f"    检测原因: {info.get('reason', 'unknown')}")
                if 'num_components' in info:
                    print(f"    连通域数量: {info['num_components']}")
                if 'aspect_ratio' in info:
                    print(f"    宽高比: {info['aspect_ratio']:.2f}")
                if 'fill_ratio' in info:
                    print(f"    填充率: {info['fill_ratio']:.2f}")
                env_results[env_id] = False
            else:
                # print(f"  ✓ Env {env_id}: {len(objs)} 个物体全部合格 (基于掩膜检测)")
                env_results[env_id] = True
        
        return env_results

    def _delete_objects(self, spawned_objects, env_ids_to_delete=None):
        """
        [功能]: 删除指定环境的物体
        [输入]: spawned_objects (list), env_ids_to_delete (list/None)
        """
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        
        objects_to_remove = []
        for obj in spawned_objects:
            if env_ids_to_delete is None:
                # 删除全部
                objects_to_remove.append(obj)
            else:
                # 只删除指定环境的物体
                env_id = self._get_env_id_from_prim_path(obj.cfg.prim_path)
                if env_id in env_ids_to_delete:
                    objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            prim_path = obj.cfg.prim_path
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)
            spawned_objects.remove(obj)
        
        # if env_ids_to_delete is None:
        #     print(f"  已删除 {len(objects_to_remove)} 个物体 (全部环境)")
        # else:
        #     print(f"  已删除 {len(objects_to_remove)} 个物体 (环境: {env_ids_to_delete})")

    def create_clutter_environment(self, num_objects_range, workspace_limits=None, env_ids=None, force_regenerate=False, enable_contact_sensors=False):
        """
        [功能]: 在指定环境中随机生成杂乱物体
        [输入]: num_objects_range (int/tuple), workspace_limits (Tensor), env_ids (list/None),
               force_regenerate (bool): 强制重新生成场景配置（用于评估）,
               enable_contact_sensors (bool): 启用接触传感器（用于力监测）
        [输出]: spawned_objects (List[RigidObject])
        """
        import os, random, torch, numpy as np
        from isaaclab.utils.math import quat_from_euler_xyz
        from pxr import Gf

        # 默认工作空间限制 (参考 Robot 类)
        if workspace_limits is None:
            # [min_x, min_y, min_z], [max_x, max_y, max_z]
            # 注意: z 固定为 0.15 (用户要求生成高度)
            workspace_limits = torch.tensor([
                [0.50, -0.30, 0.06], 
                [1.00, 0.30, 0.10]
            ])
        
        # [新增] 保存为实例属性，供 env_wrapper 使用
        self.workspace_limits = workspace_limits
        
        # [新增] 保存接触传感器开关状态
        self._enable_contact_sensors = enable_contact_sensors
        
        # [新增] 如果强制重新生成，先删除全局缓存
        if force_regenerate and hasattr(self, '_global_spawn_config'):
            del self._global_spawn_config
            print("  [场景随机化] 清除缓存，将生成全新场景配置")

        
        # Meshdata 数据集路径
        ycb_root = "/home/k/Projects/equi/IsaacLab/scripts/workspace/meshdata"
        ycb_target_root = "/home/k/Projects/equi/IsaacLab/scripts/workspace/meshdata_target"
        
        # 查找所有包含 textured.usd 的模型
        get_models = lambda root: [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "textured.usd"))] if os.path.exists(root) else []
        available_models = get_models(ycb_root)
        available_target_models = get_models(ycb_target_root)
        
        if not available_models or not available_target_models: 
            return []
        
        # === 内部函数:为指定环境生成物体 ===
        def _generate_objects_for_envs(env_ids_to_generate):
            """为指定的环境ID列表生成物体"""
            spawned_objects_local = []
            
            for env_idx in env_ids_to_generate:
                # ========== [固定位置模式] ==========
                # 所有环境共享同一个位置配置，仅角度随机
                
                # 检查是否已有全局缓存配置（所有环境共用）
                if not hasattr(self, '_global_spawn_config'):
                    # 首次生成全局配置（只生成一次）
                    n_total = num_objects_range if isinstance(num_objects_range, int) else random.randint(*num_objects_range)
                    n_obstacles = n_total - 1
                    existing_objects = []
                    
                    ws_arg = [[workspace_limits[0, i].item(), workspace_limits[1, i].item()] for i in range(3)]
                    
                    # 随机选择模型
                    target_model_name = random.choice(available_target_models)
                    available_obstacle_models = [m for m in available_models if m != target_model_name]
                    random.shuffle(available_obstacle_models)
                    selected_obstacle_models = available_obstacle_models[:n_obstacles]
                    
                    # 目标物体固定位置
                    target_pos = [0.7, 0.0, 0.06]
                    existing_objects.append((target_pos, 0.05))
                    
                    # 为每个障碍物生成固定位置
                    obstacle_positions = []
                    for obj_idx in range(n_obstacles):
                        found_pos = self.find_safe_positions(
                            objects=existing_objects,
                            candidate_radius=0.04,
                            workspace=ws_arg,
                            min_dist=0.05,
                            max_dist=0.18,  # 让障碍物分散更开
                            num_positions=1,
                            max_attempts=2000
                        )
                        if found_pos:
                            obstacle_positions.append(found_pos)
                            existing_objects.append((found_pos, 0.06))
                        else:
                            # 使用备用位置（8个物体需要7个障碍物位置）
                            backup_positions = [
                                [0.85, 0.15, 0.06],
                                [0.85, -0.15, 0.06],
                                [0.60, 0.18, 0.06],
                                [0.60, -0.18, 0.06],
                                [0.75, 0.25, 0.06],
                                [0.75, -0.25, 0.06],
                                [0.55, 0.0, 0.06],
                            ]
                            if obj_idx < len(backup_positions):
                                obstacle_positions.append(backup_positions[obj_idx])
                                existing_objects.append((backup_positions[obj_idx], 0.04))
                    
                    # 缓存全局配置（所有环境共用）
                    self._global_spawn_config = {
                        'target_model': target_model_name,
                        'target_pos': target_pos,
                        'obstacle_models': selected_obstacle_models[:len(obstacle_positions)],
                        'obstacle_positions': obstacle_positions
                    }
                    print(f"  [全局缓存] 生成固定配置: 目标={target_model_name}, 障碍物数={len(obstacle_positions)}")
                    print(f"            位置: 目标={target_pos}, 障碍物={obstacle_positions}")
                
                # 所有环境使用同一个全局配置
                config = self._global_spawn_config
                target_model_name = config['target_model']
                target_pos = config['target_pos']
                obstacle_models = config['obstacle_models']
                obstacle_positions = config['obstacle_positions']
                
                # --- 生成目标物体（缓存位置，随机角度） ---
                target_usd_path = os.path.join(ycb_target_root, target_model_name, "textured.usd")
                target_yaw_deg = random.choice([0, 90, 180, 270])
                target_yaw = np.deg2rad(target_yaw_deg)
                quat = quat_from_euler_xyz(
                    roll=torch.tensor(0.0), 
                    pitch=torch.tensor(0.0), 
                    yaw=torch.tensor(target_yaw)
                )
                
                if self.num_envs > 1:
                    prim_path = f"/World/Scene_{env_idx}/Scene/Target_{target_model_name}"
                else:
                    prim_path = f"/World/Scene/Target_{target_model_name}"
                
                target_obj = self._spawn_object(
                    name=f"Target_{target_model_name}",
                    usd_path=target_usd_path,
                    pos=target_pos,
                    quat=quat,
                    prim_path=prim_path
                )
                spawned_objects_local.append(target_obj)

                # --- 生成障碍物（缓存位置，随机角度） ---
                for obj_idx, (model_name, pos) in enumerate(zip(obstacle_models, obstacle_positions)):
                    usd_path = os.path.join(ycb_root, model_name, "textured.usd")
                    
                    yaw_deg = random.choice([0, 90, 180, 270])
                    yaw = np.deg2rad(yaw_deg)
                    quat = quat_from_euler_xyz(
                        roll=torch.tensor(0.0), 
                        pitch=torch.tensor(0.0), 
                        yaw=torch.tensor(yaw)
                    )
                    
                    if self.num_envs > 1:
                        prim_path = f"/World/Scene_{env_idx}/Scene/Obj_{obj_idx}_{model_name}"
                    else:
                        prim_path = f"/World/Scene/Obj_{obj_idx}_{model_name}"
                    
                    obj = self._spawn_object(
                        name=f"Obj_{obj_idx}_{model_name}",
                        usd_path=usd_path,
                        pos=pos,
                        quat=quat,
                        prim_path=prim_path
                    )
                    spawned_objects_local.append(obj)
                # ========== [固定位置模式] 结束 ==========
                
                # ========== [原随机生成逻辑 - 已注释] 开始 ==========
                # 以下代码保留用于恢复随机生成模式
                # n_total = num_objects_range if isinstance(num_objects_range, int) else random.randint(*num_objects_range)
                # n_obstacles = n_total - 1  # 1个target + n_obstacles个障碍物
                # existing_objects = []
                # 
                # # 准备 workspace 参数
                # ws_arg = [[workspace_limits[0, i].item(), workspace_limits[1, i].item()] for i in range(3)]
                #
                # # === 确保所有物体使用不同模型 ===
                # # 1. 随机选择一个目标模型
                # target_model_name = random.choice(available_target_models)
                # 
                # # 2. 从障碍物池中移除目标模型,确保不重复
                # available_obstacle_models = [m for m in available_models if m != target_model_name]
                # 
                # # 3. 随机打乱并选择前n_obstacles个(确保障碍物之间也不重复)
                # random.shuffle(available_obstacle_models)
                # selected_obstacle_models = available_obstacle_models[:n_obstacles]
                #
                # # --- 生成目标物体 ---
                # target_usd_path = os.path.join(ycb_target_root, target_model_name, "textured.usd")
                # 
                # # 为目标物体寻找位置
                # target_pos = None
                # target_radius = 0.05
                # 
                # # 最先生成的不需要找安全位置
                # found_target_pos = [0.75,0,0.08]
                # if found_target_pos:
                #     target_pos = found_target_pos
                #     existing_objects.append((target_pos, target_radius))
                #     
                #     # 随机姿态
                #     roll = random.uniform(-np.pi, np.pi)
                #     pitch = random.uniform(-np.pi, np.pi)
                #     yaw = random.uniform(-np.pi, np.pi)
                #     euler = torch.tensor([roll, pitch, yaw])
                #     quat = quat_from_euler_xyz(euler[0], euler[1], euler[2])
                #     
                #     # 构造 Prim 路径 (目标物体)
                #     if self.num_envs > 1:
                #         prim_path = f"/World/Scene_{env_idx}/Scene/Target_{target_model_name}"
                #     else:
                #         prim_path = f"/World/Scene/Target_{target_model_name}"
                #     
                #     # 加载目标对象
                #     target_obj = self._spawn_object(
                #         name=f"Target_{target_model_name}",
                #         usd_path=target_usd_path,
                #         pos=target_pos,
                #         quat=quat,
                #         prim_path=prim_path
                #     )
                #     spawned_objects_local.append(target_obj)
                # else:
                #     print(f"  警告: 无法为环境 {env_idx} 的目标物体找到有效位置。")
                #
                # # --- 生成障碍物 ---
                # for obj_idx, model_name in enumerate(selected_obstacle_models):
                #     usd_path = os.path.join(ycb_root, model_name, "textured.usd")
                #     
                #     # 使用 find_safe_positions 生成位置
                #     pos = None
                #     current_radius = 0.04 # 假设半径 5cm
                #     
                #     # 调用 find_safe_positions
                #     found_pos = self.find_safe_positions(
                #         objects=existing_objects,
                #         candidate_radius=current_radius,
                #         workspace=ws_arg,
                #         min_dist=0.05,
                #         max_dist=0.12,
                #         num_positions=1,
                #         max_attempts=1000
                #     )
                #     
                #     if not found_pos:
                #         continue
                #     
                #     if found_pos:
                #         pos = found_pos
                #         existing_objects.append((pos, current_radius))
                #     
                #     # 随机生成姿态 (Euler -> Quat)
                #     roll = random.uniform(-np.pi, np.pi)
                #     pitch = random.uniform(-np.pi, np.pi)
                #     yaw = random.uniform(-np.pi, np.pi)
                #     
                #     euler = torch.tensor([roll, pitch, yaw])
                #     quat = quat_from_euler_xyz(euler[0], euler[1], euler[2])
                #     
                #     # 构造 Prim 路径 (障碍物)
                #     if self.num_envs > 1:
                #         prim_path = f"/World/Scene_{env_idx}/Scene/Obj_{obj_idx}_{model_name}"
                #     else:
                #         prim_path = f"/World/Scene/Obj_{obj_idx}_{model_name}"
                #     
                #     # 加载对象 (使用USD原生材质)
                #     obj = self._spawn_object(
                #         name=f"Obj_{obj_idx}_{model_name}",
                #         usd_path=usd_path,
                #         pos=pos,
                #         quat=quat,
                #         prim_path=prim_path
                #     )
                #     spawned_objects_local.append(obj)
                # ========== [原随机生成逻辑 - 已注释] 结束 ==========
            
            # 内部函数执行结束,返回生成的物体列表
            return spawned_objects_local
        
        # === 主逻辑:向量化验证+部分环境重试 ===
        all_spawned_objects = []
        
        # 确定需要生成的环境ID集合
        if env_ids is None:
            pending_envs = set(range(self.num_envs))
        else:
            pending_envs = set(env_ids)
            
        env_attempts = {i: 0 for i in pending_envs}  # 每个环境的尝试次数
        max_retries = 5
        
        iteration = 0
        while pending_envs and iteration < max_retries * self.num_envs:
            iteration += 1
            
            # 1. 为待处理环境生成物体
            new_objects = _generate_objects_for_envs(pending_envs)
            all_spawned_objects.extend(new_objects)
            
            # 2. 聚拢
            # 如果是第一次迭代，重置整个仿真；后续迭代只聚拢新物体，不重置以保护旧环境
            is_first_iteration = (iteration == 1)
            
            if is_first_iteration:
                self._gather_objects(all_spawned_objects, reset_sim=True)
            else:
                successful_envs = [eid for eid in range(self.num_envs) if eid not in pending_envs]
                self.sim.reset()
                if successful_envs:
                    self._restore_cached_positions(all_spawned_objects, successful_envs)
                self._gather_objects(new_objects, reset_sim=False)
            
            # 3. 向量化验证
            env_results = self._check_objects_in_workspace(all_spawned_objects, workspace_limits)
            
            failed_envs = [eid for eid in pending_envs if not env_results.get(eid, False)]
            passed_envs = [eid for eid in pending_envs if env_results.get(eid, False)]
            
            if passed_envs and iteration > 1:
                previously_successful = [eid for eid in range(self.num_envs) 
                                       if eid not in pending_envs and eid not in passed_envs]
                if previously_successful:
                    self._gentle_gather_for_compensation(all_spawned_objects, previously_successful)
            
            if not failed_envs:
                # print(f"\n✓ 所有环境生成成功!\\n")
                pending_envs.clear()
                break
            
            # 5. 更新尝试次数并过滤
            envs_to_retry = []
            for eid in failed_envs:
                env_attempts[eid] += 1
                if env_attempts[eid] < max_retries:
                    envs_to_retry.append(eid)
                else:
                    print(f"  ⚠ Env {eid} 达到最大重试次数 ({max_retries}),放弃")
            
            if envs_to_retry:
                # 6. 删除失败环境的物体
                print(f"  [环境状态] 验证失败，正在重置并重试...")
                self._delete_objects(all_spawned_objects, env_ids_to_delete=failed_envs)
                # 注意: 删除物体后tensor视图会失效,不能调用step()
            
            # 7. 更新待处理环境
            pending_envs = set(envs_to_retry)
        
        # 最终检查
        if pending_envs:
            print(f"\n{'='*60}")
            print(f"⚠ 警告: 环境 {sorted(pending_envs)} 在最大迭代后仍未通过验证")
            print(f"{'='*60}\n")
        
        return all_spawned_objects

    def get_object_poses(self, objects):
        """
        [功能]: 获取列表中所有物体的当前位置和姿态。
        
        [输入]: objects (list): RigidObject 实例列表 (由 create_clutter_environment 返回)。
            
        [输出]: list: 包含每个物体状态的字典列表。
                每个字典包含:
                - 'name': 物体名称
                - 'position': 位置 [x, y, z] (Tensor)
                - 'orientation': 姿态四元数 [w, x, y, z] (Tensor)
        """
        return [{"name": obj.cfg.prim_path.split("/")[-1], "position": obj.data.root_pos_w[0], "orientation": obj.data.root_quat_w[0]} 
                for obj in objects if obj.data.root_pos_w is not None]

    def get_all_object_poses(self, objects):
        """
        获取所有物体（包括目标物体和障碍物）的姿态。
        """
        return self.get_object_poses(objects)

    def get_target_pose(self, objects):
        """
        仅获取目标物体的姿态。
        """
        all_poses = self.get_object_poses(objects)
        # 过滤名称中包含 "Target_" 的物体
        return [p for p in all_poses if "Target_" in p['name']]

    def get_obstacle_poses(self, objects):
        """
        获取除目标物体以外的所有障碍物的姿态。
        """
        all_poses = self.get_object_poses(objects)
        # 过滤名称中不包含 "Target_" 的物体
        return [p for p in all_poses if "Target_" not in p['name']]

    def get_env_offset(self, env_idx):
        """
        [功能]: 获取指定环境的全局偏移量
        [输入]: env_idx (int): 环境索引
        [输出]: (x_offset, y_offset): 环境在世界坐标系中的偏移（米）
        """
        import math
        if self.num_envs == 1:
            return 0.0, 0.0
        
        grid_width = int(math.ceil(math.sqrt(self.num_envs)))
        row = env_idx // grid_width
        col = env_idx % grid_width
        return row * self.env_spacing, col * self.env_spacing
