import argparse
from typing import Optional, TYPE_CHECKING
from config import initialize_app, configure_simulation
from project_paths import project_path, resolve_project_path

if TYPE_CHECKING:
    from robot import Robot

class Scene:
    """
    管理仿真场景和应用程序生命周期的类。
    """
    def __init__(self, description="Isaac Lab Scene", num_envs=1, env_spacing=3.0):
        """
        [功能]: 初始化 Scene 类。
        [输入]: description (str): 应用程序的描述信息。
                num_envs (int): 并行环境数量。
                env_spacing (float): 环境间距 (米)。
        """
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        # 动态 USD 拓扑更新期间禁止任何物理步或 tensor 数据访问。该标志在
        # STOP 前置位，只在 reset + contact-view rebuild 全部成功后清除。
        self._topology_update_pending = False
        # 静态覆盖层在 stage 打开前为左右 inner/outer finger 补齐 ContactReportAPI。
        # 原始 mesh/env.usd 不修改；运行期间也不再改 finger schema，保护 instance proto。
        self.usd_path = project_path("assets", "env_contact_report.usda")
        
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

        # 先在空 Stage 上初始化 SimulationContext/Fabric，再组合场景资产。
        # 原顺序是 open_stage(env) -> SimulationContext；4.5 会复用 env 自带的
        # /physicsScene，并对 Hydra 已填充的整套材质 Sprim 做二次销毁/重建，
        # 从而触发 _MarkSprimDirty changeTracker 竞态。
        self.stage_utils.create_new_stage()
        self.sim = configure_simulation(self.app_launcher)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("创建空 USD Stage 失败。")

        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        if self.num_envs <= 1:
            if not world.GetReferences().AddReference(self.usd_path):
                raise RuntimeError(f"场景 USD 引用失败: {self.usd_path}")
            if not stage.GetPrimAtPath("/World/Scene").IsValid():
                raise RuntimeError(f"场景 USD 缺少 /World/Scene: {self.usd_path}")
        else:
            grid_width = int(math.ceil(math.sqrt(self.num_envs)))
            
            for i in range(self.num_envs):
                x, y = (i // grid_width) * self.env_spacing, (i % grid_width) * self.env_spacing
                env_path = f"/World/Scene_{i}"
                UsdGeom.Xformable(stage.DefinePrim(env_path, "Xform")).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
                stage.GetPrimAtPath(env_path).GetReferences().AddReference(self.usd_path)
                
                if i > 0 and (g := stage.GetPrimAtPath(f"{env_path}/{self.env_paths['Ground']}")).IsValid():
                    g.SetActive(False)

        print("[Scene] Stage loaded. Ready for camera creation.")

    def start_simulation(self):
        """
        阶段2：启动物理引擎和初始化机器人
        支持多场景：为每个场景创建独立的Camera和State
        """
        if not self.sim:
            raise RuntimeError("Must call load_stage() before start_simulation()")

        # === 四指接触追踪：PhysX 成对接触数据（Isaac Lab 2.1.1 兼容） ===
        # 不创建 Isaac Lab ContactSensor，也不在运行期修改 instanceable 几何。
        # 环境变量 DISABLE_GRIPPER_CONTACT=1 可完全关闭接触追踪。
        import os as _os
        if _os.environ.get("DISABLE_GRIPPER_CONTACT", "0") == "1":
            self.contact_tracker = None
            print("[Scene] DISABLE_GRIPPER_CONTACT=1 → 跳过夹爪接触追踪")
        else:
            from physx_contact_report import PhysXContactReportTracker
            self.contact_tracker = PhysXContactReportTracker(
                gripper_prim_path=self.gripper_prim_path,
                num_envs=self.num_envs,
            )

        # 在创建 Camera/RenderProduct 前只验证静态 API；此调用不写 USD stage。
        if self.contact_tracker is not None:
            self.contact_tracker.activate_api()

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
        if self._topology_update_pending:
            raise RuntimeError(
                "USD 拓扑更新尚未完成，禁止在 tensor handles 重建前执行物理步。"
            )
        if not self.sim:
            return

        # 单环境和多环境使用同一条控制生命周期。每个 Robot 都是
        # 一个独立的单实例 Articulation，因此不能只更新单环境便捷引用
        # self.robot。这里统一 write -> 一次物理步 -> update，调用方不得
        # 再用伪造的 dt 重复更新 asset 时间戳。
        robots = list(getattr(self, "robots", None) or [])
        for robot in robots:
            robot.write()

        self.sim.step()

        physics_dt = float(self.sim.get_physics_dt())
        for robot in robots:
            robot.update(physics_dt)

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

    def close(self):
        """兼容 Isaac Lab 2.1.1，在无桌面服务器上安全关闭仿真。"""
        if self.sim is not None:
            # STOP 事件触发后继续 render 会让 Replicator 的关闭流程死循环。
            self.sim._disable_app_control_on_stop_handle = True
        # 本项目只读取 Camera annotator，不运行需要等待写完的 Replicator writer。
        self.simulation_app.close(wait_for_replicator=False)

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
        import os
        from isaaclab.assets import RigidObject, RigidObjectCfg
        import isaaclab.sim as sim_utils

        # 保留 wrapper 替换前的原始模型路径。空推旋转检测需要从同目录
        # textured.obj 读取真实网格顶点；071 的仿真 USD 会在下方替换成 wrapper。
        source_usd_path = usd_path

        # 071 是唯一“引用根为容器、真实刚体在 Root/textured”的模型。
        # 静态 wrapper 直接把真实刚体作为 default prim；保留质量和视觉内容，
        # 用离线低面数凸包替代异常高面数碰撞网格，使根路径可作为精确 filter。
        is_model_071 = os.path.basename(os.path.dirname(usd_path)) == "071"
        if is_model_071:
            wrapper_name = (
                "ycb_071_target_flat.usda"
                if "meshdata_target" in usd_path
                else "ycb_071_ch_flat.usda"
            )
            usd_path = project_path("assets", wrapper_name)

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
            
        obj_cfg = RigidObjectCfg(
            prim_path=prim_path_pattern,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(0.01, 0.01, 0.01),  # 缩小100倍
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                ),
                # 071 wrapper 已在低面数 contact_collision 上静态配置碰撞。
                # 此处若递归覆盖 collision_props，UsdFileCfg 会重新给 152 万面
                # 的视觉网格应用 CollisionAPI，导致 PhysX 在 reset 时长时间烹饪。
                collision_props=(
                    None
                    if is_model_071
                    else sim_utils.CollisionPropertiesCfg(collision_enabled=True)
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot if init_rot is not None else [1.0, 0.0, 0.0, 0.0]
            ),
        )
        
        obj = RigidObject(obj_cfg)
        obj.source_usd_path = source_usd_path
        
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

    def _gather_objects(self, spawned_objects, reset_sim=True, views_ready=False):
        """
        [功能]: 聚集所有已生成的物体。
        [输入]: spawned_objects (list): 已生成的物体列表。
                reset_sim (bool, optional): 是否重置模拟器。默认为 True。
                views_ready (bool, optional): 调用方是否已经完成注册、reset 和
                    tensor-view 重建。仅用于多环境失败重试路径。
        """
        import torch
        if not spawned_objects or self.sim is None:
            return

        # 注册物体根/刚体路径，随后把 PhysX 成对接触列精确映射到三位模型 ID。
        tracker = getattr(self, 'contact_tracker', None)
        if tracker is not None and not views_ready:
            tracker.register_objects(spawned_objects)

        if reset_sim:
            self._reset_simulation_and_rebuild_views("物体生成")

            for _ in range(50):
                self.step()
        else:
            # 重试路径在调用本方法前已经执行过 sim.reset()。
            if tracker is not None and not views_ready:
                tracker.rebuild_tensor_views()
            print(f"  [初始化] 让新物体稳定 20 步...")
            for _ in range(20):
                self.step()

        # 按环境分组物体
        env_objs = {}
        for obj in spawned_objects:
            if "Scene_" in obj.cfg.prim_path:
                eid = int(obj.cfg.prim_path.split("Scene_")[1].split("/")[0])
            else:
                eid = 0
            env_objs.setdefault(eid, []).append(obj)
        
        # 聚拢方向改为朝着目标物体中心，而不是所有物体的几何中心
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
        
        gather_steps = 48
        gather_strength = 7.5  # 降低聚拢力，避免物体挤压后崩飞
        
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

    def _begin_topology_update(self, object_count):
        """停止物理并使所有 Isaac Lab/tensor handles 进入可安全删除状态。"""
        if object_count <= 0:
            return
        if self.sim is None:
            raise RuntimeError("无法开始 USD 拓扑更新：SimulationContext 尚未创建。")

        self._topology_update_pending = True
        # Isaac Lab 在非 terminal 启动方式下会响应 STOP 并进入渲染等待；拓扑
        # 事务必须临时禁止该行为，直到 PLAY/reset 恢复全部物理 handles。
        self.sim._disable_app_control_on_stop_handle = True
        try:
            if not self.sim.is_stopped():
                self.sim.stop()
            if not self.sim.is_stopped():
                raise RuntimeError("timeline.stop() 返回后仿真仍未进入 stopped 状态。")
        except Exception as exc:
            self._topology_update_pending = False
            self.sim._disable_app_control_on_stop_handle = False
            raise RuntimeError(
                f"停止物理以删除 {object_count} 个 USD 物体失败。"
            ) from exc

    def _reset_simulation_and_rebuild_views(self, phase):
        """统一 PLAY/reset Isaac Lab handles，并重建全部 GPU 接触视图。"""
        if self.sim is None:
            raise RuntimeError(f"{phase} reset 失败：SimulationContext 尚未创建。")

        try:
            self.sim.reset()
        except Exception as exc:
            # 保持 pending=True；调用方不得在半初始化状态继续 step/tensor 查询。
            self._topology_update_pending = True
            raise RuntimeError(f"{phase} 后重建 PhysX/Isaac Lab handles 失败。") from exc

        if not self.sim.is_playing():
            self._topology_update_pending = True
            raise RuntimeError(f"{phase} reset 返回后 timeline 未处于 playing 状态。")

        try:
            if self.robots:
                for robot in self.robots:
                    robot.reset()

            tracker = getattr(self, "contact_tracker", None)
            if tracker is not None:
                rebuilt_count = tracker.rebuild_tensor_views()
                if rebuilt_count != self.num_envs:
                    raise RuntimeError(
                        "GPU 接触视图重建数量错误："
                        f"期望 {self.num_envs}，实际 {rebuilt_count}。"
                    )
        except Exception as exc:
            self._topology_update_pending = True
            raise RuntimeError(f"{phase} 后重建 robot/contact tensor views 失败。") from exc

        self._topology_update_pending = False
        # SimulationContext.reset() 正常返回时也会清除此标志；这里显式恢复，
        # 兼容已经 stopped 后直接 PLAY 的路径。
        self.sim._disable_app_control_on_stop_handle = False

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

        if not objects_to_remove:
            return

        # STOP 会通过 Isaac Lab 正式 timeline 回调使 RigidObject/Articulation
        # handles 失效。必须在它完成后，才能删除这些 handles 曾引用的 prim。
        self._begin_topology_update(len(objects_to_remove))

        tracker = getattr(self, "contact_tracker", None)
        if tracker is not None:
            tracker.unregister_objects(objects_to_remove)

        for obj in objects_to_remove:
            prim_path = obj.cfg.prim_path
            try:
                if stage.GetPrimAtPath(prim_path).IsValid():
                    stage.RemovePrim(prim_path)
            except Exception as exc:
                raise RuntimeError(
                    f"物理停止后删除 USD prim 失败: {prim_path}"
                ) from exc
            spawned_objects.remove(obj)

    def create_clutter_environment(self, num_objects_range, workspace_limits=None, env_ids=None, force_task_config=None):
        """
        [功能]: 在指定环境中随机生成杂乱物体
        [输入]: num_objects_range (int/tuple), workspace_limits (Tensor), env_ids (list/None)
                force_task_config (dict/None): 强制设定的任务配置，可包含 obstacle_count，
                或包含 target_pos 和 obstacle_positions 以复用已有布局
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

        
        # Meshdata 数据集路径
        # [MAML] 如果 force_task_config 指定了模型文件夹，优先使用
        if force_task_config is not None and 'obstacle_model_dir' in force_task_config:
            ycb_root = resolve_project_path(force_task_config['obstacle_model_dir'])
        else:
            ycb_root = resolve_project_path("meshdata/meshdata_CH")
        
        if force_task_config is not None and 'target_model_dir' in force_task_config:
            ycb_target_root = resolve_project_path(force_task_config['target_model_dir'])
        else:
            ycb_target_root = resolve_project_path("meshdata/meshdata_target")
        
        # 查找所有包含 textured.usd 的模型
        get_models = lambda root: [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "textured.usd"))] if os.path.exists(root) else []
        available_models = get_models(ycb_root)
        available_target_models = get_models(ycb_target_root)
        
        if not available_models or not available_target_models: 
            return []

        def _build_grid_candidates(workspace_limits, target_pos, count):
            """Build deterministic fallback positions that scale with count."""
            x_min, x_max = workspace_limits[0, 0].item(), workspace_limits[1, 0].item()
            y_min, y_max = workspace_limits[0, 1].item(), workspace_limits[1, 1].item()
            z = target_pos[2]
            grid_size = max(4, int(np.ceil(np.sqrt(max(count, 1) * 2))))
            xs = np.linspace(x_min, x_max, grid_size)
            ys = np.linspace(y_min, y_max, grid_size)
            candidates = [
                [round(float(x), 4), round(float(y), 4), z]
                for x in xs
                for y in ys
            ]
            return sorted(
                candidates,
                key=lambda p: (p[0] - target_pos[0]) ** 2 + (p[1] - target_pos[1]) ** 2
            )

        def _take_grid_candidate(candidates, existing_objects, min_center_dist=0.08):
            for candidate in candidates:
                if all(np.linalg.norm(np.array(candidate[:2]) - np.array(pos[:2])) >= min_center_dist
                       for pos, _ in existing_objects):
                    candidates.remove(candidate)
                    return candidate
            return None

        def _generate_scene_layout(n_obstacles, target_pos):
            existing_objects = [(target_pos, 0.05)]
            obstacle_positions = []
            ws_arg = [[workspace_limits[0, i].item(), workspace_limits[1, i].item()] for i in range(3)]
            grid_candidates = _build_grid_candidates(
                workspace_limits, target_pos, n_obstacles
            )

            for _ in range(n_obstacles):
                found_pos = self.find_safe_positions(
                    objects=existing_objects,
                    candidate_radius=0.05,
                    workspace=ws_arg,
                    min_dist=0.08,
                    max_dist=0.20,
                    num_positions=1,
                    max_attempts=2000
                )
                if found_pos:
                    obstacle_positions.append(found_pos)
                    existing_objects.append((found_pos, 0.06))
                else:
                    fallback_pos = _take_grid_candidate(grid_candidates, existing_objects)
                    if fallback_pos is not None:
                        obstacle_positions.append(fallback_pos)
                        existing_objects.append((fallback_pos, 0.04))

            while len(obstacle_positions) < n_obstacles:
                fallback_pos = _take_grid_candidate(grid_candidates, existing_objects)
                if fallback_pos is None:
                    raise RuntimeError(
                        f"请求生成障碍物 {n_obstacles} 个，"
                        f"但只生成了 {len(obstacle_positions)} 个障碍物位置"
                    )
                obstacle_positions.append(fallback_pos)
                existing_objects.append((fallback_pos, 0.04))

            if len(obstacle_positions) != n_obstacles:
                raise RuntimeError(
                    f"障碍物数量不一致: 期望 {n_obstacles}, "
                    f"实际 {len(obstacle_positions)}"
                )

            return obstacle_positions
        
        # === 内部函数:为指定环境生成物体 ===
        def _generate_objects_for_envs(env_ids_to_generate):
            """为指定的环境ID列表生成物体"""
            spawned_objects_local = []
            
            for env_idx in env_ids_to_generate:
                # ========== [固定位置模式] ==========
                # 所有环境共享同一个位置配置，仅角度随机
                
                # 检查是否已有全局缓存配置（所有环境共用）或传入了强制配置
                if force_task_config is not None:
                    # 使用传入的 MAML Task 配置
                    if 'obstacle_positions' not in force_task_config:
                        n_obstacles = force_task_config.get('obstacle_count')
                        if n_obstacles is None:
                            n_total = num_objects_range if isinstance(num_objects_range, int) else random.randint(*num_objects_range)
                            n_obstacles = n_total - 1
                        target_pos = force_task_config.get('target_pos', [0.75, 0.0, 0.06])
                        force_task_config['target_pos'] = target_pos
                        force_task_config['obstacle_count'] = n_obstacles
                        force_task_config['obstacle_positions'] = _generate_scene_layout(
                            n_obstacles, target_pos
                        )
                    else:
                        n_obstacles = len(force_task_config['obstacle_positions'])
                        force_task_config['obstacle_count'] = n_obstacles
                        force_task_config.setdefault('target_pos', [0.75, 0.0, 0.06])

                    requested_target_model = force_task_config.get('target_model')
                    if requested_target_model is not None:
                        target_model_name = str(requested_target_model)
                        if target_model_name not in available_target_models:
                            raise ValueError(
                                f"指定目标模型不存在或缺少 textured.usd: "
                                f"{os.path.join(ycb_target_root, target_model_name)}"
                            )
                    else:
                        target_model_name = random.choice(available_target_models)
                    available_obstacle_models = [m for m in available_models if m != target_model_name]
                    if len(available_obstacle_models) < n_obstacles:
                        raise RuntimeError(
                            f"请求生成 {n_obstacles} 个障碍物，"
                            f"但可用障碍物模型只有 {len(available_obstacle_models)} 个"
                        )
                    random.shuffle(available_obstacle_models)
                    selected_obstacle_models = available_obstacle_models[:n_obstacles]
                    
                    self._global_spawn_config = {
                        'target_model': target_model_name,
                        'target_pos': force_task_config['target_pos'],
                        'obstacle_models': selected_obstacle_models,
                        'obstacle_positions': force_task_config['obstacle_positions']
                    }
                elif not hasattr(self, '_global_spawn_config'):
                    # 首次生成全局配置（只生成一次）
                    n_total = num_objects_range if isinstance(num_objects_range, int) else random.randint(*num_objects_range)
                    n_obstacles = n_total - 1
                    
                    # 随机选择模型
                    target_model_name = random.choice(available_target_models)
                    available_obstacle_models = [m for m in available_models if m != target_model_name]
                    if len(available_obstacle_models) < n_obstacles:
                        raise RuntimeError(
                            f"请求生成 {n_obstacles} 个障碍物，"
                            f"但可用障碍物模型只有 {len(available_obstacle_models)} 个"
                        )
                    random.shuffle(available_obstacle_models)
                    selected_obstacle_models = available_obstacle_models[:n_obstacles]
                    
                    # 目标物体固定位置
                    target_pos = [0.75, 0.0, 0.06]
                    
                    # 为每个障碍物生成固定位置
                    obstacle_positions = _generate_scene_layout(n_obstacles, target_pos)
                    
                    # 缓存全局配置（所有环境共用）
                    self._global_spawn_config = {
                        'target_model': target_model_name,
                        'target_pos': target_pos,
                        'obstacle_models': selected_obstacle_models,
                        'obstacle_positions': obstacle_positions
                    }
                    print(
                        f"  [全局缓存] 生成固定配置: 目标={target_model_name}, "
                        f"目标数=1, 障碍物数={len(obstacle_positions)}, "
                        f"总物体数={1 + len(obstacle_positions)}"
                    )
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
                #         min_dist=0.07,
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
            try:
                new_objects = _generate_objects_for_envs(pending_envs)
            except Exception as exc:
                raise RuntimeError(
                    f"为环境 {sorted(pending_envs)} 生成 USD 物体失败。"
                ) from exc
            all_spawned_objects.extend(new_objects)
            
            # 2. 聚拢
            # 如果是第一次迭代，重置整个仿真；后续迭代只聚拢新物体，不重置以保护旧环境
            is_first_iteration = (iteration == 1)
            
            if is_first_iteration:
                self._gather_objects(all_spawned_objects, reset_sim=True)
            else:
                successful_envs = [eid for eid in range(self.num_envs) if eid not in pending_envs]
                tracker = getattr(self, 'contact_tracker', None)
                if tracker is not None:
                    # 先注册新路径，使统一 rebuild 能一次覆盖成功环境和重试环境。
                    tracker.register_objects(new_objects)
                self._reset_simulation_and_rebuild_views("环境生成重试")
                if successful_envs:
                    self._restore_cached_positions(all_spawned_objects, successful_envs)
                self._gather_objects(
                    new_objects,
                    reset_sim=False,
                    views_ready=True,
                )
            
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
