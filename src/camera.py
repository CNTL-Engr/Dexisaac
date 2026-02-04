import torch
import numpy as np
import omni.usd
import omni.kit.app
from pxr import UsdGeom
from isaaclab.sensors import Camera as IsaacCamera
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz
import cv2
import os

class Camera:
    def __init__(self, prim_path: str, exclude_prim_paths=None, height=480, width=640):
        """
        Initialize Camera configuration but defer actual camera creation.
        Camera sensor will be created when initialize() is called.
        """
        # Ensure parent Xform exists
        import isaacsim.core.utils.prims as prim_utils
        # Assuming prim_path ends with the camera prim name, we want the parent path.
        if "/World" in prim_path:
             parent_path = prim_path.rsplit("/", 1)[0]
             if parent_path and parent_path != "/World":
                  prim_utils.create_prim(parent_path, "Xform")
        
        # Euler (180, 0, -90) -> Quaternion
        rot_euler = torch.tensor([180.0 * np.pi / 180.0, 0.0, 90.0 * np.pi / 180.0])
        rotation_quat = quat_from_euler_xyz(rot_euler[0], rot_euler[1], rot_euler[2])
        
        self.cfg = CameraCfg(
            prim_path=prim_path,
            update_period=0,
            height=height,
            width=width,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_id_segmentation_fast",  # 实例分割
            ],
            colorize_instance_id_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.14756, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            # Position: 0.75, 0, 1. Orientation: 180, 0, -90
            offset=CameraCfg.OffsetCfg(pos=(0.75, 0.0, 1.0), rot=rotation_quat.tolist()),
        )
        self.exclude_prim_paths = exclude_prim_paths
        self.camera = None  # Will be created in initialize()
        self._exclude_imageables = []
        self._initialized = False

    def initialize(self):
        """Create the actual camera sensor. Should be called after sim.reset()."""
        if self._initialized:
            return
            
        # Create the Isaac Camera
        self.camera = IsaacCamera(self.cfg)
        
        # Set up exclusion prims
        stage = omni.usd.get_context().get_stage()
        self._exclude_imageables = []
        paths = [self.exclude_prim_paths] if isinstance(self.exclude_prim_paths, str) else (self.exclude_prim_paths or [])
        
        for path in paths:
            if path and (prim := stage.GetPrimAtPath(path)).IsValid():
                self._exclude_imageables.append(UsdGeom.Imageable(prim))
        
        self._initialized = True

    def setup_exclusions(self):
        """
        Set up or refresh exclusion prims for hiding.
        Call this AFTER robot prims are created.
        """
        stage = omni.usd.get_context().get_stage()
        self._exclude_imageables = []
        paths = [self.exclude_prim_paths] if isinstance(self.exclude_prim_paths, str) else (self.exclude_prim_paths or [])
        
        print(f"[Camera] Setting up exclusions for paths: {paths}")
        for path in paths:
            if path:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    self._exclude_imageables.append(UsdGeom.Imageable(prim))
                    print(f"[Camera] ✓ Added exclusion for: {path}")
                else:
                    print(f"[Camera] ✗ Prim not found: {path}")
        
        print(f"[Camera] Total exclusions: {len(self._exclude_imageables)}")

    def reset(self):
        """Reset the camera and initialize buffers."""
        self.camera.reset()
    def update(self, dt):
        self.camera.update(dt)
    def get_images(self, hide_robot=True):
        """
        [功能]: 捕获RGB, 深度和分割图像
        [输入]: hide_robot (bool) - 捕获时是否隐藏指定Prims
        [输出]: dict of images
        """
        from pxr import UsdGeom
        
        # print(f"[Camera] get_images called with hide_robot={hide_robot}")
        # print(f"[Camera] Available exclusions: {len(self._exclude_imageables)} items")
        
        if hide_robot:
            [img.MakeInvisible() for img in self._exclude_imageables]
        # CRITICAL: Update camera to render with robot hidden
        # Do NOT restore visibility yet!
        # print("[DEBUG] Triggering app update for rendering...")
        omni.kit.app.get_app().update()
        # print("[DEBUG] App update done. Updating camera data...")

        self.camera.update(dt=0.0)
        # print("[DEBUG] Camera update done. Fetching output...")
        
        # Get the output data WHILE robot is still hidden
        output = self.camera.data.output
        # print(f"[DEBUG] Camera Output Keys: {output.keys()}")
        
        # NOW restore visibility after we've captured the data
        if hide_robot:
            [img.MakeVisible() for img in self._exclude_imageables]
            
        return output

    def save_images(self, images, save_path):
        """
        [功能]: 保存图像
        [输入]: images (dict), save_path
        """
        import numpy as np
        import cv2
        import os
        import torch

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.cpu().numpy()
            return t

        for key, value in images.items():
            data_np = to_numpy(value)
            if data_np is None: 
                continue
            
            # 1. 处理 Batch 维度 (num_envs, H, W, C) -> (H, W, C)
            if data_np.ndim == 4:
                data_np = data_np[0] 
            
            print(f"[Camera] Saving {key}: shape={data_np.shape}, dtype={data_np.dtype}")
            
            # --- 分支 A: 处理 RGB 图像 ---
            if "rgb" in key:
                 # 确保范围在 [0, 255] 且为 uint8
                 if data_np.dtype == np.float32 or data_np.dtype == np.float64:
                     if data_np.max() <= 1.0:
                         data_np = (data_np * 255).astype(np.uint8)
                     else:
                         data_np = data_np.astype(np.uint8)
                 
                 # RGBA -> BGR (OpenCV)
                 if data_np.shape[-1] == 4:
                    bgr = cv2.cvtColor(data_np, cv2.COLOR_RGBA2BGR)
                 else:
                    bgr = cv2.cvtColor(data_np, cv2.COLOR_RGB2BGR)
                 cv2.imwrite(os.path.join(save_path, f"{key}.png"), bgr)
                 
            # --- 分支 B: 处理 实例分割 (Instance ID) ---
            elif "instance_id" in key:
                 # 1. 数据清洗：去除多余的通道维度 (H, W, 1) -> (H, W)
                 if data_np.ndim == 3 and data_np.shape[-1] == 1:
                     data_np = data_np.squeeze(-1)
                 elif data_np.ndim == 3 and data_np.shape[-1] == 4:
                     # 极其罕见的情况：如果 ID 被存成了 RGBA，取第一通道（通常 ID 只在 R 通道）
                     # 但通常 Isaac Lab 输出的是 int32 单通道
                     data_np = data_np[:, :, 0]

                 # 2. 核心：保存原始数据 (Int32)，绝对不要转 uint8
                 np.save(os.path.join(save_path, f"{key}.npy"), data_np)
                 
                 # 3. 统计真实 ID
                 unique_ids = np.unique(data_np)
                 print(f"[Camera] {key} 真实 ID 统计: {len(unique_ids)} 个 -> {unique_ids}")
                 
                 # 4. 生成高辨识度可视化图 (Hash Coloring)
                 # 创建一个彩色画布
                 h, w = data_np.shape
                 vis_img = np.zeros((h, w, 3), dtype=np.uint8)
                 
                 # 背景（ID=0）保持黑色，其他 ID 随机上色
                 # 使用 ID 作为随机种子，保证同一物体的颜色在不同帧是一样的
                 for uid in unique_ids:
                     if uid == 0: continue # 背景跳过
                     
                     # 简单的哈希算法生成颜色：
                     # 利用位移和异或让接近的 ID 生成完全不同的 RGB
                     np.random.seed(int(uid)) 
                     color = np.random.randint(50, 255, 3) # 随机生成亮色
                     
                     # 填色
                     vis_img[data_np == uid] = color
                 
                 cv2.imwrite(os.path.join(save_path, f"{key}_vis.png"), vis_img)

            # --- 分支 C: 处理 语义分割 (Semantic) 或其他 ---
            elif "segmentation" in key:
                 # 针对普通语义分割的处理
                 np.save(os.path.join(save_path, f"{key}.npy"), data_np)
                 
                 # 简单的可视化（如果需要）
                 if data_np.ndim == 3: data_np = data_np[:,:,0]
                 vis_img = (data_np * (255 // (data_np.max() + 1))).astype(np.uint8)
                 cv2.imwrite(os.path.join(save_path, f"{key}_vis.png"), vis_img)

            # --- 分支 D: 处理 深度图 ---
            elif "distance_to_image_plane" in key or "depth" in key:
                 np.save(os.path.join(save_path, f"{key}.npy"), data_np)
                 if data_np.max() > 0:
                    # 归一化便于人眼观察
                    depth_vis = (data_np / data_np.max() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, f"{key}.png"), depth_vis)