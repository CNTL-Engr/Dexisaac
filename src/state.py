import torch
import numpy as np
import cv2
# from camera import Camera  # Camera now passed from outside
# from sam2_wrapper import SAM2Wrapper  # Commented out SAM2

class State:
    def __init__(self, camera=None, env_idx=0, env_origin=(0.0, 0.0)):
        """
        初始化State，接收外部传入的Camera实例
        Args:
            camera: Camera实例（已初始化）
            env_idx: 环境索引（用于区分多环境下的物体）
            env_origin: 环境原点 (x, y) 世界坐标
        """
        if camera is None:
            raise ValueError("Camera instance must be provided to State")
        self.camera = camera
        self.env_idx = env_idx
        self.env_origin = env_origin
        self._sam = None
        
    # @property
    # def sam(self):
    #     if self._sam is None:
    #         print("Initializing SAM2 model...")
    #         self._sam = SAM2Wrapper()
    #     return self._sam
        
        # Define workspace limits for normalization (adjust as needed)
        self.x_min, self.x_max = 0.375, 1.125
        self.y_min, self.y_max = -0.375, 0.375
        
        # Image resolution
        self.img_width = 320
        self.img_height = 320
    def update(self, dt):
        self.camera.update(dt)
    def reset(self):
        self.camera.reset()
    
    def get_state(self, spawned_objects):
        """
        获取用于神经网络输入的状态tensor
        
        Args:
            spawned_objects: 物体列表,用于提取目标掩膜
        
        Returns:
            torch.Tensor: (1, 3, 320, 320) tensor
                - Channel 0: 深度图 (归一化到[0,1])
                - Channel 1: 目标掩膜 (归一化到[0,1])
                - Channel 2: 障碍物掩膜 (归一化到[0,1])
        """
        import torch
        import numpy as np
        
        # 1. 获取原始图像
        rgb, depth, seg = self.get_img()
        
        if depth is None or seg is None:
            print("⚠ 警告: 无法获取图像数据")
            return None
        
        # 2. 提取目标掩膜
        target_mask = self.extract_target_mask(seg, spawned_objects)
        if target_mask is None:
            print("⚠ 警告: 无法提取目标掩膜,使用全零掩膜")
            target_mask = np.zeros_like(depth, dtype=np.uint8)
        
        # 3. 提取全局掩膜
        global_mask = self.extract_global_mask(seg, exclude_floor=True)
        if global_mask is None:
            print("⚠ 警告: 无法提取全局掩膜,使用全零掩膜")
            global_mask = np.zeros_like(depth, dtype=np.uint8)
        
        # 4. 计算障碍物掩膜 (已修改为全局掩膜)
        # 用户要求: Global Mask containing target mask
        # 之前逻辑: obstacle_mask = global_mask - target_mask
        # 现在逻辑: Channel 2 = global_mask (包含 target)
        obstacle_mask = global_mask.copy()
        # obstacle_mask[target_mask > 0] = 0  # [User Request] Keep target in global mask
        
        # 5. 转换为tensor (保持 uint8 [0, 255])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        depth_tensor = torch.from_numpy(depth).to(device, dtype=torch.uint8)
        target_tensor = torch.from_numpy(target_mask).to(device, dtype=torch.uint8)
        global_tensor = torch.from_numpy(obstacle_mask).to(device, dtype=torch.uint8)
        
        # 6. [关键] Stack 成 (3, 320, 320) - 输入始终是 3 通道
        # Ch0: Depth, Ch1: Target Mask, Ch2: Global Mask
        state_tensor = torch.stack([depth_tensor, target_tensor, global_tensor], dim=0)
        
        # 7. 添加batch维度: (1, 3, 320, 320)
        
        state_tensor = state_tensor.unsqueeze(0)
        
        # print(f"[get_state] 生成状态tensor: {state_tensor.shape}, device={state_tensor.device}, dtype={state_tensor.dtype}")
        
        return state_tensor
    def get_img(self, hide_robot=True, normalize_depth=True):
        """
        获取RGB、深度图和实例分割图，并统一resize到320x320
        Args:
            hide_robot: 是否隐藏机器人
            normalize_depth: 是否将深度图归一化到0-255 (False则返回米为单位的float深度)
        Returns: (rgb_320, depth_320, seg_320) 或 None
        """
        # 获取原始图像数据（包含segmentation）
        images = self.camera.get_images(hide_robot=True)
        if images is None:
            return None
        
        # 提取各个通道
        rgb = images.get('rgb')
        depth = images.get('distance_to_image_plane')
        seg = images.get('instance_id_segmentation_fast')
        
        if rgb is None or depth is None:
            return None

        # Convert to numpy and handle batch dimension
        rgb_img = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else rgb
        depth_img = depth.cpu().numpy() if hasattr(depth, 'cpu') else depth
        seg_img = seg.cpu().numpy() if seg is not None and hasattr(seg, 'cpu') else seg if seg is not None else None
        
        # Remove batch dimension if present
        if rgb_img.ndim == 4: rgb_img = rgb_img[0]
        if depth_img.ndim == 4: depth_img = depth_img[0]
        if seg_img is not None and seg_img.ndim == 4: seg_img = seg_img[0]
        
        # Process RGB
        if rgb_img.shape[-1] == 4: rgb_img = rgb_img[..., :3]
        if rgb_img.dtype == np.float32: rgb_img = (rgb_img * 255).astype(np.uint8)

        # Process Depth
        depth_img = depth_img.squeeze(-1)
        
        # Process Segmentation (remove channel dim if present)
        if seg_img is not None:
            if seg_img.ndim == 3 and seg_img.shape[-1] == 1:
                seg_img = seg_img.squeeze(-1)
            elif seg_img.ndim == 3 and seg_img.shape[-1] == 4:
                seg_img = seg_img[:, :, 0]  # Take first channel if RGBA

        # Crop logic based on depth histogram
        # Strategy: Find dominant depth values. Assuming Table and Floor are the two most frequent.
        # Table should be 'closer' (lower value in standard depth, but let's check relative values)
        # Based on analysis: Floor ~ 254, Table ~ 249. (Higher value = Further away)
        # So Table is min(Top2_Freq_Values).
        
        crop_rgb, crop_depth, crop_seg = rgb_img, depth_img, seg_img
        
        # Calculate unique value counts
        # We use a flattened view for efficiency
        # Only consider valid depth (>0) for histogram
        valid_depth = depth_img[depth_img > 0]
        if valid_depth.size > 0:
            vals, counts = np.unique(valid_depth, return_counts=True)
            
            # Need at least a few values to distinguish table from floor
            if len(vals) >= 2:
                # Sort by frequency (descending)
                sorted_indices = np.argsort(counts)[::-1]
                top_vals = vals[sorted_indices[:2]] # Top 2 most frequent
                
                # Identify Workspace (Table) vs Floor
                table_val = np.min(top_vals) 
                
                # Create mask for the table
                table_mask = (depth_img == table_val).astype(np.uint8)
                
                # specific fix: if table is not found or very small, fallback?
                if np.sum(table_mask) > 100:
                    # Find bounding box of the table
                    x, y, w, h = cv2.boundingRect(table_mask)
                    
                    # Make it square to avoid stretching
                    # We use the max dimension to ensure we cover the workspace
                    center_x = x + w / 2
                    center_y = y + h / 2
                    side_length = max(w, h)
                    
                    # Re-calculate square box coordinates
                    x_sq = int(center_x - side_length / 2)
                    y_sq = int(center_y - side_length / 2)
                    w_sq = h_sq = side_length
                    
                    # Re-calculate padding requirements
                    rows, cols = rgb_img.shape[:2]
                    pad_top = max(0, -y_sq)
                    pad_bottom = max(0, (y_sq + h_sq) - rows)
                    pad_left = max(0, -x_sq)
                    pad_right = max(0, (x_sq + w_sq) - cols)
                    
                    # If padding is needed, pad inputs first
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                         rgb_p = cv2.copyMakeBorder(rgb_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                         # Use max valid depth (likely floor) for depth pad
                         # [修复] 确保pad_depth_val是标量float，不是numpy array
                         if len(vals) > 0:
                             pad_depth_val = float(np.max(vals))  # 转换为Python float
                         else:
                             pad_depth_val = 10.0
                         depth_p = cv2.copyMakeBorder(depth_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_depth_val) 
                         seg_p = cv2.copyMakeBorder(seg_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0) if seg_img is not None else None
                         
                         # Adjust crop coords to padded image
                         x_cr = x_sq + pad_left
                         y_cr = y_sq + pad_top
                    else:
                         rgb_p, depth_p, seg_p = rgb_img, depth_img, seg_img
                         x_cr, y_cr = x_sq, y_sq
                         
                    # Crop
                    crop_rgb = rgb_p[y_cr:y_cr+h_sq, x_cr:x_cr+w_sq]
                    crop_depth = depth_p[y_cr:y_cr+h_sq, x_cr:x_cr+w_sq]
                    crop_seg = seg_p[y_cr:y_cr+h_sq, x_cr:x_cr+w_sq] if seg_p is not None else None

        # Resize all to 320x320
        resized_rgb = cv2.resize(crop_rgb, (320, 320), interpolation=cv2.INTER_AREA)
        resized_depth = cv2.resize(crop_depth, (320, 320), interpolation=cv2.INTER_NEAREST)
        resized_seg = cv2.resize(crop_seg, (320, 320), interpolation=cv2.INTER_NEAREST) if crop_seg is not None else None
        
        # [DEBUG] Print depth stats
        if not normalize_depth:
             # Only print for raw depth retrieval (used by Z-calc)
             d_min, d_max, d_mean = resized_depth.min(), resized_depth.max(), resized_depth.mean()
             if d_min < 0.5: # 异常近
                 print(f"[State DEBUG] Depth Low! Min: {d_min:.3f}, Max: {d_max:.3f}, Mean: {d_mean:.3f}")
                 print(f"  - Crop Coords: x={x_cr}, y={y_cr}, w={w_sq}, h={h_sq}")
                 print(f"  - Original Depth Min: {depth_img.min():.3f}, Max: {depth_img.max():.3f}")

        # Normalize depth
        if normalize_depth:
            
            d_min, d_max = resized_depth.min(), resized_depth.max()
            if d_max - d_min > 1e-3:
                resized_depth = ((resized_depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            else:
                resized_depth[:] = 127 # Middle gray if flat
                resized_depth = resized_depth.astype(np.uint8)

        return resized_rgb, resized_depth, resized_seg

    def extract_global_mask(self, seg_img, exclude_floor=True):
        """
        [功能]: 提取全局掩膜 - 所有物体的合并掩膜
        [输入]: seg_img (np.ndarray), exclude_floor (bool, 默认True)
        [输出]: 全局二值化掩膜 (np.ndarray (H,W) uint8 或 None)
        """
        import numpy as np
        
        if seg_img is None:
            print("⚠ 警告: segmentation图像为空")
            return None
        
        # 获取所有唯一ID
        unique_ids = np.unique(seg_img)
        # print(f"\n[全局掩膜] 检测到 {len(unique_ids)} 个唯一ID")
        
        # [关键修复] 使用像素数量识别背景，而不是最小 ID
        # 原因：Isaac Sim 的 ID 分配可能在 reset 后改变
        # 正确逻辑：背景（地板/桌面）通常占据最多像素
        if exclude_floor and len(unique_ids) > 0:
            # 统计每个 ID 的像素数量
            id_counts = [(id_val, np.sum(seg_img == id_val)) for id_val in unique_ids]
            # 选择像素数量最多的 ID 作为背景
            background_id = max(id_counts, key=lambda x: x[1])[0]
            unique_ids = unique_ids[unique_ids != background_id]
            # print(f"[全局掩膜] 排除背景ID={background_id} (像素数: {max(id_counts, key=lambda x: x[1])[1]}), 剩余 {len(unique_ids)} 个物体")
        
        # 创建全局掩膜:所有非背景像素=255
        global_mask = np.zeros_like(seg_img, dtype=np.uint8)
        for obj_id in unique_ids:
            global_mask[seg_img == obj_id] = 255
        
        # pixel_count = np.sum(global_mask > 0)
        # print(f"[全局掩膜] 总像素数: {pixel_count}")
        
        return global_mask
    
    def extract_target_mask(self, seg_img, spawned_objects):
        """
        [功能]: 从segmentation图像中提取目标物体(Target)的掩膜
        [输入]: seg_img (np.ndarray (H,W)), spawned_objects (list)
        [输出]: 目标物体的二值化掩膜 (np.ndarray (H,W) uint8 或 None)
        """
        import numpy as np
        
        if seg_img is None:
            print("⚠ 警告: segmentation图像为空")
            return None
        
        if not spawned_objects:
            print("⚠ 警告: 物体列表为空")
            return None
        
        # 1. 找到目标物体
        # 1. 找到目标物体
        target_obj = None
        for obj in spawned_objects:
            path = obj.cfg.prim_path
            
            # [Fix] 过滤非当前环境的物体
            if self.env_idx is not None:
                # 检查路径是否包含当前环境ID
                # 多环境: /World/Scene_{i}/...
                # 单环境: /World/Scene/... (等同于 Env 0)
                
                # Check for explicit Scene_{i} match
                if f"/Scene_{self.env_idx}/" in path:
                    pass # Match
                # Check for implicit Env 0 match (single env legacy)
                elif self.env_idx == 0 and "/Scene/" in path and "/Scene_" not in path:
                    pass # Match
                else:
                    continue # Not in this env
            
            obj_name = path.split('/')[-1]
            if "Target_" in obj_name:
                target_obj = obj
                # print(f"\n[目标掩膜] 找到目标物体: {obj_name}")
                break
        
        if target_obj is None:
            print(f"⚠ 警告[Env{self.env_idx}]: 未找到目标物体 (名称包含'Target_')")
            print(f"  - spawned_objects数量: {len(spawned_objects)}")
            if spawned_objects:
                print(f"  - 示例路径: {spawned_objects[0].cfg.prim_path}")
            return None
        
        # 2. 获取目标物体的semantic ID
        # Isaac Sim的segmentation使用semantic ID
        try:
            # 方法1: 尝试从prim获取semantic ID
            import omni.usd
            from pxr import UsdGeom
            
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(target_obj.cfg.prim_path)
            
            # 获取prim的所有子prim,找到带几何体的
            geom_prim = None
            if prim.IsValid():
                for child in prim.GetAllChildren():
                    if UsdGeom.Mesh(child):
                        geom_prim = child
                        break
                if geom_prim is None and UsdGeom.Mesh(prim):
                    geom_prim = prim
            
            # 方法2: 通过seg图像中的主要ID推断
            # 获取目标物体中心位置对应的seg ID
            # 注意: 物体应该在调用此方法前已经更新过状态
            pos_3d = target_obj.data.root_pos_w[0]  # 世界坐标
            
            # 转换为像素坐标
            u, v = self.world_to_pixel([pos_3d[0], pos_3d[1]])
            
            # 确保坐标在图像范围内
            h, w = seg_img.shape
            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)
            
            target_id = seg_img[v, u]
            # print(f"[目标掩膜] 目标物体中心像素({u}, {v})对应ID={target_id}")
            
            # 3. 创建目标掩膜
            target_mask = (seg_img == target_id).astype(np.uint8) * 255
            # pixel_count = np.sum(target_mask > 0)
            # print(f"[目标掩膜] 掩膜像素数: {pixel_count}")
            
            return target_mask
            
        except Exception as e:
            print(f"⚠ 提取目标掩膜时出错: {e}")
            return None

    def world_to_pixel(self, world_pos):
        """
        [功能]: 世界坐标转像素坐标 (320x320)
        [输入]: world_pos (Tensor/List) [x, y]
        [输出]: u, v (int)
        """
        WS_CENTER_X, WS_CENTER_Y = 0.75, 0.0
        PIXELS_PER_METER = 320 / 0.75
        
        x_world = world_pos[0].item() if isinstance(world_pos, torch.Tensor) else world_pos[0]
        y_world = world_pos[1].item() if isinstance(world_pos, torch.Tensor) else world_pos[1]
        
        # [Fix] 减去环境原点偏移，得到相对于该环境相机的局部坐标
        # 假设相机相对于环境原点是固定的
        # env_origin 默认为 (0,0), 如果多环境需要正确设置
        x_local = x_world - self.env_origin[0]
        y_local = y_world - self.env_origin[1]
            
        u = 160 + int((y_local - WS_CENTER_Y) * PIXELS_PER_METER)
        v = 160 + int((x_local - WS_CENTER_X) * PIXELS_PER_METER)
        return u, v
    
    def get_raw_global_mask(self, include_table=True):
        """
        [功能]: 获取原始未切割的全局掩膜二值图 (640x480)，基于深度图过滤远处地板
        [输入]: include_table (bool, 默认True)
        [输出]: 全局掩膜 (np.ndarray (480,640) uint8 或 None)
        """
        import numpy as np
        
        # 获取原始图像
        images = self.camera.get_images(hide_robot=True)
        if images is None:
            print("⚠ 警告: 无法获取原始图像")
            return None
        
        # 获取深度图
        depth = images.get('distance_to_image_plane')
        if depth is None:
            print("⚠ 警告: 无法获取深度图")
            return None
        
        # 转换为 numpy
        depth_img = depth.cpu().numpy() if hasattr(depth, 'cpu') else depth
        
        # 移除 batch 维度
        if depth_img.ndim == 4:
            depth_img = depth_img[0]
        
        # 移除通道维度
        if depth_img.ndim == 3:
            depth_img = depth_img.squeeze(-1)
        
        # 使用深度直方图找到桌面和地板的深度值
        vals, counts = np.unique(depth_img[depth_img > 0], return_counts=True)
        
        if len(vals) < 2:
            # 没有足够的数据，使用所有非零深度
            global_mask = np.zeros_like(depth_img, dtype=np.uint8)
            global_mask[depth_img > 0] = 255
            return global_mask
        
        # 按频率排序，取前2个
        sorted_indices = np.argsort(counts)[::-1]
        top_2_vals = vals[sorted_indices[:2]]
        
        # 桌面是较近的（深度值较小）
        table_depth = np.min(top_2_vals)
        floor_depth = np.max(top_2_vals)
        
        # 创建掩膜：只包含桌面及其以上的像素
        depth_threshold = (table_depth + floor_depth) / 2
        
        global_mask = np.zeros_like(depth_img, dtype=np.uint8)
        global_mask[(depth_img > 0) & (depth_img < depth_threshold)] = 255
        
        return global_mask
    
    def check_out_of_bounds(self, verbose=False):
        """
        [功能]: 检查物体是否出界，基于全局掩膜形状判断
        [输入]: verbose (bool)
        [输出]: (out_of_bounds: bool, info: dict)
        """
        import cv2
        import numpy as np
        
        # 获取原始全局掩膜（包含桌面）
        global_mask = self.get_raw_global_mask(include_table=True)
        
        if global_mask is None:
            if verbose:
                print("[出界检测] ⚠ 无法获取全局掩膜，默认判定为出界")
            return True, {'error': 'no_mask'}
        
        # 查找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            global_mask, connectivity=8
        )
        
        num_components = num_labels - 1
        
        # 判断1：连通域数量
        # 理想情况：只有 1 个连通域（桌面+所有物体）
        if num_components > 1:
            return True, {
                'num_components': num_components,
                'reason': 'multiple_components'
            }
        
        if num_components == 0:
            return True, {
                'num_components': 0,
                'reason': 'no_components'
            }
        
        main_label = 1
        main_area = stats[main_label, cv2.CC_STAT_AREA]
        main_x = stats[main_label, cv2.CC_STAT_LEFT]
        main_y = stats[main_label, cv2.CC_STAT_TOP]
        main_w = stats[main_label, cv2.CC_STAT_WIDTH]
        main_h = stats[main_label, cv2.CC_STAT_HEIGHT]
        
        width_height_diff = abs(main_w - main_h)
        
        if width_height_diff > 1:
            return True, {
                'num_components': num_components,
                'width': main_w,
                'height': main_h,
                'diff': width_height_diff,
                'reason': 'not_square'
            }
        
        margin = 10
        h, w = global_mask.shape
        touches_edge = (
            main_x < margin or
            main_y < margin or
            (main_x + main_w) > (w - margin) or
            (main_y + main_h) > (h - margin)
        )
        
        if touches_edge:
            return True, {
                'num_components': num_components,
                'reason': 'touches_edge',
                'bbox': (main_x, main_y, main_w, main_h)
            }
        
        # 所有检查通过
        
        return False, {
            'num_components': num_components,
            'width': main_w,
            'height': main_h,
            'reason': 'ok'
        }
