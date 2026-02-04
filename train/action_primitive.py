"""
辅助函数：根据离散动作索引计算推点和推动方向
"""
import cv2
import numpy as np
import torch

def compute_push_point_from_action(action_idx, env_idx, state, scene, spawned_objects):
    """
    根据离散动作索引计算推点和推动方向
    
    Args:
        action_idx: int (0-7)
            - 0-3: 推目标物体，从n*90°方向推
            - 4-7: 推障碍物，从目标中心朝(n-4)*90°方向推
        env_idx: 环境索引
        state: State对象
        scene: Scene对象
        spawned_objects: 当前环境中的物体列表
    
    Returns:
        push_point: (3,) tensor - 世界坐标推点 (x, y, z)
        direction_idx: int (0-3) - 推动方向索引
    """
    # [优化] 只获取一次图像，避免重复隐藏机械臂
    # 使用normalize_depth=False获取米制深度图，同时获取分割图
    images = state.get_img(hide_robot=True, normalize_depth=False)
    if images is None:
        raise ValueError(f"Env {env_idx}: 无法获取图像")
    
    rgb, depth_map, seg_map = images  # depth_map 是米制深度(float)
    
    # 使用state的方法获取正确的target_mask
    target_mask = state.extract_target_mask(seg_map, spawned_objects)
    
    # [Mod] 创建全局物体掩膜（通过投影物体中心点来识别前景ID）
    # 之前直接 seg_map > 0 会包含背景（如果背景ID非0），导致由于全白
    global_mask = np.zeros_like(seg_map, dtype=np.uint8)
    
    detected_ids = set()
    if spawned_objects:
        for obj in spawned_objects:
            try:
                # 获取物体位置 (tensor -> numpy)
                if not hasattr(obj, 'data') or obj.data.root_pos_w is None:
                    continue
                    
                pos_3d = obj.data.root_pos_w[0]
                if isinstance(pos_3d, torch.Tensor):
                    pos_3d = pos_3d.cpu().numpy()
                
                # 转换到像素坐标
                u, v = state.world_to_pixel([pos_3d[0], pos_3d[1]])
                
                # 边界检查
                h, w = seg_map.shape
                if 0 <= u < w and 0 <= v < h:
                    obj_id = seg_map[v, u]
                    # 只有当ID有效且未添加时处理
                    if obj_id > 0: # 假设0是空/无效
                        global_mask[seg_map == obj_id] = 255
                        detected_ids.add(obj_id)
            except Exception as e:
                # 忽略单个物体的错误
                continue
    
    # 如果没找到任何物体（可能是投影误差），尝试使用面积排除法作为Fallback
    if len(detected_ids) == 0:
        print(f"⚠ [Env{env_idx}] 无法通过投影识别物体，使用最大面积排除法作为Global Mask")
        vals, counts = np.unique(seg_map, return_counts=True)
        if len(vals) > 0:
            # 假设面积最大的ID是背景
            bg_id = vals[np.argmax(counts)]
            global_mask[seg_map != bg_id] = 255
        else:
            global_mask = (seg_map > 0).astype(np.uint8) * 255
            
    # 调试日志
    # print(f"[调试] Global Mask IDs: {detected_ids}, Sum: {global_mask.sum()}")
    
    # 如果Target Mask提取失败，使用Global Mask的一部分作为Fallback
    if target_mask is None or np.sum(target_mask > 0) == 0:
        print(f"❌ [Env{env_idx}] target_mask提取失败，使用中心区域Fallback")
        target_mask = np.zeros_like(global_mask)
        h, w = depth_map.shape
        # 如果 global mask 有内容，用从 global mask 里取一部分？ 
        # 暂时还是用中心矩形，稳妥
        target_mask[h//4:3*h//4, w//4:3*w//4] = 255
    else:
        # 确保mask是uint8
        if target_mask.dtype != np.uint8:
            target_mask = (target_mask * 255).astype(np.uint8) if target_mask.max() <= 1 else target_mask.astype(np.uint8)
            
    if global_mask.dtype != np.uint8:
        global_mask = (global_mask * 255).astype(np.uint8) if global_mask.max() <= 1 else global_mask.astype(np.uint8)
    
    # 获取目标中心
    target_coords = np.where(target_mask > 0)
    if len(target_coords[0]) == 0:
        raise ValueError(f"Env {env_idx}: 目标物体掩膜为空")
    
    center_v = int(np.mean(target_coords[0]))
    center_u = int(np.mean(target_coords[1]))
    center_depth = depth_map[center_v, center_u]
    
    # 像素到米的转换
    PPM = 320 / 0.75  # pixels per meter
    
    if action_idx <= 3:
        # ===== 推目标物体 =====
        direction_idx = action_idx
        
        # [Fix] 动作方向到图像角度的映射
        # u对应World Y, v对应World X
        # atan2(dy, dx) -> atan2(v_diff, u_diff) -> atan2(X_diff, Y_diff)
        # 0度(dx>0) -> Y+ (Act 1)
        # 90度(dy>0) -> X+ (Act 0)
        # 180度(dx<0) -> Y- (Act 3)
        # 270度(dy<0) -> X- (Act 2)
        
        if action_idx == 0:   # Push X+
            push_angle_deg = 90
        elif action_idx == 1: # Push Y+
            push_angle_deg = 0
        elif action_idx == 2: # Push X-
            push_angle_deg = 270
        elif action_idx == 3: # Push Y-
            push_angle_deg = 180
        else:
            push_angle_deg = 0 # Should not reach here
            
        # 1. 计算搜索弧形（反方向±15°）
        opposite_angle_deg = (push_angle_deg + 180) % 360
        angle_range = 15 # ±15度
        
        # 创建扇形mask
        h, w = target_mask.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        
        # 计算每个像素相对中心的角度
        dx = x_grid - center_u
        dy = y_grid - center_v
        angles = np.degrees(np.arctan2(dy, dx))  # -180 to 180
        
        # 标准化到0-360
        angles = (angles + 360) % 360
        
        # 创建扇形范围：opposite_angle_deg ± angle_range
        angle_min = (opposite_angle_deg - angle_range + 360) % 360
        angle_max = (opposite_angle_deg + angle_range) % 360
        
        if angle_min < angle_max:
            sector_mask = (angles >= angle_min) & (angles <= angle_max)
        else:  # 跨越0度
            sector_mask = (angles >= angle_min) | (angles <= angle_max)
        
        sector_mask = sector_mask.astype(np.uint8) * 255
        
        # 2. 膨胀目标掩膜3.5cm (扩大搜索范围)
        dilate_radius_px = int(0.04 * PPM)  
        kernel_dilate = np.ones((dilate_radius_px*2+1, dilate_radius_px*2+1), np.uint8)
        target_dilated = cv2.dilate(target_mask, kernel_dilate, iterations=1)
        
        # 3. 组合Mask: 扇形区域 & 膨胀掩膜
        search_mask = (target_dilated > 0) & (sector_mask > 0)
        
        if np.sum(search_mask) == 0:
            # Fallback: 全局扇形
            search_mask = sector_mask > 0
            
        # 4. 新逻辑：寻找安全推点（夹爪范围内无碰撞）
        # 条件：夹爪范围内所有像素的高度都必须低于夹爪下降高度
        
        # 定义夹爪尺寸 (像素)
        gripper_h_px = int(0.032 * PPM)
        gripper_w_px = int(0.02 * PPM)
        
        # 根据推动方向调整Kernel方向
        if push_angle_deg % 180 == 0: 
            # 0, 180 -> Horizontal Push -> Vertical Gripper
            k_rows, k_cols = gripper_h_px, gripper_w_px
        else:
            # 90, 270 -> Vertical Push -> Horizontal Gripper
            k_rows, k_cols = gripper_w_px, gripper_h_px
            
        kernel_roi = np.ones((k_rows, k_cols), np.uint8)
        
        # 计算每个像素作为推点时，夹爪范围内的最小深度（即最高点）
        # eroded_depth[v, u] = Min(depth) = 最高点高度对应的深度
        eroded_depth = cv2.erode(depth_map, kernel_roi, iterations=1)
        
        # 相机高度
        CAMERA_Z = 1.0
        # 安全余量：夹爪下降高度 = 最高点 + 1.5cm
        SAFETY_MARGIN = 0.015
        
        # 在搜索区域内寻找安全推点
        # 策略：找 eroded_depth 最大的点（即夹爪范围内最高点最低的位置）
        # 同时确保该点是"安全"的（下降高度不会太高导致碰撞）
        
        # 创建安全掩膜：过滤掉明显不安全的区域
        # 规则：如果一个点的夹爪范围内最高点高度 > 15cm，跳过
        max_allowed_height = 0.15  # 最大允许高度15cm
        min_allowed_depth = CAMERA_Z - max_allowed_height
        safe_mask = (eroded_depth >= min_allowed_depth) & search_mask
        
        # 在安全区域内找最优点
        masked_score = eroded_depth.copy()
        masked_score[~safe_mask] = -1.0  # 排除不安全区域
        
        # 找到最大值位置（夹爪范围内最高点最低的位置）
        best_idx = np.argmax(masked_score)
        push_v, push_u = np.unravel_index(best_idx, depth_map.shape)
        push_depth_val = eroded_depth[push_v, push_u]
        
        # 检查是否找到有效点
        if masked_score[push_v, push_u] == -1.0:
            # Fallback: 如果没有安全点，使用search_mask内eroded_depth最大的点
            masked_score_fallback = eroded_depth.copy()
            masked_score_fallback[~search_mask] = -1.0
            best_idx = np.argmax(masked_score_fallback)
            push_v, push_u = np.unravel_index(best_idx, depth_map.shape)
            push_depth_val = eroded_depth[push_v, push_u]
            
            if masked_score_fallback[push_v, push_u] == -1.0:
                # 最终Fallback: 使用中心
                push_u, push_v = center_u, center_v
                push_depth_val = center_depth
        
        # 5. 计算 push_z：确保不会碰撞
        # push_z = 夹爪范围内最高点高度 + 安全余量
        max_height_in_roi = CAMERA_Z - push_depth_val
        push_z = max_height_in_roi + SAFETY_MARGIN
        
        # 限制范围
        push_z = max(0.03, min(push_z, 0.15))
        
        # 用于debug打印
        push_depth = push_depth_val
        # print(f"[调试] Env{env_idx} 推目标: push_depth={push_depth:.3f}, max_h_roi={max_height_in_roi:.3f}, push_z={push_z:.3f}")
        
    else:
        # ===== 推障碍物 =====
        direction_idx = action_idx - 4
        
        # 推点为目标中心
        push_u, push_v = center_u, center_v
        push_depth = center_depth
        
        # 计算夹爪区域内的最高点
        gripper_h_px = int(0.032 * PPM)
        gripper_w_px = int(0.02 * PPM)
        
        h_map, w_map = depth_map.shape
        v_min = max(0, int(center_v - gripper_h_px // 2))
        v_max = min(h_map, int(center_v + gripper_h_px // 2))
        u_min = max(0, int(center_u - gripper_w_px // 2))
        u_max = min(w_map, int(center_u + gripper_w_px // 2))
        
        roi_depth = depth_map[v_min:v_max, u_min:u_max]
        
        if roi_depth.size > 0:
            min_depth_roi = np.min(roi_depth) # 最高点
        else:
            min_depth_roi = center_depth
            
        CAMERA_Z = 1.0
        max_height_in_roi = CAMERA_Z - min_depth_roi
        
        # 规则：以目标中心为夹爪范围中心，在该范围内找最高点深度 + 1cm（防止碰撞）
        push_z = max_height_in_roi + 0.015  # [修改] 1.5cm安全余量
        
        # 限制范围
        push_z = max(0.03, min(push_z, 0.15))
        
        # print(f"[调试] Env{env_idx} 推障碍: max_h_roi={max_height_in_roi:.3f}, push_z={push_z:.3f}")
    
    # 转换为世界坐标
    # [Fix] 坐标系修复：根据State.world_to_pixel的定义
    # u = 160 + int((y_local - 0.0) * PPM)    -> u 对应 World Y
    # v = 160 + int((x_local - 0.75) * PPM)   -> v 对应 World X
    
    PPM = 320 / 0.75  # pixels per meter
    
    # 从像素反推局部坐标
    y_local = (push_u - 160.0) / PPM
    x_local = (push_v - 160.0) / PPM + 0.75
    
    # 获取环境偏移
    env_offset_x, env_offset_y = scene.get_env_offset(env_idx)
    
    # 加上环境偏移得到世界坐标
    push_x = x_local + env_offset_x
    push_y = y_local + env_offset_y
    
    push_point = torch.tensor([push_x, push_y, push_z], dtype=torch.float32, device='cuda')
    
    return push_point, direction_idx
