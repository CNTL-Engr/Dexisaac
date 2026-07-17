"""
辅助函数：根据离散动作索引计算推点和推动方向
"""
import cv2
import numpy as np
import torch
from pathlib import Path


PPM = 320.0 / 1.0  # pixels per meter
CAMERA_Z = 1.25
TARGET_PUSH_SAFETY_MARGIN = 0.008
LEGACY_OBSTACLE_SAFETY_MARGIN = 0.015


def _ensure_uint8_mask(mask):
    """Normalize binary-like masks to uint8 [0, 255]."""
    if mask is None:
        return None
    if mask.dtype == np.uint8:
        return mask
    if mask.max() <= 1:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)


def _compute_mask_center(mask):
    """Return the integer pixel center of a binary mask."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def _get_push_angle_deg(direction_idx):
    """Map action direction index to image-frame push angle."""
    mapping = {
        0: 90,   # X+ -> image down (+V)
        1: 0,    # Y+ -> image right (+U)
        2: 270,  # X- -> image up (-V)
        3: 180,  # Y- -> image left (-U)
    }
    return mapping.get(int(direction_idx), 0)


def _infer_target_and_background_id(seg_map, target_mask):
    """Infer background as largest segment and target as dominant id inside target_mask."""
    if seg_map is None:
        return -1, 0

    unique_ids, counts = np.unique(seg_map, return_counts=True)
    if unique_ids.size == 0:
        return -1, 0

    background_id = int(unique_ids[np.argmax(counts)])
    if target_mask is None:
        return -1, background_id

    target_pixels = seg_map[target_mask > 0]
    target_pixels = target_pixels[target_pixels != background_id]
    if target_pixels.size == 0:
        return -1, background_id

    target_ids, target_counts = np.unique(target_pixels, return_counts=True)
    target_id = int(target_ids[np.argmax(target_counts)])
    return target_id, background_id


def _build_sector_mask(shape, center_u, center_v, opposite_angle_deg, angle_range_deg):
    """Create the angular search sector used by the target-push policy."""
    h, w = shape
    y_grid, x_grid = np.ogrid[:h, :w]

    dx = x_grid - center_u
    dy = y_grid - center_v
    angles = np.degrees(np.arctan2(dy, dx))
    angles = (angles + 360) % 360

    angle_min = (opposite_angle_deg - angle_range_deg + 360) % 360
    angle_max = (opposite_angle_deg + angle_range_deg) % 360

    if angle_min < angle_max:
        sector_mask = (angles >= angle_min) & (angles <= angle_max)
    else:
        sector_mask = (angles >= angle_min) | (angles <= angle_max)

    return sector_mask.astype(np.uint8) * 255


def _compute_safe_push_from_mask(depth_map, object_mask, center_u, center_v,
                                 push_angle_deg, center_depth):
    """
    Reuse the current target-push heuristic on an arbitrary object mask.
    Returns a safe XY push point and push_z.
    """
    object_mask = _ensure_uint8_mask(object_mask)
    if object_mask is None or np.sum(object_mask > 0) == 0:
        max_height = CAMERA_Z - center_depth
        push_z = max(0.03, min(max_height + TARGET_PUSH_SAFETY_MARGIN, 0.15))
        return center_u, center_v, push_z, center_depth

    opposite_angle_deg = (push_angle_deg + 180) % 360
    sector_mask = _build_sector_mask(
        object_mask.shape, center_u, center_v, opposite_angle_deg, angle_range_deg=15
    )

    dilate_radius_px = int(0.04 * PPM)
    kernel_dilate = np.ones((dilate_radius_px * 2 + 1, dilate_radius_px * 2 + 1), np.uint8)
    object_dilated = cv2.dilate(object_mask, kernel_dilate, iterations=1)

    search_mask = (object_dilated > 0) & (sector_mask > 0)
    if np.sum(search_mask) == 0:
        search_mask = sector_mask > 0

    gripper_h_px = int(0.032 * PPM)
    gripper_w_px = int(0.02 * PPM)

    if push_angle_deg % 180 == 0:
        k_rows, k_cols = gripper_h_px, gripper_w_px
    else:
        k_rows, k_cols = gripper_w_px, gripper_h_px

    kernel_roi = np.ones((k_rows, k_cols), np.uint8)
    eroded_depth = cv2.erode(depth_map, kernel_roi, iterations=1)

    max_allowed_height = 0.15
    min_allowed_depth = CAMERA_Z - max_allowed_height
    safe_mask = (eroded_depth >= min_allowed_depth) & search_mask

    masked_score = eroded_depth.copy()
    masked_score[~safe_mask] = -1.0

    best_idx = np.argmax(masked_score)
    push_v, push_u = np.unravel_index(best_idx, depth_map.shape)
    push_depth_val = eroded_depth[push_v, push_u]

    if masked_score[push_v, push_u] == -1.0:
        masked_score_fallback = eroded_depth.copy()
        masked_score_fallback[~search_mask] = -1.0
        best_idx = np.argmax(masked_score_fallback)
        push_v, push_u = np.unravel_index(best_idx, depth_map.shape)
        push_depth_val = eroded_depth[push_v, push_u]

        if masked_score_fallback[push_v, push_u] == -1.0:
            push_u, push_v = center_u, center_v
            push_depth_val = center_depth

    max_height_in_roi = CAMERA_Z - push_depth_val
    push_z = max_height_in_roi + TARGET_PUSH_SAFETY_MARGIN
    push_z = max(0.03, min(push_z, 0.15))

    return push_u, push_v, push_z, push_depth_val


def _compute_legacy_obstacle_push(depth_map, center_u, center_v, center_depth):
    """Keep the old 4-7 behavior for no-hit fallback."""
    gripper_h_px = int(0.032 * PPM)
    gripper_w_px = int(0.02 * PPM)

    h_map, w_map = depth_map.shape
    v_min = max(0, int(center_v - gripper_h_px // 2))
    v_max = min(h_map, int(center_v + gripper_h_px // 2))
    u_min = max(0, int(center_u - gripper_w_px // 2))
    u_max = min(w_map, int(center_u + gripper_w_px // 2))

    roi_depth = depth_map[v_min:v_max, u_min:u_max]
    min_depth_roi = np.min(roi_depth) if roi_depth.size > 0 else center_depth

    max_height_in_roi = CAMERA_Z - min_depth_roi
    push_z = max_height_in_roi + LEGACY_OBSTACLE_SAFETY_MARGIN
    push_z = max(0.03, min(push_z, 0.15))

    return center_u, center_v, push_z, center_depth


def _find_dominant_obstacle_mask(seg_map, target_mask, global_mask, center_u, center_v,
                                 direction_idx, target_id, background_id):
    """
    Adapt the source-project beam search:
    launch a 6 cm beam in the push direction and choose the most-hit obstacle id.
    """
    target_mask = _ensure_uint8_mask(target_mask)
    global_mask = _ensure_uint8_mask(global_mask)

    if seg_map is None or target_mask is None or global_mask is None:
        return None, -1
    if np.sum(target_mask > 0) == 0 or np.sum(global_mask > 0) == 0:
        return None, -1

    push_angle_deg = _get_push_angle_deg(direction_idx)
    angle_rad = np.radians(push_angle_deg)
    du_push = np.cos(angle_rad)
    dv_push = np.sin(angle_rad)
    perp_du = -dv_push
    perp_dv = du_push

    target_ys, target_xs = np.where(target_mask > 0)
    if target_ys.size == 0:
        return None, -1

    rel_u = target_xs.astype(float) - center_u
    rel_v = target_ys.astype(float) - center_v
    longitudinal = rel_u * du_push + rel_v * dv_push
    lateral = rel_u * perp_du + rel_v * perp_dv

    edge_line = {}
    for lon_val, lat_val in zip(longitudinal, lateral):
        lat_key = int(round(lat_val))
        lon_val = float(lon_val)
        if lat_key not in edge_line or lon_val > edge_line[lat_key]:
            edge_line[lat_key] = lon_val

    if not edge_line:
        return None, -1

    mm_to_px = PPM / 1000.0
    beam_half_width_px = 30.0 * mm_to_px
    n_rays = max(3, int(beam_half_width_px * 2) + 1)
    max_dist = int(np.hypot(*target_mask.shape))

    h, w = target_mask.shape
    hit_id_counts = {}

    for ri in range(n_rays):
        offset = -beam_half_width_px + (
            2.0 * beam_half_width_px * ri / max(n_rays - 1, 1)
        )
        lat_key = int(round(offset))

        if lat_key in edge_line:
            edge_dist = edge_line[lat_key]
        else:
            closest_key = min(edge_line.keys(), key=lambda k: abs(k - lat_key))
            if abs(closest_key - lat_key) > 2:
                continue
            edge_dist = edge_line[closest_key]

        start_u = center_u + perp_du * offset
        start_v = center_v + perp_dv * offset
        start_step = int(edge_dist) + 1

        for step in range(start_step, max_dist):
            ray_u = int(round(start_u + du_push * step))
            ray_v = int(round(start_v + dv_push * step))

            if ray_u < 0 or ray_u >= w or ray_v < 0 or ray_v >= h:
                break
            if target_mask[ray_v, ray_u] > 0:
                continue

            seg_val = int(seg_map[ray_v, ray_u])
            if (
                seg_val != background_id
                and seg_val != target_id
                and global_mask[ray_v, ray_u] > 0
            ):
                hit_id_counts[seg_val] = hit_id_counts.get(seg_val, 0) + 1
                break

    if not hit_id_counts:
        return None, -1

    dominant_seg_id = max(hit_id_counts, key=hit_id_counts.get)
    dominant_mask = (seg_map == dominant_seg_id).astype(np.uint8) * 255
    return dominant_mask, int(dominant_seg_id)


def _prim_path_for_seg_id(seg_id, seg_map, state, spawned_objects):
    """
    [点1] 把一个 seg_id 翻成对应物体的 prim_path。
    做法: 把每个物体中心投影到 seg 像素读 id, 命中 seg_id 的那个物体即是。
    返回 prim_path (str) 或 None (未找到, 如 seg_id=-1 或背景)。
    """
    if seg_id is None or seg_id == -1 or seg_map is None or not spawned_objects:
        return None
    # 首选 Isaac Lab 2.1.1 已提供的 instance-id 元信息。它直接给出 ID 对应的
    # prim 标签/路径，堆叠时不会因“物体中心像素被上层物体遮挡”而错配。
    try:
        camera_data = state.camera.camera.data
        info_all = camera_data.info
        info = info_all[0] if isinstance(info_all, list) else info_all
        seg_info = (info or {}).get("instance_id_segmentation_fast", {})
        labels = seg_info.get("idToLabels", {})
        label = labels.get(str(int(seg_id)), labels.get(int(seg_id)))

        def _strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from _strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from _strings(item)

        label_strings = list(_strings(label))
        for obj in spawned_objects:
            root = obj.cfg.prim_path.rstrip('/')
            if any(root in text for text in label_strings):
                return root
    except Exception:
        pass

    # 兼容兜底：保留原来的中心投影方法。
    h, w = seg_map.shape
    for obj in spawned_objects:
        try:
            pos_3d = obj.data.root_pos_w[0]
            if hasattr(pos_3d, "cpu"):
                pos_3d = pos_3d.cpu().numpy()
            u, v = state.world_to_pixel([pos_3d[0], pos_3d[1]])
            u_c = int(np.clip(u, 0, w - 1))
            v_c = int(np.clip(v, 0, h - 1))
            if int(seg_map[v_c, u_c]) == int(seg_id):
                return obj.cfg.prim_path
        except Exception:
            continue
    return None


def _model_id_for_prim_path(prim_path, spawned_objects):
    """从匹配物体 USD 的父文件夹读取三位模型 ID。"""
    if not prim_path:
        return None
    for obj in spawned_objects or []:
        try:
            if obj.cfg.prim_path.rstrip('/') != prim_path.rstrip('/'):
                continue
            model_id = Path(str(obj.cfg.spawn.usd_path)).parent.name
            if len(model_id) == 3 and model_id.isdigit():
                return model_id
            leaf = prim_path.rsplit('/', 1)[-1]
            candidate = leaf[len('Target_'):] if leaf.startswith('Target_') else leaf.split('_', 2)[-1]
            if len(candidate) == 3 and candidate.isdigit():
                return candidate
        except Exception:
            continue
    return None


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
    
    target_id, background_id = _infer_target_and_background_id(seg_map, target_mask)

    # 获取目标中心
    target_center = _compute_mask_center(target_mask)
    if target_center is None:
        raise ValueError(f"Env {env_idx}: 目标物体掩膜为空")

    center_u, center_v = target_center
    center_depth = depth_map[center_v, center_u]

    # [点1] 记录该动作"第一下应该碰到的物体" seg id, 供 env_wrapper 的接触判定使用。
    #   推目标 (0-3): intended = target_id
    #   推障碍 (4-7): intended = 选中的 dominant 障碍 id (无则 -1)
    if action_idx <= 3:
        # ===== 推目标物体 =====
        direction_idx = action_idx
        push_angle_deg = _get_push_angle_deg(direction_idx)
        push_u, push_v, push_z, _ = _compute_safe_push_from_mask(
            depth_map, target_mask, center_u, center_v, push_angle_deg, center_depth
        )
        intended_seg_id = int(target_id)
    else:
        # ===== 推障碍物 =====
        direction_idx = action_idx - 4
        push_angle_deg = _get_push_angle_deg(direction_idx)

        dominant_obstacle_mask, dominant_seg_id = _find_dominant_obstacle_mask(
            seg_map=seg_map,
            target_mask=target_mask,
            global_mask=global_mask,
            center_u=center_u,
            center_v=center_v,
            direction_idx=direction_idx,
            target_id=target_id,
            background_id=background_id,
        )

        if dominant_seg_id != -1:
            obstacle_center = _compute_mask_center(dominant_obstacle_mask)
            if obstacle_center is not None:
                obstacle_u, obstacle_v = obstacle_center
                obstacle_depth = depth_map[obstacle_v, obstacle_u]
                push_u, push_v, push_z, _ = _compute_safe_push_from_mask(
                    depth_map,
                    dominant_obstacle_mask,
                    obstacle_u,
                    obstacle_v,
                    push_angle_deg,
                    obstacle_depth,
                )
            else:
                push_u, push_v, push_z, _ = _compute_legacy_obstacle_push(
                    depth_map, center_u, center_v, center_depth
                )
        else:
            push_u, push_v, push_z, _ = _compute_legacy_obstacle_push(
                depth_map, center_u, center_v, center_depth
            )
        intended_seg_id = int(dominant_seg_id)  # -1 表示未选到障碍 (走 legacy fallback)

    # 转换为世界坐标
    # [Fix] 坐标系修复：根据State.world_to_pixel的定义
    # u = 160 + int((y_local - 0.0) * PPM)    -> u 对应 World Y
    # v = 160 + int((x_local - 0.75) * PPM)   -> v 对应 World X
    
    # 从像素反推局部坐标
    y_local = (push_u - 160.0) / PPM
    x_local = (push_v - 160.0) / PPM + 0.75
    
    # 获取环境偏移
    env_offset_x, env_offset_y = scene.get_env_offset(env_idx)
    
    # 加上环境偏移得到世界坐标
    push_x = x_local + env_offset_x
    push_y = y_local + env_offset_y
    
    push_point = torch.tensor([push_x, push_y, push_z], dtype=torch.float32, device='cuda')

    # [点1] 接触判定所需信息: 该动作"意图碰到的物体" prim_path + 动作类型。
    #   env_wrapper 用 PhysX 成对接触列拿到“第一下实际碰到的刚体 prim_path”，与此比对。
    #   intended_prim_path=None 表示未选到明确物体 (如推障碍时没选到 dominant, 走 legacy)。
    if action_idx <= 3:
        # 推目标不需要经过分割 ID 反查；Target_* 对象引用就是唯一意图实体。
        intended_prim_path = next(
            (
                obj.cfg.prim_path
                for obj in spawned_objects
                if obj.cfg.prim_path.rsplit('/', 1)[-1].startswith('Target_')
            ),
            None,
        )
    else:
        intended_prim_path = _prim_path_for_seg_id(
            intended_seg_id, seg_map, state, spawned_objects
        )
    contact_spec = {
        'intended_prim_path': intended_prim_path,
        'intended_model_id': _model_id_for_prim_path(intended_prim_path, spawned_objects),
        'kind': 'target' if action_idx <= 3 else 'obstacle',
    }

    return push_point, direction_idx, contact_spec
