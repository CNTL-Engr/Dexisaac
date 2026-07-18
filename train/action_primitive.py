"""
辅助函数：根据离散动作索引计算推点和推动方向
"""
import cv2
import numpy as np
import torch
from pathlib import Path


PPM = 320.0 / 1.0  # pixels per meter
CAMERA_Z = 1.25

# 夹爪在图像平面内的近似足迹尺寸。沿推动方向的尺寸较短，横向尺寸较长。
# 这些尺寸同时用于“候选窗口是否覆盖意图物体/躲避物体”的硬约束。
GRIPPER_LATERAL_M = 0.03
GRIPPER_PUSH_AXIS_M = 0.02

# sim-ur10 在 452 px/m 图像中以 15 px 作为线段断开阈值。
# 使用米制常量保存同一物理含义，后续可直接在此处调整。
EDGE_SEGMENT_GAP_M = 9.0 / 452.0

# 迁移后的边缘防撞邻域半径。当前推点图像覆盖 1 m / 320 px，30 mm
# 对应 9.6 px；运行时向上取整为 10 px，保证实际检查范围不小于 30 mm。
EDGE_NEIGHBOR_RADIUS_M = 0.030
EDGE_INWARD_SAMPLE_M = 0.009
EDGE_HEIGHT_OFFSET_M = 0.015
EDGE_HEIGHT_FLOOR_M = 0.035

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


def _cardinal_push_step(push_angle_deg):
    """将离散推动角映射成图像坐标中的单位步长 (du, dv)。"""
    angle = int(round(float(push_angle_deg))) % 360
    mapping = {
        0: (1, 0),
        90: (0, 1),
        180: (-1, 0),
        270: (0, -1),
    }
    if angle not in mapping:
        raise ValueError(f"推点算法只支持 C4 方向，收到角度 {push_angle_deg}")
    return mapping[angle]

def _extract_rear_edge_for_direction(object_mask, push_angle_deg):
    """提取意图物体迎向夹爪一侧的离散边缘。

    这是 sim-ur10 ``PushPointState`` 中逐行/逐列扫描逻辑的 320px 版本。
    连接阈值用米制常数换算，避免直接复用 452px 工作空间中的像素阈值。
    """
    mask = _ensure_uint8_mask(object_mask)
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8), []

    h, w = mask.shape
    points = []
    angle = int(round(float(push_angle_deg))) % 360
    if angle == 90:  # X+：每列最上方像素
        for u in range(w):
            rows = np.where(mask[:, u] > 0)[0]
            if rows.size:
                points.append((u, int(rows[0])))
    elif angle == 0:  # Y+：每行最左方像素
        for v in range(h):
            cols = np.where(mask[v, :] > 0)[0]
            if cols.size:
                points.append((int(cols[0]), v))
    elif angle == 270:  # X-：每列最下方像素
        for u in range(w):
            rows = np.where(mask[:, u] > 0)[0]
            if rows.size:
                points.append((u, int(rows[-1])))
    elif angle == 180:  # Y-：每行最右方像素
        for v in range(h):
            cols = np.where(mask[v, :] > 0)[0]
            if cols.size:
                points.append((int(cols[-1]), v))
    else:
        raise ValueError(f"推点算法只支持 C4 方向，收到角度 {push_angle_deg}")

    edge = np.zeros_like(mask, dtype=np.uint8)
    if not points:
        return edge, []

    max_gap_px = max(1, int(np.ceil(0.022 * PPM)))
    bridge_gap_px = max(1, int(np.ceil(0.009 * PPM)))
    for idx, (u, v) in enumerate(points):
        edge[v, u] = 255
        if idx == 0:
            continue
        pu, pv = points[idx - 1]
        du = abs(u - pu)
        dv = abs(v - pv)
        dist = float(np.hypot(du, dv))
        slope = float(np.degrees(np.arctan2(dv, du))) if du > 0 else 90.0
        if (dist <= max_gap_px and slope < 75.0) or dist <= bridge_gap_px:
            cv2.line(edge, (pu, pv), (u, v), 255, 1)

    ys, xs = np.where(edge > 0)
    return edge, list(zip(xs.tolist(), ys.tolist()))


def _filter_edge_by_height_30mm(edge_mask, edge_points, global_mask,
                                object_mask, depth_map):
    """按 30 mm 邻域和相对高度过滤边缘点。

    与源项目一致：其他物体深度小于等于意图物体局部边缘深度时，
    认为其物理高度等于或高于边缘，删除对应边缘点。所有剩余片段均保留，
    由后续横向投影聚类和线段中心选择统一处理。
    """
    edge_mask = _ensure_uint8_mask(edge_mask)
    object_mask = _ensure_uint8_mask(object_mask)
    global_mask = _ensure_uint8_mask(global_mask)
    depth = np.asarray(depth_map, dtype=np.float32)
    if edge_mask is None or object_mask is None:
        return edge_mask, [], np.zeros_like(depth, dtype=bool), np.zeros_like(depth, dtype=bool)

    radius_px = max(1, int(np.ceil(EDGE_NEIGHBOR_RADIUS_M * PPM)))
    neighborhood_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    neighborhood_mask = cv2.dilate(edge_mask, neighborhood_kernel, iterations=1) > 0

    other_mask = np.zeros_like(object_mask, dtype=bool)
    if global_mask is not None:
        other_mask = (global_mask > 0) & ~(object_mask > 0)

    other_depth = np.full(depth.shape, 10.0, dtype=np.float32)
    other_depth[other_mask] = depth[other_mask]
    highest_other_depth = cv2.erode(other_depth, neighborhood_kernel, iterations=1)

    local_radius_px = max(1, int(np.ceil(EDGE_INWARD_SAMPLE_M * PPM)))
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * local_radius_px + 1, 2 * local_radius_px + 1)
    )
    object_depth = np.full(depth.shape, 10.0, dtype=np.float32)
    object_depth[object_mask > 0] = depth[object_mask > 0]
    highest_object_depth = cv2.erode(object_depth, local_kernel, iterations=1)

    filtered = edge_mask.copy()
    deleted = np.zeros_like(filtered, dtype=np.uint8)
    kept = []
    for u, v in edge_points:
        if highest_other_depth[v, u] <= highest_object_depth[v, u]:
            filtered[v, u] = 0
            deleted[v, u] = 255
        else:
            kept.append((u, v))

    return filtered, kept, neighborhood_mask, deleted > 0


def _select_sim_ur10_edge_anchor(filtered_edge_mask, object_mask, global_mask,
                                  center_u, center_v, push_angle_deg):
    """迁移 sim-ur10 的线段中心锚点，并做 R2F140 足迹可行性检查。

    边缘先按横向投影间距聚类；线段按中心到意图物体质心的距离排序。
    每条线段只检查其中心锚点，不再沿线滑动。若当前线段过窄、越界或
    回撤后的 32×15 mm 足迹覆盖其他物体，则继续尝试下一线段。
    """
    if filtered_edge_mask is None:
        return None, {
            'segment_count': 0,
            'wide_segment_count': 0,
            'feasible_segment_count': 0,
            'selected_segment_index': None,
            'selected_segment_range': None,
            'selected_segment_mask': None,
            'geometry_invalid_reason': 'no_filtered_edge',
        }

    empty_debug = {
        'segment_count': 0,
        'wide_segment_count': 0,
        'feasible_segment_count': 0,
        'selected_segment_index': None,
        'selected_segment_range': None,
        'selected_segment_mask': np.zeros_like(filtered_edge_mask, dtype=bool),
    }
    if not np.any(filtered_edge_mask > 0):
        empty_debug['geometry_invalid_reason'] = 'no_filtered_edge'
        return None, empty_debug

    edge_v, edge_u = np.where(filtered_edge_mask > 0)
    edge_points = np.column_stack((edge_u, edge_v)).astype(np.float32)
    angle_rad = np.radians(float(push_angle_deg))
    du, dv = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    tu, tv = -dv, du

    projections = edge_points[:, 0] * tu + edge_points[:, 1] * tv
    center_projection = float(center_u) * tu + float(center_v) * tv
    sorted_projections = np.sort(projections)
    gap_threshold_px = float(EDGE_SEGMENT_GAP_M * PPM)

    clusters = []
    current_cluster = [float(sorted_projections[0])]
    for idx in range(1, len(sorted_projections)):
        value = float(sorted_projections[idx])
        if value - float(sorted_projections[idx - 1]) > gap_threshold_px:
            clusters.append(current_cluster)
            current_cluster = [value]
        else:
            current_cluster.append(value)
    clusters.append(current_cluster)

    # sim-ur10 首先选择中心最靠近质心的线段。为适配当前夹爪，这里按
    # 同一距离排序逐条验证，最近线段不可执行时才尝试下一条。
    ranked_clusters = []
    for original_index, cluster in enumerate(clusters):
        segment_center = 0.5 * (cluster[0] + cluster[-1])
        ranked_clusters.append((
            abs(segment_center - center_projection),
            original_index,
            cluster,
            segment_center,
        ))
    ranked_clusters.sort(key=lambda item: (item[0], item[1]))

    object_mask = _ensure_uint8_mask(object_mask)
    global_mask = _ensure_uint8_mask(global_mask)
    other_mask = np.zeros_like(filtered_edge_mask, dtype=bool)
    if global_mask is not None:
        other_mask = (global_mask > 0) & ~(
            object_mask > 0 if object_mask is not None else False
        )

    h, w = filtered_edge_mask.shape
    min_segment_width_px = float(GRIPPER_LATERAL_M * PPM)
    half_lateral_px = 0.5 * min_segment_width_px
    half_push_px = 0.5 * float(GRIPPER_PUSH_AXIS_M * PPM)
    retreat_px = float(0.005 * PPM)
    half_u_px = abs(tu) * half_lateral_px + abs(du) * half_push_px
    half_v_px = abs(tv) * half_lateral_px + abs(dv) * half_push_px
    wide_segment_count = sum(
        float(cluster[-1] - cluster[0]) >= min_segment_width_px
        for cluster in clusters
    )
    feasible_segment_count = 0

    for _, original_index, cluster, segment_center in ranked_clusters:
        segment_width_px = float(cluster[-1] - cluster[0])
        if segment_width_px < min_segment_width_px:
            continue

        # 将线段中心吸附到最近的真实横向投影，再取对应的实际边缘像素。
        target_projection = projections[np.argmin(np.abs(projections - segment_center))]
        contact_index = int(np.argmin(np.abs(projections - target_projection)))
        contact_u, contact_v = edge_points[contact_index]
        push_u = float(contact_u - du * retreat_px)
        push_v = float(contact_v - dv * retreat_px)

        if (
            push_u - half_u_px < 0.0
            or push_u + half_u_px > w - 1
            or push_v - half_v_px < 0.0
            or push_v + half_v_px > h - 1
        ):
            continue

        # 当前仅支持 C4，足迹在图像中始终轴对齐。使用像素中心判断
        # 32×15 mm 矩形内是否存在任何非意图物体。
        u_min = max(0, int(np.ceil(push_u - half_u_px)))
        u_max = min(w - 1, int(np.floor(push_u + half_u_px)))
        v_min = max(0, int(np.ceil(push_v - half_v_px)))
        v_max = min(h - 1, int(np.floor(push_v + half_v_px)))
        if np.any(other_mask[v_min:v_max + 1, u_min:u_max + 1]):
            continue

        feasible_segment_count += 1
        selected_segment_mask = (
            (projections >= float(cluster[0]))
            & (projections <= float(cluster[-1]))
        )
        selected_mask_image = np.zeros_like(filtered_edge_mask, dtype=bool)
        selected_points = edge_points[selected_segment_mask].astype(np.int32)
        selected_mask_image[selected_points[:, 1], selected_points[:, 0]] = True
        return {
            'contact_pixel': (int(round(contact_u)), int(round(contact_v))),
            'push_pixel_float': (push_u, push_v),
        }, {
            'segment_count': int(len(clusters)),
            'wide_segment_count': int(wide_segment_count),
            'feasible_segment_count': int(feasible_segment_count),
            'selected_segment_index': int(original_index),
            'selected_segment_range': (
                float(cluster[0]), float(cluster[-1])
            ),
            'selected_segment_mask': selected_mask_image,
            'segment_gap_threshold_px': gap_threshold_px,
        }

    reason = (
        'no_segment_wide_enough'
        if wide_segment_count == 0
        else 'no_collision_free_gripper_window'
    )
    empty_debug.update({
        'segment_count': int(len(clusters)),
        'wide_segment_count': int(wide_segment_count),
        'feasible_segment_count': int(feasible_segment_count),
        'segment_gap_threshold_px': gap_threshold_px,
        'geometry_invalid_reason': reason,
    })
    return None, empty_debug


def _compute_migrated_edge_push(depth_map, object_mask, center_u, center_v,
                                push_angle_deg, global_mask=None,
                                debug_info=None):
    """执行迁移后的边缘过滤、线段锚点和最高边缘高度推点算法。"""
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=CAMERA_Z, posinf=CAMERA_Z, neginf=CAMERA_Z)
    object_mask = _ensure_uint8_mask(object_mask)
    if debug_info is not None:
        debug_info.clear()

    edge_mask, edge_points = _extract_rear_edge_for_direction(object_mask, push_angle_deg)
    filtered_mask, filtered_points, neighborhood_mask, deleted_mask = _filter_edge_by_height_30mm(
        edge_mask, edge_points, global_mask, object_mask, depth
    )
    result, selection_debug = _select_sim_ur10_edge_anchor(
        filtered_mask,
        object_mask,
        global_mask,
        center_u,
        center_v,
        push_angle_deg,
    )

    if debug_info is not None:
        debug_info.update({
            'raw_edge_mask': edge_mask > 0,
            'candidate_mask': filtered_mask > 0,
            'edge_neighborhood_mask': neighborhood_mask,
            'height_deleted_mask': deleted_mask,
            'selected_from_candidates': result is not None,
        })
        debug_info.update(selection_debug)

    if result is None:
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = selection_debug.get(
                'geometry_invalid_reason', 'no_valid_edge_segment'
            )
        return None

    # 高度采样使用全部高度过滤后边缘，并沿推动方向向物体内部偏移约 9 mm。
    inward_px = max(1, int(np.ceil(EDGE_INWARD_SAMPLE_M * PPM)))
    du, dv = _cardinal_push_step(push_angle_deg)
    edge_depths = []
    for u, v in filtered_points:
        sample_u = int(np.clip(u + du * inward_px, 0, depth.shape[1] - 1))
        sample_v = int(np.clip(v + dv * inward_px, 0, depth.shape[0] - 1))
        value = float(depth[sample_v, sample_u])
        if np.isfinite(value) and 0.0 < value < 1.9:
            edge_depths.append(value)

    if not edge_depths:
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = 'no_valid_edge_depth'
        return None

    # 深度越小代表物理高度越高。与 sim-ur10 一致，使用全部绿色过滤
    # 边缘中的最高高度，而不是均值或仅选中线段的高度。
    max_edge_height = CAMERA_Z - float(np.min(edge_depths))
    push_z = max(max_edge_height - EDGE_HEIGHT_OFFSET_M, EDGE_HEIGHT_FLOOR_M)
    push_u, push_v = result['push_pixel_float']
    contact_u, contact_v = result['contact_pixel']
    selected_depth = float(depth[contact_v, contact_u])

    if debug_info is not None:
        debug_info.update({
            'selected_contact_pixel': result['contact_pixel'],
            'push_pixel_float': (push_u, push_v),
            'edge_max_height': float(max_edge_height),
            'edge_height_sample_count': int(len(edge_depths)),
            'push_z': float(push_z),
        })
    return push_u, push_v, push_z, selected_depth



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


def compute_push_point_from_action(action_idx, env_idx, state, scene, spawned_objects,
                                   debug_info=None):
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
    # 直接复用生成网络输入状态时缓存的同一帧。缓存中的 depth_map 是米制
    # float 深度，而网络使用的是由它归一化得到的 uint8 深度通道。
    # 此处禁止重新调用相机，否则推点对应的场景可能与网络评价动作时不同。
    images = state.get_cached_observation_images()
    if images is None:
        raise ValueError(
            f"Env {env_idx}: 缺少网络观测帧缓存；"
            "请先通过 State.get_state() 获取网络输入，再计算推点"
        )
    
    _rgb, depth_map, seg_map = images  # depth_map 是米制深度(float)

    # 掩膜也直接复用生成网络第 2/3 通道时的数组，不再二次
    # 投影物体中心或重新判断背景，避免“同帧但掩膜不同”。
    masks = state.get_cached_observation_masks()
    if masks is None:
        raise ValueError(f"Env {env_idx}: 缺少网络观测掩膜缓存")
    target_mask, global_mask = masks
    target_mask = _ensure_uint8_mask(target_mask)
    global_mask = _ensure_uint8_mask(global_mask)
    if target_mask is None or not np.any(target_mask > 0):
        raise ValueError(f"Env {env_idx}: 网络观测中的目标物体掩膜为空")
    if global_mask is None:
        global_mask = np.zeros_like(target_mask, dtype=np.uint8)
    
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
    geometry_debug = {} if debug_info is not None else None
    geometry_invalid_reason = None
    if action_idx <= 3:
        # ===== 推目标物体 =====
        direction_idx = action_idx
        push_angle_deg = _get_push_angle_deg(direction_idx)
        migrated_result = _compute_migrated_edge_push(
            depth_map, target_mask, center_u, center_v, push_angle_deg,
            global_mask=global_mask, debug_info=geometry_debug
        )
        if migrated_result is None:
            push_u = push_v = push_z = None
            geometry_invalid_reason = geometry_debug.get(
                'geometry_invalid_reason', 'no_valid_edge_segment'
            ) if geometry_debug is not None else 'no_valid_edge_segment'
        else:
            push_u, push_v, push_z, _ = migrated_result
        intended_seg_id = int(target_id)
        intended_mask = target_mask
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
                migrated_result = _compute_migrated_edge_push(
                    depth_map,
                    dominant_obstacle_mask,
                    obstacle_u,
                    obstacle_v,
                    push_angle_deg,
                    global_mask=global_mask,
                    debug_info=geometry_debug,
                )
                if migrated_result is None:
                    push_u = push_v = push_z = None
                    geometry_invalid_reason = geometry_debug.get(
                        'geometry_invalid_reason', 'no_valid_edge_segment'
                    ) if geometry_debug is not None else 'no_valid_edge_segment'
                else:
                    push_u, push_v, push_z, _ = migrated_result
            else:
                push_u = push_v = push_z = None
                geometry_invalid_reason = 'no_dominant_obstacle'
                if geometry_debug is not None:
                    geometry_debug.clear()
                    geometry_debug.update({
                        'candidate_mask': np.zeros_like(target_mask, dtype=bool),
                        'edge_neighborhood_mask': np.zeros_like(target_mask, dtype=bool),
                        'selected_from_candidates': False,
                        'geometry_invalid_reason': geometry_invalid_reason,
                    })
        else:
            push_u = push_v = push_z = None
            geometry_invalid_reason = 'no_dominant_obstacle'
            if geometry_debug is not None:
                geometry_debug.clear()
                geometry_debug.update({
                    'candidate_mask': np.zeros_like(target_mask, dtype=bool),
                    'edge_neighborhood_mask': np.zeros_like(target_mask, dtype=bool),
                    'selected_from_candidates': False,
                    'geometry_invalid_reason': geometry_invalid_reason,
                })
        intended_seg_id = int(dominant_seg_id)  # -1 表示未选到障碍，动作几何无效
        intended_mask = (
            dominant_obstacle_mask
            if dominant_obstacle_mask is not None
            else np.zeros_like(target_mask, dtype=np.uint8)
        )

    # 没有安全几何推点时不产生任何默认运动点，由环境按空推处理。
    geometry_valid = push_u is not None and push_v is not None and push_z is not None

    # 转换为世界坐标
    # [Fix] 坐标系修复：根据State.world_to_pixel的定义
    # u = 160 + int((y_local - 0.0) * PPM)    -> u 对应 World Y
    # v = 160 + int((x_local - 0.75) * PPM)   -> v 对应 World X
    
    # 从像素反推局部坐标
    if geometry_valid:
        y_local = (float(push_u) - 160.0) / PPM
        x_local = (float(push_v) - 160.0) / PPM + 0.75
    else:
        y_local = x_local = 0.0
    
    # 获取环境偏移
    env_offset_x, env_offset_y = scene.get_env_offset(env_idx)
    
    # 加上环境偏移得到世界坐标
    push_x = x_local + env_offset_x
    push_y = y_local + env_offset_y

    push_point = None
    if geometry_valid:
        push_point = torch.tensor([push_x, push_y, push_z], dtype=torch.float32, device='cuda')

    # [点1] 接触判定所需信息: 该动作"意图碰到的物体" prim_path + 动作类型。
    #   env_wrapper 用 PhysX 成对接触列拿到“第一下实际碰到的刚体 prim_path”，与此比对。
    #   intended_prim_path=None 表示未选到明确物体（例如没有主障碍物）。
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
        'geometry_valid': bool(geometry_valid),
        'geometry_invalid_reason': geometry_invalid_reason,
    }

    if debug_info is not None:
        debug_info.clear()
        debug_info.update(geometry_debug or {})
        debug_info.update({
            'rgb': _rgb,
            'depth_map': depth_map,
            'seg_map': seg_map,
            'target_mask': target_mask,
            'global_mask': global_mask,
            'intended_mask': intended_mask,
            'push_pixel': (
                int(round(push_u)), int(round(push_v))
            ) if geometry_valid else None,
            'push_pixel_float': (float(push_u), float(push_v)) if geometry_valid else None,
            'push_z': float(push_z) if geometry_valid else None,
            'direction_idx': int(direction_idx),
            'intended_seg_id': int(intended_seg_id),
            'geometry_valid': bool(geometry_valid),
            'geometry_invalid_reason': geometry_invalid_reason,
        })

    return push_point, direction_idx, contact_spec
