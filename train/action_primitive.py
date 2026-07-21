"""
辅助函数：根据离散动作索引计算推点和推动方向
"""
import cv2
import numpy as np
import torch
from pathlib import Path


PPM = 320.0 / 1.0  # pixels per meter 1像素约为3.125 mm
CAMERA_Z = 1.25 # 顶视相机在世界坐标中的高度
TABLE_Z = 0.02 # 桌面在世界坐标中的高度

# 夹爪最低端物理足迹
GRIPPER_LATERAL_M = 0.03 #夹爪尖端检测窗口长度
GRIPPER_PUSH_AXIS_M = 0.02 #夹爪尖端检测窗口宽度

# G02 / 高度可行区间参数。
APPROACH_BAND_M = 0.050 #G00向动作反方向延伸的长度
TARGET_HEIGHT_STRIP_MIN_M = 0.005 # 从迎推边缘向意图物体内部采样高度，接触条的最近距离
TARGET_HEIGHT_STRIP_MAX_M = 0.010 # 从迎推边缘向意图物体内部采样高度，接触条的最远距离
VERTICAL_CLEARANCE_M = 0.010 # G02像素高度筛选时，环境像素+1cm
MIN_PUSH_OVERLAP_M = 0.010 # G02像素通过筛选时，要求夹爪与意图物体至少在z轴方向上保持1cm重叠
TARGET_HEIGHT_PERCENTILE = 95.0 # 接触条带内目标高度取95%
FINAL_PUSH_RETREAT_M = 0.003 # 确定窗口的xy坐标后，向-d平移的距离
CHANNEL_Z_CLEARANCE_M = 0.01 # 在检测窗口前进通道最高高度上增加的安全余量

# 动作 4--7 的两档意图障碍物搜索半径。
OBSTACLE_RAY_TIER1_M = 0.020
OBSTACLE_RAY_TIER2_M = 0.050

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

    # 不跨列/跨行补线。分割中的凹口、遮挡和断裂均保持原样，避免把不存在
    # 的迎推表面补成可接触边缘。
    for u, v in points:
        edge[v, u] = 255

    ys, xs = np.where(edge > 0)
    return edge, list(zip(xs.tolist(), ys.tolist()))


def _pixel_radius(meters):
    return max(1, int(np.floor(float(meters) * PPM + 1e-6)))


def _cardinal_components(push_angle_deg):
    du, dv = _cardinal_push_step(push_angle_deg)
    tu, tv = -dv, du
    return float(du), float(dv), float(tu), float(tv)


def _rect_bounds(center_u, center_v, push_angle_deg, depth_shape):
    """Return the raster footprint for the 3.5 x 2.5 cm low gripper."""
    h, w = depth_shape
    du, dv, tu, tv = _cardinal_components(push_angle_deg)
    half_lateral = _pixel_radius(GRIPPER_LATERAL_M / 2.0)
    half_push = _pixel_radius(GRIPPER_PUSH_AXIS_M / 2.0)
    half_u = abs(tu) * half_lateral + abs(du) * half_push
    half_v = abs(tv) * half_lateral + abs(dv) * half_push
    u0 = int(np.ceil(float(center_u) - half_u))
    u1 = int(np.floor(float(center_u) + half_u))
    v0 = int(np.ceil(float(center_v) - half_v))
    v1 = int(np.floor(float(center_v) + half_v))
    inside = u0 >= 0 and v0 >= 0 and u1 < w and v1 < h
    return (u0, u1, v0, v1), inside


def _height_map(depth_map, camera_height=None):
    """Return median-filtered height and the validity of the original depth."""
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    camera_z = CAMERA_Z if camera_height is None else float(camera_height)
    if np.any(valid_depth):
        replacement = float(np.median(depth[valid_depth]))
    else:
        replacement = camera_z
    filtered_depth = np.where(valid_depth, depth, replacement).astype(np.float32)
    height = camera_z - filtered_depth
    if np.any(valid_depth):
        # 3x3 median suppresses isolated RGB-D speckles without changing the
        # centimeter-scale geometry of the candidate windows.
        height = cv2.medianBlur(height.astype(np.float32), 3)
    return height, valid_depth


def _percentile_or_none(values, percentile):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _candidate_contact_strip(height, object_mask, q_u, q_v, push_angle_deg):
    """Sample a 5--10 mm inward, lateral contact strip."""
    du, dv, tu, tv = _cardinal_components(push_angle_deg)
    inward0 = max(1, int(np.ceil(TARGET_HEIGHT_STRIP_MIN_M * PPM)))
    inward1 = max(inward0, int(np.floor(TARGET_HEIGHT_STRIP_MAX_M * PPM)))
    lateral = _pixel_radius(GRIPPER_LATERAL_M / 2.0)
    h, w = object_mask.shape
    values = []
    for s in range(inward0, inward1 + 1):
        for lt in range(-lateral, lateral + 1):
            u = int(round(q_u + du * s + tu * lt))
            v = int(round(q_v + dv * s + tv * lt))
            if 0 <= u < w and 0 <= v < h and object_mask[v, u] > 0:
                values.append(float(height[v, u]))
    return _percentile_or_none(values, TARGET_HEIGHT_PERCENTILE)


def _build_directional_band(object_mask, edge_points, push_angle_deg):
    """Build G01/G02 by sweeping only the first-hit迎推 edge toward the gripper."""
    mask = object_mask > 0
    g01 = mask.copy()
    h, w = mask.shape
    du, dv = _cardinal_push_step(push_angle_deg)
    max_s = max(1, int(np.ceil(APPROACH_BAND_M * PPM)))
    for u, v in edge_points:
        for s in range(1, max_s + 1):
            su = int(u - du * s)
            sv = int(v - dv * s)
            if 0 <= su < w and 0 <= sv < h:
                g01[sv, su] = True
    return g01, g01 & ~mask


def _build_contact_height_maps(height, object_mask, edge_points,
                               push_angle_deg, g02):
    """Associate every G02 pixel with its source edge point and P90 height."""
    h, w = g02.shape
    du, dv = _cardinal_push_step(push_angle_deg)
    max_s = max(1, int(np.ceil(APPROACH_BAND_M * PPM)))
    target_height = np.full((h, w), np.nan, dtype=np.float32)
    contact_u = np.full((h, w), -1, dtype=np.int32)
    contact_v = np.full((h, w), -1, dtype=np.int32)
    for q_u, q_v in edge_points:
        q_u = int(q_u)
        q_v = int(q_v)
        h_target = _candidate_contact_strip(
            height, object_mask, q_u, q_v, push_angle_deg
        )
        for s in range(1, max_s + 1):
            u = int(q_u - du * s)
            v = int(q_v - dv * s)
            if not (0 <= u < w and 0 <= v < h and g02[v, u]):
                continue
            contact_u[v, u] = q_u
            contact_v[v, u] = q_v
            if h_target is not None:
                target_height[v, u] = float(h_target)
    return target_height, contact_u, contact_v


def _window_kernel(push_angle_deg):
    """Return the exact axis-aligned C4 footprint used by ``_rect_bounds``."""
    du, dv, tu, tv = _cardinal_components(push_angle_deg)
    half_lateral = _pixel_radius(GRIPPER_LATERAL_M / 2.0)
    half_push = _pixel_radius(GRIPPER_PUSH_AXIS_M / 2.0)
    half_u = int(abs(tu) * half_lateral + abs(du) * half_push)
    half_v = int(abs(tv) * half_lateral + abs(dv) * half_push)
    return np.ones((2 * half_v + 1, 2 * half_u + 1), dtype=np.uint8)


def _build_push_channel_mask(start_u, start_v, contact_u, contact_v,
                             push_angle_deg, image_shape):
    """Sweep the gripper window from the final XY until first edge contact."""
    du, dv, _, _ = _cardinal_components(push_angle_deg)
    half_push = _pixel_radius(GRIPPER_PUSH_AXIS_M / 2.0)
    end_u = float(contact_u) - du * half_push
    end_v = float(contact_v) - dv * half_push
    distance_px = max(
        0.0,
        (end_u - float(start_u)) * du + (end_v - float(start_v)) * dv,
    )
    channel_mask = np.zeros(image_shape, dtype=bool)
    sample_count = max(1, int(np.ceil(distance_px)))
    for step in range(sample_count + 1):
        travel = min(float(step), distance_px)
        center_u = float(start_u) + du * travel
        center_v = float(start_v) + dv * travel
        bounds, inside = _rect_bounds(
            center_u, center_v, push_angle_deg, image_shape
        )
        if not inside:
            return channel_mask, False, (end_u, end_v)
        u0, u1, v0, v1 = bounds
        channel_mask[v0:v1 + 1, u0:u1 + 1] = True
    return channel_mask, True, (end_u, end_v)


def _compute_migrated_edge_push(depth_map, object_mask, center_u, center_v,
                                push_angle_deg, global_mask=None,
                                debug_info=None, table_height=None,
                                camera_height=None):
    """Filter G02 per pixel, slide the gripper window, and select a push."""
    depth = np.asarray(depth_map, dtype=np.float32)
    object_mask = _ensure_uint8_mask(object_mask)
    if object_mask is None or object_mask.size == 0:
        if debug_info is not None:
            debug_info.clear()
            debug_info['geometry_invalid_reason'] = 'empty_intended_mask'
        return None

    edge_mask, edge_points = _extract_rear_edge_for_direction(object_mask, push_angle_deg)
    g01, g02 = _build_directional_band(object_mask, edge_points, push_angle_deg)
    height, valid_depth = _height_map(depth, camera_height=camera_height)
    table_z = TABLE_Z if table_height is None else float(table_height)

    if debug_info is not None:
        debug_info.clear()
        debug_info.update({
            'raw_edge_mask': edge_mask > 0,
            'candidate_mask': np.zeros_like(g02, dtype=bool),
            'g00_mask': object_mask > 0,
            'g01_mask': g01,
            'g02_mask': g02,
            'edge_neighborhood_mask': g02,
            'raw_obs_mask': np.zeros_like(g02, dtype=bool),
            'obs_mask': np.zeros_like(g02, dtype=bool),
            'height_feasible_mask': np.zeros_like(g02, dtype=bool),
            'window_center_mask': np.zeros_like(g02, dtype=bool),
            'selected_window_mask': np.zeros_like(g02, dtype=bool),
            'channel_mask': np.zeros_like(g02, dtype=bool),
            'selected_from_candidates': False,
            'edge_points': edge_points,
        })

    if not edge_points:
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = 'no_filtered_edge'
        return None

    du, dv, tu, tv = _cardinal_components(push_angle_deg)
    centroid_u, centroid_v = _compute_mask_center(object_mask)
    center_t = float(centroid_u) * tu + float(centroid_v) * tv

    target_height_map, contact_u_map, contact_v_map = _build_contact_height_maps(
        height, object_mask, edge_points, push_angle_deg, g02
    )
    environment_height = np.maximum(height, table_z)
    pixel_z_min = environment_height + VERTICAL_CLEARANCE_M
    pixel_z_max = target_height_map - MIN_PUSH_OVERLAP_M
    raw_obs_mask = g02 & (
        ~valid_depth
        | ~np.isfinite(target_height_map)
        | (pixel_z_min > pixel_z_max)
    )
    obs_mask = cv2.dilate(
        raw_obs_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    obs_mask &= g02
    height_feasible = g02 & ~obs_mask

    if debug_info is not None:
        debug_info.update({
            'raw_obs_mask': raw_obs_mask,
            'obs_mask': obs_mask,
            'height_feasible_mask': height_feasible,
            'target_height_map_p90': target_height_map,
        })

    if not np.any(height_feasible):
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = 'no_height_feasible_pixels'
        return None

    kernel = _window_kernel(push_angle_deg)
    window_fit_mask = cv2.erode(
        height_feasible.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    window_fit_mask &= height_feasible
    if debug_info is not None:
        debug_info['window_center_mask'] = window_fit_mask.copy()

    if not np.any(window_fit_mask):
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = 'no_window_fit_after_height_filter'
        return None

    candidates = []
    center_vs, center_us = np.where(window_fit_mask)
    for p_v, p_u in zip(center_vs.tolist(), center_us.tolist()):
        q_u = int(contact_u_map[p_v, p_u])
        q_v = int(contact_v_map[p_v, p_u])
        h_target = float(target_height_map[p_v, p_u])
        if q_u < 0 or q_v < 0 or not np.isfinite(h_target):
            continue
        bounds, inside = _rect_bounds(p_u, p_v, push_angle_deg, depth.shape)
        if not inside:
            continue
        line_error = abs((float(q_u) * tu + float(q_v) * tv) - center_t)
        standoff = (float(q_u - p_u) * du + float(q_v - p_v) * dv)
        candidates.append({
            'push_u': float(p_u),
            'push_v': float(p_v),
            'contact_u': q_u,
            'contact_v': q_v,
            'h_target_p90': float(h_target),
            'line_error_px': float(line_error),
            'standoff_px': float(standoff),
            'window_bounds': bounds,
        })

    if not candidates:
        if debug_info is not None:
            debug_info['geometry_invalid_reason'] = 'no_window_fit_after_height_filter'
        return None

    candidate_mask = np.zeros_like(g02, dtype=bool)
    for item in candidates:
        candidate_mask[int(item['push_v']), int(item['push_u'])] = True
    clearance = cv2.distanceTransform(candidate_mask.astype(np.uint8), cv2.DIST_L2, 3)
    for item in candidates:
        u = int(item['push_u'])
        v = int(item['push_v'])
        item['clearance_px'] = float(clearance[v, u])
    candidates.sort(key=lambda item: (
        item['line_error_px'],
        -item['clearance_px'],
        abs(item['standoff_px'] - 0.02 * PPM),
        int(round(item['push_v'])),
        int(round(item['push_u'])),
    ))
    selected = candidates[0]
    u0, u1, v0, v1 = selected['window_bounds']
    selected_center_u = selected['push_u']
    selected_center_v = selected['push_v']
    retreat_px = FINAL_PUSH_RETREAT_M * PPM
    final_push_u = selected_center_u - du * retreat_px
    final_push_v = selected_center_v - dv * retreat_px
    channel_mask, channel_inside, channel_end = _build_push_channel_mask(
        final_push_u,
        final_push_v,
        selected['contact_u'],
        selected['contact_v'],
        push_angle_deg,
        depth.shape,
    )
    channel_height_mask = channel_mask & ~(object_mask > 0)
    if not channel_inside or not np.any(channel_height_mask):
        if debug_info is not None:
            debug_info.update({
                'candidate_mask': candidate_mask,
                'window_center_mask': candidate_mask,
                'channel_mask': channel_height_mask,
                'geometry_invalid_reason': 'invalid_push_channel',
            })
        return None
    if not np.all(valid_depth[channel_height_mask]):
        if debug_info is not None:
            debug_info.update({
                'candidate_mask': candidate_mask,
                'window_center_mask': candidate_mask,
                'channel_mask': channel_height_mask,
                'geometry_invalid_reason': 'invalid_channel_depth',
            })
        return None
    channel_max_height = float(np.max(environment_height[channel_height_mask]))
    z_value = channel_max_height + CHANNEL_Z_CLEARANCE_M
    if z_value > 0.4:
        if debug_info is not None:
            debug_info.update({
                'candidate_mask': candidate_mask,
                'window_center_mask': candidate_mask,
                'channel_mask': channel_height_mask,
                'channel_max_height': channel_max_height,
                'geometry_invalid_reason': 'no_feasible_z',
            })
        return None

    selected_mask = np.zeros_like(g02, dtype=bool)
    selected_mask[v0:v1 + 1, u0:u1 + 1] = True
    if debug_info is not None:
        debug_info.update({
            'candidate_mask': candidate_mask,
            'window_center_mask': candidate_mask,
            'selected_window_mask': selected_mask,
            'selected_window_bounds': selected['window_bounds'],
            'selected_from_candidates': True,
            'selected_contact_pixel': (
                int(selected['contact_u']), int(selected['contact_v'])
            ),
            'selected_window_center_pixel': (
                selected_center_u, selected_center_v
            ),
            'channel_mask': channel_height_mask,
            'channel_end_center_pixel': channel_end,
            'channel_max_height': channel_max_height,
            'channel_z_clearance_m': CHANNEL_Z_CLEARANCE_M,
            'final_push_retreat_m': FINAL_PUSH_RETREAT_M,
            'push_pixel_float': (final_push_u, final_push_v),
            'push_z': float(z_value),
            'h_target_p90': selected['h_target_p90'],
            'line_error_px': selected['line_error_px'],
            'clearance_px': selected['clearance_px'],
            'geometry_valid': True,
        })
    return (
        final_push_u, final_push_v,
        float(z_value), float(depth[int(selected['contact_v']), int(selected['contact_u'])])
    )



def _find_dominant_obstacle_mask(seg_map, target_mask, global_mask, center_u, center_v,
                                 direction_idx, target_id, background_id):
    """Choose the action-4--7 intended obstacle using two first-hit ray tiers.

    Tier 1 searches [0, 20 mm]. Tier 2 is consulted only when tier 1 has no
    hit and searches (20, 50 mm]. Within a tier, the widest projected obstacle
    wins; distance only breaks a width tie.
    """
    target_mask = _ensure_uint8_mask(target_mask)
    global_mask = _ensure_uint8_mask(global_mask)

    if seg_map is None or target_mask is None or global_mask is None:
        return None, -1
    if np.sum(target_mask > 0) == 0 or np.sum(global_mask > 0) == 0:
        return None, -1

    push_angle_deg = _get_push_angle_deg(direction_idx)
    du, dv, tu, tv = _cardinal_components(push_angle_deg)
    target = target_mask > 0
    h, w = target.shape
    target_ys, target_xs = np.where(target)
    if target_ys.size == 0:
        return None, -1, {'outcome': 'empty', 'empty_reason': 'empty_target_mask'}

    # Build the +d front contour: one first-hit target pixel per transverse
    # coordinate. This is the same contour used by the two ray tiers.
    edge_line = {}
    for u, v in zip(target_xs.tolist(), target_ys.tolist()):
        lateral = int(round(float(u) * tu + float(v) * tv))
        longitudinal = float(u) * du + float(v) * dv
        if lateral not in edge_line or longitudinal > edge_line[lateral][0]:
            edge_line[lateral] = (longitudinal, int(u), int(v))
    if not edge_line:
        return None, -1, {'outcome': 'empty', 'empty_reason': 'no_target_front_edge'}

    tier_stats = []
    for tier, (min_m, max_m) in enumerate(((0.0, OBSTACLE_RAY_TIER1_M),
                                            (OBSTACLE_RAY_TIER1_M, OBSTACLE_RAY_TIER2_M)), 1):
        hits = {}
        total_rays = len(edge_line)
        max_step = int(np.floor(max_m * PPM + 1e-6))
        min_step = max(1, int(np.floor(min_m * PPM + 1e-6)) + 1)
        for _, (_, edge_u, edge_v) in sorted(edge_line.items()):
            for step in range(min_step, max_step + 1):
                ray_u = int(round(edge_u + du * step))
                ray_v = int(round(edge_v + dv * step))
                if ray_u < 0 or ray_u >= w or ray_v < 0 or ray_v >= h:
                    break
                if target[ray_v, ray_u]:
                    continue
                seg_val = int(seg_map[ray_v, ray_u])
                if (
                    seg_val != int(background_id)
                    and seg_val != int(target_id)
                    and global_mask[ray_v, ray_u] > 0
                ):
                    entry = hits.setdefault(seg_val, [])
                    entry.append(float(step / PPM))
                    break
        stat = {
            'tier': tier,
            'min_distance_m': float(min_m),
            'max_distance_m': float(max_m),
            'ray_count': int(total_rays),
            'instances': {
                int(seg_id): {
                    'hit_count': int(len(distances)),
                    'coverage': float(len(distances) / max(total_rays, 1)),
                    'median_distance_m': float(np.median(distances)),
                    'min_distance_m': float(np.min(distances)),
                }
                for seg_id, distances in hits.items()
            },
        }
        tier_stats.append(stat)
        if hits:
            dominant_seg_id = min(
                hits,
                key=lambda seg_id: (
                    -len(hits[seg_id]),
                    float(np.median(hits[seg_id])),
                    float(np.min(hits[seg_id])),
                    int(seg_id),
                ),
            )
            dominant_mask = (seg_map == dominant_seg_id).astype(np.uint8) * 255
            return dominant_mask, int(dominant_seg_id), {
                'outcome': 'selected',
                'selected_tier': tier,
                'selected_seg_id': int(dominant_seg_id),
                'tier_stats': tier_stats,
            }

    return None, -1, {
        'outcome': 'empty',
        'empty_reason': 'no_obstacle_within_5cm',
        'tier_stats': tier_stats,
    }


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

    # [点1] 记录该动作"第一下应该碰到的物体" seg id, 供 env_wrapper 的接触判定使用。
    #   推目标 (0-3): intended = target_id
    #   推障碍 (4-7): intended = 选中的 dominant 障碍 id (无则 -1)
    geometry_debug = {} if debug_info is not None else None
    geometry_invalid_reason = None
    outcome = 'push'
    empty_reason = None
    obstacle_selection_debug = None
    table_height = getattr(state, '_external_table_height', None)
    camera_height = getattr(state, '_external_camera_height', None)
    if action_idx <= 3:
        # ===== 推目标物体 =====
        direction_idx = action_idx
        push_angle_deg = _get_push_angle_deg(direction_idx)
        migrated_result = _compute_migrated_edge_push(
            depth_map, target_mask, center_u, center_v, push_angle_deg,
            global_mask=global_mask, debug_info=geometry_debug,
            table_height=table_height,
            camera_height=camera_height,
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

        dominant_obstacle_mask, dominant_seg_id, obstacle_selection_debug = _find_dominant_obstacle_mask(
            seg_map=seg_map,
            target_mask=target_mask,
            global_mask=global_mask,
            center_u=center_u,
            center_v=center_v,
            direction_idx=direction_idx,
            target_id=target_id,
            background_id=background_id,
        )

        if obstacle_selection_debug.get('outcome') == 'empty':
            outcome = 'empty'
            empty_reason = obstacle_selection_debug.get(
                'empty_reason', 'no_obstacle_within_5cm'
            )
            push_u = push_v = push_z = None
            intended_seg_id = -1
            intended_mask = np.zeros_like(target_mask, dtype=np.uint8)
            if geometry_debug is not None:
                geometry_debug.clear()
                geometry_debug.update({
                    'candidate_mask': np.zeros_like(target_mask, dtype=bool),
                    'edge_neighborhood_mask': np.zeros_like(target_mask, dtype=bool),
                    'selected_from_candidates': False,
                    'obstacle_selection': obstacle_selection_debug,
                    'geometry_valid': False,
                })
        elif dominant_seg_id != -1:
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
                    table_height=table_height,
                    camera_height=camera_height,
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
        if outcome != 'empty':
            intended_seg_id = int(dominant_seg_id)  # -1 表示未选到障碍
        intended_mask = (
            dominant_obstacle_mask
            if dominant_obstacle_mask is not None
            else np.zeros_like(target_mask, dtype=np.uint8)
        ) if outcome != 'empty' else intended_mask

    # 没有安全几何推点时不产生任何默认运动点，由环境按空推处理。
    geometry_valid = push_u is not None and push_v is not None and push_z is not None
    if outcome == 'empty':
        geometry_valid = False
    elif not geometry_valid:
        outcome = 'invalid_geometry'

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
        device = getattr(state, 'device', None)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        push_point = torch.tensor([push_x, push_y, push_z], dtype=torch.float32, device=device)

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
        'outcome': outcome,
        'empty_push': outcome == 'empty',
        'empty_reason': empty_reason,
        'obstacle_selection': obstacle_selection_debug,
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
            'outcome': outcome,
            'empty_reason': empty_reason,
            'obstacle_selection': obstacle_selection_debug,
        })

    return push_point, direction_idx, contact_spec
