"""
成功分离判定（无状态 2.5D 版本）。

仅依赖相机俯拍 RGB + Depth + Segmentation，每 step 从头计算。
sim 与真机使用同一套逻辑。

判定流程：
  1. target_mask × depth → z_map (物体高度图)
  2. dist_map = distanceTransform(target_mask)
  3. score = z_map^Z_POW × dist_map^DIST_POW, anchor_uv = argmax(score)
       → 物理意义：物体最厚 + 最不靠边的位置 = 双指夹最该夹的部位
  4. dist² 加权全局 PCA → 短轴 = grasp_axis (夹爪开合方向),
                            长轴 = body_axis  (物体延伸方向)
       (附件如瓶颈/把手的 dist 值小, dist² 让主体主导)
  5. 沿 grasp / body 双向 walk_to_boundary 得到半宽 / 半长
  6. 在 anchor 两侧画两块清空带, 尺寸与 R_2F_140 物理参数对齐
  7. 短清空带要求整条两侧清空; 长清空带允许左/中/右任一段两侧清空 → 成功

宿主对象需提供: scene, num_envs, env_steps, depth_debug_dir, depth_debug_episode, target_grasp_specs。
"""

import os


# ====== 物理参数 ======
PIXELS_PER_METER = 320.0 / 1.0          # state.world_to_pixel 的固定比例 (320 像素 = 1 m)
MAX_JAW_OPENING = 0.14                  # R_2F_140 最大开合 (m)
GRIPPER_FINGER_THICKNESS = 0.02        # 单指厚度 (m), grasp 方向 anchor → 清空带的内偏移
CLEAR_BAND_DEPTH = 0.022                # 单侧清空带宽度 (m), 指头插入空间, 沿 grasp_axis
BODY_EXTRA_PADDING = 0.01              # body 方向余量 (m), 主干外稍微外扩

# ====== 算法参数 ======
WALK_BOUNDARY_TOLERANCE = 3             # 连续 N 像素出 mask 视为边界
WALK_MAX_STEPS = 80
Z_POW = 1.0                             # score 中 z 的指数
DIST_POW = 2.0                          # score 中 dist 的指数
ANCHOR_PLATEAU_RATIO = 0.95             # score >= ratio*peak 的像素视为平台, 取重心做锚点
NOISE_THRESHOLD = 15                    # 障碍物像素噪声容忍
LONG_CLEAR_BAND_THRESHOLD = 0.09        # 清空带长于该值时启用分段判定 (m)
CLEAR_BAND_SEGMENT_LENGTH = 0.08        # 长清空带左/中/右单段长度 (m)
MIN_TARGET_AREA = 100                   # mask 太小直接 False

# ====== 主干段识别 (沿 body_axis 找质量集中段, 清空带覆盖该段两侧) ======
# ====== 主干段识别 (沿 body_axis 找 dist_map 高的连续段, 即"质量最集中"段) ======
# 方法: 用 distanceTransform 的局部峰值做绝对阈值。
#       dist_map[v,u] = 该像素到 mask 边界的最大距离, 物理意义直接对应"质量集中度":
#         · 瓶肚 (粗) → 中心远离边界, dist_max 大
#         · 瓶颈 (细) → 再厚也就是半径那么远, dist_max 小
#         · 均匀盒子 → 整段 dist_max 接近 peak, 主干 = 整体
TRUNK_BIN_PX = 2                        # 切片宽度 (像素, ~6mm)
TRUNK_DIST_RATIO = 0.5                  # bin 的 dist_max ≥ ratio×全局 peak 视为主干
                                        # 0.5: 圆/椭圆主体边缘也保留 (主干≈整个瓶肚)
                                        # 0.7: 只取核心区域 (主干会缩到瓶肚中段)
TRUNK_MIN_BIN_COUNT = 3                 # bin 内至少有几个像素才参与统计
TRUNK_MIN_HALF_PX = 5                   # 主干半长下限 (避免主干过短)

# ====== 双轴判定 (近方形物体, 任一对边清空都算成功) ======
ASPECT_SQUARE_THRESHOLD = 1.35          # PCA aspect ≤ 此值视为近方形, 启用双轴尝试


class SuccessSeparationMixin:
    """
    PushEnv 成功分离判定 mixin (无状态 2.5D)。
    """

    # ============================================================
    #  入口 1: reset 时调用 (兼容现有调用方, 但本实现不缓存任何东西)
    # ============================================================
    def _precompute_target_grasp_specs(self, spawned_objects, env_ids=None):
        """
        [功能]: 兼容旧接口。无状态实现下不预计算模板，仅清空缓存槽位。
                可选地保存 reset 时的可视化（锚点 + 清空带）便于调试。
        """
        if env_ids is None:
            env_ids = range(self.num_envs)
        for env_idx in env_ids:
            self.target_grasp_specs[env_idx] = None
            if self.depth_debug_dir is not None:
                # 走一遍判定逻辑只为了出 debug 图
                try:
                    self._check_successful_separation(
                        env_idx, spawned_objects, debug_tag="grasp_spec"
                    )
                except Exception as e:
                    print(f"  [grasp_spec debug] env {env_idx} 失败: {e}")

    # ============================================================
    #  入口 2: 每个 step 调用
    # ============================================================
    def _check_successful_separation(self, env_idx, spawned_objects,
                                     debug_tag=None):
        """
        [功能]: 判定目标物体两侧抓取位置是否已清空。
        [输入]:
            env_idx (int)
            spawned_objects (list)
            debug_tag (str|None): 调试图前缀, "grasp_spec" 表示来自 reset。
        [输出]: (success: bool, max_required_obstacle: int, detail_str: str)
        """
        import numpy as np

        state = self.scene.states[env_idx]

        # ── 取图 ──
        result = state.get_img(hide_robot=True, normalize_depth=False)
        if result is None:
            return False, 0, "无图像"
        rgb_img, depth_img, seg_img = result
        if seg_img is None or depth_img is None:
            return False, 0, "无分割/深度图"

        # ── target / obstacle mask ──
        target_mask = state.extract_target_mask(seg_img, spawned_objects)
        if target_mask is None or int(np.sum(target_mask > 0)) < MIN_TARGET_AREA:
            return False, 0, f"目标掩膜过小 (<{MIN_TARGET_AREA})"

        all_ids, target_id = state.get_object_ids(seg_img, spawned_objects)
        if not all_ids or target_id is None:
            return False, 0, "无物体ID"
        obstacle_mask = state.build_obstacle_mask(seg_img, all_ids - {target_id})

        # ── 调试 dump: 把原始输入存成 .npy, 离线用 debug_grasp_anchor.py 复现 ──
        # 在宿主上设 self.success_dump_dir = "/tmp/success_dump" 即可启用
        dump_dir = getattr(self, "success_dump_dir", None)
        if dump_dir:
            self._dump_inputs(dump_dir, env_idx, rgb_img, depth_img,
                              target_mask, obstacle_mask)

        # ── 计算锚点 + 抓取方向 + 清空带 ──
        primary = self._analyze_grasp_geometry(target_mask, depth_img)
        if primary is None:
            return False, 0, "几何分析失败"

        # 近方形 (aspect ≤ ASPECT_SQUARE_THRESHOLD) 时同时尝试副方向 (互换 grasp/body)
        candidates = [primary]
        alt = primary.get("alt_axis_analysis")
        if alt is not None:
            candidates.append(alt)

        # 任一候选满足两侧清空即 success; 否则取 max_required 最小的当 best (供详情/可视化)
        success_entry = None
        best_entry = None
        for cand in candidates:
            regions = self._build_clear_regions(target_mask.shape, cand)
            eval_result = self._evaluate_clear_regions(obstacle_mask, regions)
            entry = {
                "analysis": cand, "regions": regions,
                "pos_count": eval_result["pos_count"],
                "neg_count": eval_result["neg_count"],
                "max_required": eval_result["max_required"],
                "is_clear": eval_result["is_clear"],
                "segment_tag": eval_result["segment_tag"],
                "band_length_m": eval_result["band_length_m"],
            }
            if best_entry is None or entry["max_required"] < best_entry["max_required"]:
                best_entry = entry
            if entry["is_clear"]:
                success_entry = entry
                break

        chosen = success_entry if success_entry is not None else best_entry
        analysis = chosen["analysis"]
        regions = chosen["regions"]
        pos_count = chosen["pos_count"]
        neg_count = chosen["neg_count"]
        max_required = chosen["max_required"]
        success = chosen["is_clear"]
        segment_tag = chosen.get("segment_tag", "whole")

        # ── 详情字符串 ──
        anchor = analysis["anchor_uv"]
        ga = analysis["grasp_axis"]
        jaw_w_mm = int(round(analysis["jaw_full_width_m"] * 1000))
        z_at_anchor = analysis["z_at_anchor_m"]
        edge_angle = float(np.degrees(np.arctan2(ga[1], ga[0])))
        trunk_lo = analysis.get("trunk_lo_px", 0.0)
        trunk_hi = analysis.get("trunk_hi_px", 0.0)
        trunk_len_mm = int(round((trunk_hi - trunk_lo) / PIXELS_PER_METER * 1000))
        is_square = bool(primary.get("is_square_like", False))
        axis_tag = ""
        if is_square:
            axis_tag = "[A]" if analysis.get("is_primary", True) else "[B]"
        detail_str = (
            f"锚:({int(anchor[0])},{int(anchor[1])}) {axis_tag}"
            f"夹宽:{jaw_w_mm}mm 主干:{trunk_len_mm}mm h={z_at_anchor*1000:.0f}mm "
            f"角:{edge_angle:.1f} "
            f"清:{int(success)} 段={segment_tag} [正={pos_count},负={neg_count}]"
        )

        # ── 调试图 ──
        if self.depth_debug_dir is not None:
            self._save_success_debug(
                env_idx=env_idx,
                rgb_img=rgb_img,
                target_mask=target_mask,
                obstacle_mask=obstacle_mask,
                analysis=analysis,
                regions=regions,
                pos_count=pos_count,
                neg_count=neg_count,
                success=success,
                detail_str=detail_str,
                segment_tag=segment_tag,
                tag=debug_tag,
            )

        return success, max_required, detail_str

    # ============================================================
    #  几何分析: 锚点 + 抓取轴 + 半宽 / 半长
    # ============================================================
    def _analyze_grasp_geometry(self, target_mask, depth_img):
        """
        [功能]: 在 mask + depth 上找抓取锚点和方向。
        [输出]: dict 或 None
            {
                "anchor_uv":          (u, v) int,
                "grasp_axis":         (gx, gy) unit, 夹爪开合方向 (短轴),
                "body_axis":          (bx, by) unit, 物体延伸方向 (长轴),
                "jaw_half_w_px":      float, 锚点处沿 grasp_axis 的半宽 (像素),
                "jaw_full_width_m":   float (米),
                "z_at_anchor_m":      float, 锚点高度 (米),
                "table_depth":        float, 桌面深度 (米),
                "z_map":              np.ndarray (H,W) float32, 物体高度图,
                "dist_map":           np.ndarray (H,W) float32, 距离场,
                "out_of_jaw":         bool, 物体太宽超出夹爪开合,
            }
        """
        import cv2
        import numpy as np

        binary = (target_mask > 0).astype(np.uint8)
        if int(np.sum(binary)) < MIN_TARGET_AREA:
            return None

        # ── z_map: 物体高度 ──
        table_depth = self._estimate_table_depth(depth_img, binary)
        if table_depth is None:
            return None
        depth = np.asarray(depth_img, dtype=np.float32)
        z_map = np.clip(table_depth - depth, 0.0, None).astype(np.float32)
        z_map = z_map * (binary > 0)

        # ── dist_map: 距边界 ──
        dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
        if float(np.max(dist_map)) < 1.0:
            return None

        # ── 锚点: argmax(z^Z_POW · dist^DIST_POW) ──
        # z 全 0 时退化为只用 dist (深度异常的兜底)
        if float(np.max(z_map)) < 1e-4:
            score = dist_map ** DIST_POW
        else:
            score = (z_map ** Z_POW) * (dist_map ** DIST_POW)
        # 平台重心: argmax 单点对均匀厚度物体不稳 (矩形骨架是常数线),
        # 改取 score >= 0.95*peak 的所有像素的几何中心。
        score_max = float(np.max(score))
        if score_max < 1e-9:
            return None
        plateau = score >= (ANCHOR_PLATEAU_RATIO * score_max)
        ys_p, xs_p = np.nonzero(plateau)
        if xs_p.size == 0:
            anchor_idx = int(np.argmax(score))
            av, au = np.unravel_index(anchor_idx, score.shape)
        else:
            au = int(round(float(np.mean(xs_p))))
            av = int(round(float(np.mean(ys_p))))
            # 平台重心可能落在外面 (凹形物体), 落回 plateau 上最近的点
            if not plateau[av, au]:
                d2 = (xs_p - au) ** 2 + (ys_p - av) ** 2
                idx = int(np.argmin(d2))
                au, av = int(xs_p[idx]), int(ys_p[idx])
        anchor_uv = (int(au), int(av))

        # ── 主轴: dist² 加权的全局 PCA ──
        # 附件 (瓶颈/瓶盖/把手) 的 dist 值小, dist² 权重让主体主导;
        # anchor 自身位置已由 z·dist² argmax 决定, 与 PCA 解耦。
        ys, xs = np.nonzero(binary)
        points = np.column_stack([xs, ys]).astype(np.float32)
        weights = np.maximum(dist_map[ys, xs], 1e-3) ** 2
        pca = self._weighted_pca_2d(points, weights)
        if pca is None:
            return None

        _center, main_axis, second_axis, eigvals = pca
        # aspect ≥ 1: 主轴特征值 / 次轴特征值 开根号. =1 完美方形, 越大越长条
        if eigvals[1] > 1e-9:
            aspect = float(np.sqrt(eigvals[0] / eigvals[1]))
        else:
            aspect = 99.0
        is_square_like = aspect <= ASPECT_SQUARE_THRESHOLD

        # 主方向: 短轴 = grasp (夹爪开合), 长轴 = body (物体延伸)
        primary = self._build_axis_analysis(
            anchor_uv=anchor_uv,
            grasp_axis=second_axis,
            body_axis=main_axis,
            binary=binary,
            z_map=z_map,
            dist_map=dist_map,
            table_depth=table_depth,
        )
        if primary is None:
            return None
        primary["aspect"] = aspect
        primary["is_primary"] = True
        primary["is_square_like"] = is_square_like

        if is_square_like:
            # 副方向: 互换 grasp / body
            secondary = self._build_axis_analysis(
                anchor_uv=anchor_uv,
                grasp_axis=main_axis,
                body_axis=second_axis,
                binary=binary,
                z_map=z_map,
                dist_map=dist_map,
                table_depth=table_depth,
            )
            if secondary is not None:
                secondary["aspect"] = aspect
                secondary["is_primary"] = False
                secondary["is_square_like"] = is_square_like
                primary["alt_axis_analysis"] = secondary
        return primary

    def _build_axis_analysis(self, anchor_uv, grasp_axis, body_axis,
                             binary, z_map, dist_map, table_depth):
        """
        [功能]: 给定一对 (grasp_axis, body_axis), 测 anchor 处的 jaw 宽度 + 主干段,
                返回 analysis dict (与原 _analyze_grasp_geometry 输出格式相同, 不含 aspect 等元字段)。
        """
        import numpy as np

        jaw_half_w_px = self._walk_to_boundary(anchor_uv, grasp_axis, binary)
        if jaw_half_w_px < 1.0:
            return None

        jaw_full_width_m = (2.0 * jaw_half_w_px) / PIXELS_PER_METER
        out_of_jaw = jaw_full_width_m > MAX_JAW_OPENING

        trunk_lo_px, trunk_hi_px = self._compute_trunk_extent(
            binary, dist_map, anchor_uv, body_axis,
        )

        return {
            "anchor_uv":         anchor_uv,
            "grasp_axis":        (float(grasp_axis[0]), float(grasp_axis[1])),
            "body_axis":         (float(body_axis[0]),  float(body_axis[1])),
            "jaw_half_w_px":     float(jaw_half_w_px),
            "jaw_full_width_m":  float(jaw_full_width_m),
            "z_at_anchor_m":     float(z_map[anchor_uv[1], anchor_uv[0]]),
            "table_depth":       float(table_depth),
            "z_map":             z_map,
            "dist_map":          dist_map,
            "out_of_jaw":        bool(out_of_jaw),
            "trunk_lo_px":       float(trunk_lo_px),
            "trunk_hi_px":       float(trunk_hi_px),
        }

    # ============================================================
    #  主干段识别: 沿 body_axis 找 dist_map 高的连续段
    # ============================================================
    @staticmethod
    def _compute_trunk_extent(binary, dist_map, anchor_uv, body_axis):
        """
        [功能]: 沿 body_axis 把 mask 切成 TRUNK_BIN_PX 像素宽切片,
                每片取 dist_max(b) = 该切片像素中 distanceTransform 的最大值;
                dist_max(b) ≥ TRUNK_DIST_RATIO × peak 视为主干。
                从 anchor bin 起向两侧连续扩展, 遇到非主干 bin 即停。

                物理含义: dist_max 直接代表"该切片中心离 mask 边界的最大距离",
                          瓶颈细处再粗也只能到管半径; 瓶肚厚处中心远离边界。
                          所以这是"质量集中"的最直接代理, 用全局 peak 做绝对阈值。

        [输出]: (trunk_lo_px, trunk_hi_px) — anchor 局部坐标 (沿 body_axis)。
                返回 (0, 0) 表示主干识别失败 (调用方退化到对称小范围)。
        """
        import numpy as np

        ys, xs = np.nonzero(binary)
        if xs.size < 5:
            return 0.0, 0.0

        au, av = anchor_uv
        bx, by = float(body_axis[0]), float(body_axis[1])
        dx = xs.astype(np.float32) - float(au)
        dy = ys.astype(np.float32) - float(av)
        along_body = dx * bx + dy * by
        dist_at = dist_map[ys, xs].astype(np.float32)

        bin_w = float(TRUNK_BIN_PX)
        bin_origin = float(np.min(along_body))
        bin_max    = float(np.max(along_body))
        n_bins = max(1, int(np.ceil((bin_max - bin_origin) / bin_w)))
        bin_idx = np.clip(((along_body - bin_origin) / bin_w).astype(np.int32),
                          0, n_bins - 1)

        # 每 bin 取 dist_max
        dist_max = np.zeros(n_bins, dtype=np.float32)
        valid    = np.zeros(n_bins, dtype=bool)
        for b in range(n_bins):
            in_b = (bin_idx == b)
            if int(np.sum(in_b)) < TRUNK_MIN_BIN_COUNT:
                continue
            dist_max[b] = float(np.max(dist_at[in_b]))
            valid[b] = True

        peak = float(np.max(dist_max))
        if peak < 1.0:
            return 0.0, 0.0

        is_trunk = valid & (dist_max >= TRUNK_DIST_RATIO * peak)

        # 锚点 bin: along_body=0 落入哪一格 (anchor 来自 z·dist² argmax, 通常 dist_max[anchor_bin] ≈ peak)
        anchor_bin = int(np.clip(int(np.floor((0.0 - bin_origin) / bin_w)),
                                 0, n_bins - 1))
        if not valid[anchor_bin]:
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size == 0:
                return 0.0, 0.0
            anchor_bin = int(valid_idx[np.argmin(np.abs(valid_idx - anchor_bin))])
        is_trunk[anchor_bin] = True   # 锚点必属主干

        # 从锚点 bin 起向两侧连续扩展, 遇到非主干 bin 即停
        hi_bin = anchor_bin
        for b in range(anchor_bin + 1, n_bins):
            if not is_trunk[b]:
                break
            hi_bin = b
        lo_bin = anchor_bin
        for b in range(anchor_bin - 1, -1, -1):
            if not is_trunk[b]:
                break
            lo_bin = b

        trunk_lo_px = bin_origin + lo_bin * bin_w
        trunk_hi_px = bin_origin + (hi_bin + 1) * bin_w

        # 主干太短的兜底 (避免清空带退化为零长)
        min_half = float(TRUNK_MIN_HALF_PX)
        if trunk_hi_px < min_half:
            trunk_hi_px = min_half
        if trunk_lo_px > -min_half:
            trunk_lo_px = -min_half

        return float(trunk_lo_px), float(trunk_hi_px)

    # ============================================================
    #  清空带 mask
    # ============================================================
    def _build_clear_regions(self, image_shape, analysis):
        """
        [功能]: 在主干段两侧画清空带 (夹爪指头插入空间)。
                几何 (anchor 局部坐标):
                  body 方向:  [trunk_lo_px - pad, trunk_hi_px + pad]   (跟随主干段, 瓶颈/把手被剔除)
                  grasp 方向: 主体 ±jaw_half_w_px, 两侧 inner→outer    (anchor 处物体宽度 + 指头插入空间)
        [输出]: {
            "body": HxW bool, "clear_pos": HxW bool, "clear_neg": HxW bool,
            "along_body": HxW float32, "body_lo_px": float, "body_hi_px": float
        }
        """
        import numpy as np

        h, w = image_shape
        au, av = analysis["anchor_uv"]
        gx, gy = analysis["grasp_axis"]
        bx, by = analysis["body_axis"]
        jaw_half_w_px = float(analysis["jaw_half_w_px"])

        # 清空带几何 (米 → 像素)
        finger_half_px = (GRIPPER_FINGER_THICKNESS * 0.5) * PIXELS_PER_METER
        inner_offset_px = jaw_half_w_px + finger_half_px
        outer_offset_px = inner_offset_px + CLEAR_BAND_DEPTH * PIXELS_PER_METER

        # body 范围由主干段决定; 主干识别失败时退化到对称小范围
        trunk_lo = float(analysis.get("trunk_lo_px", 0.0))
        trunk_hi = float(analysis.get("trunk_hi_px", 0.0))
        pad = BODY_EXTRA_PADDING * PIXELS_PER_METER
        if trunk_hi - trunk_lo <= 0.0:
            half = max(TRUNK_MIN_HALF_PX, jaw_half_w_px)
            body_lo, body_hi = -half - pad, half + pad
        else:
            body_lo = trunk_lo - pad
            body_hi = trunk_hi + pad

        yy, xx = np.mgrid[0:h, 0:w]
        du = xx.astype(np.float32) - float(au)
        dv = yy.astype(np.float32) - float(av)
        along_grasp = du * gx + dv * gy
        along_body = du * bx + dv * by

        inside_body = (along_body >= body_lo) & (along_body <= body_hi)
        body = inside_body & (np.abs(along_grasp) <= jaw_half_w_px)
        clear_pos = inside_body & (along_grasp >= inner_offset_px) & (along_grasp <= outer_offset_px)
        clear_neg = inside_body & (along_grasp <= -inner_offset_px) & (along_grasp >= -outer_offset_px)
        return {
            "body": body,
            "clear_pos": clear_pos,
            "clear_neg": clear_neg,
            "along_body": along_body,
            "body_lo_px": float(body_lo),
            "body_hi_px": float(body_hi),
        }

    @staticmethod
    def _evaluate_clear_regions(obstacle_mask, regions):
        """
        [功能]: 统计清空带障碍像素。
                短清空带按整条判定; 长清空带检查 left/center/right 三个 0.07m 段,
                任意一段两侧障碍像素都小于 NOISE_THRESHOLD 即成功。
        [输出]: dict, 含 is_clear / pos_count / neg_count / max_required / segment_tag / band_length_m
        """
        import numpy as np

        obstacle = np.asarray(obstacle_mask) > 0
        clear_pos = np.asarray(regions["clear_pos"], dtype=bool)
        clear_neg = np.asarray(regions["clear_neg"], dtype=bool)
        body_lo = float(regions.get("body_lo_px", 0.0))
        body_hi = float(regions.get("body_hi_px", 0.0))
        band_length_px = max(0.0, body_hi - body_lo)
        band_length_m = band_length_px / PIXELS_PER_METER

        def count_pair(segment_mask, tag):
            pos_count = int(np.sum(obstacle & clear_pos & segment_mask))
            neg_count = int(np.sum(obstacle & clear_neg & segment_mask))
            max_required = max(pos_count, neg_count)
            return {
                "is_clear": (pos_count < NOISE_THRESHOLD) and (neg_count < NOISE_THRESHOLD),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "max_required": max_required,
                "segment_tag": tag,
                "band_length_m": float(band_length_m),
            }

        if band_length_m <= LONG_CLEAR_BAND_THRESHOLD:
            return count_pair(np.ones_like(clear_pos, dtype=bool), "whole")

        along_body = np.asarray(regions["along_body"], dtype=np.float32)
        seg_len_px = CLEAR_BAND_SEGMENT_LENGTH * PIXELS_PER_METER
        center = (body_lo + body_hi) * 0.5
        segment_specs = (
            ("left", body_lo, body_lo + seg_len_px),
            ("center", center - seg_len_px * 0.5, center + seg_len_px * 0.5),
            ("right", body_hi - seg_len_px, body_hi),
        )

        best = None
        for tag, seg_lo, seg_hi in segment_specs:
            segment_mask = (along_body >= seg_lo) & (along_body <= seg_hi)
            result = count_pair(segment_mask, tag)
            if best is None or result["max_required"] < best["max_required"]:
                best = result
            if result["is_clear"]:
                return result
        return best

    # ============================================================
    #  辅助: 加权 PCA / walk_to_boundary / 桌面深度估计
    # ============================================================
    @staticmethod
    def _weighted_pca_2d(points_xy, weights):
        """
        [功能]: 加权 2D PCA, 返回 (center, main_axis, second_axis, eigvals)。
                second_axis 强制取与 main_axis 严格垂直, 避免数值噪声。
        """
        import numpy as np

        points = np.asarray(points_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3:
            return None
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weights.size != points.shape[0]:
            weights = np.ones(points.shape[0], dtype=np.float32)
        weights = np.maximum(weights, 1e-6)
        wsum = float(np.sum(weights))
        if wsum <= 1e-6:
            return None

        center = np.sum(points * weights[:, None], axis=0) / wsum
        centered = points - center
        cov = (centered * weights[:, None]).T @ centered / wsum
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        main_axis = eigvecs[:, 0]
        norm = float(np.linalg.norm(main_axis))
        if norm < 1e-6:
            return None
        main_axis = main_axis / norm
        second_axis = np.array([-main_axis[1], main_axis[0]], dtype=np.float32)
        return (center.astype(np.float32),
                main_axis.astype(np.float32),
                second_axis,
                eigvals.astype(np.float32))

    @staticmethod
    def _walk_to_boundary(anchor_uv, axis, binary_mask):
        """
        [功能]: 从 anchor 沿 ±axis 方向步进, 连续 WALK_BOUNDARY_TOLERANCE 个像素出 mask 即视为边界。
                返回较短一侧的半长 (保守取较小值)。
        [输出]: float (像素)
        """
        import numpy as np

        h, w = binary_mask.shape
        au, av = anchor_uv
        ax = float(axis[0])
        ay = float(axis[1])
        norm = (ax * ax + ay * ay) ** 0.5
        if norm < 1e-6:
            return 0.0
        ax /= norm
        ay /= norm

        half_lengths = []
        for sign in (1.0, -1.0):
            consecutive_out = 0
            last_inside = 0
            for d in range(1, WALK_MAX_STEPS + 1):
                u = int(round(au + sign * d * ax))
                v = int(round(av + sign * d * ay))
                if u < 0 or u >= w or v < 0 or v >= h:
                    consecutive_out += 1
                    if consecutive_out >= WALK_BOUNDARY_TOLERANCE:
                        break
                    continue
                if binary_mask[v, u] > 0:
                    last_inside = d
                    consecutive_out = 0
                else:
                    consecutive_out += 1
                    if consecutive_out >= WALK_BOUNDARY_TOLERANCE:
                        break
            half_lengths.append(float(last_inside))
        if not half_lengths:
            return 0.0
        return float(min(half_lengths))

    @staticmethod
    def _estimate_table_depth(depth_img, target_binary):
        """
        [功能]: 从背景深度直方图估计桌面深度 (米单位)。
        [输出]: float 或 None
        """
        import numpy as np

        if depth_img is None:
            return None
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim != 2:
            return None
        valid = np.isfinite(depth) & (depth > 0)
        background = valid & ~(target_binary > 0)
        values = depth[background]
        if values.size < 50:
            values = depth[valid]
        if values.size < 50:
            return None

        # 离散值 (Isaac depth 经常是少量离散桌面/地板值)
        unique_vals, counts = np.unique(values, return_counts=True)
        if unique_vals.size <= 2048:
            top_count = int(np.max(counts))
            top_mask = counts >= max(5, int(top_count * 0.25))
            return float(np.min(unique_vals[top_mask]))

        lo, hi = np.percentile(values, [1, 99])
        if hi - lo < 1e-6:
            return float(np.median(values))
        hist, edges = np.histogram(values, bins=128, range=(lo, hi))
        if int(np.max(hist)) <= 0:
            return float(np.median(values))
        top_mask = hist >= max(5, int(np.max(hist) * 0.25))
        centers = (edges[:-1] + edges[1:]) * 0.5
        return float(np.min(centers[top_mask]))

    # ============================================================
    #  调试可视化
    # ============================================================
    def _dump_inputs(self, dump_dir, env_idx, rgb_img, depth_img,
                     target_mask, obstacle_mask):
        """
        [功能]: 把当前 step 的原始输入存成 .npy + .png, 离线用 debug_grasp_anchor.py 复现。
        [文件]: <dump>/ep{ep}_env{i}_step{s}_{rgb.png|depth.npy|target.npy|obstacle.npy}
        """
        import cv2
        import numpy as np

        os.makedirs(dump_dir, exist_ok=True)
        step = int(self.env_steps[env_idx].item()) if hasattr(self.env_steps, 'item') \
            else int(self.env_steps[env_idx])
        ep = int(self.depth_debug_episode)
        prefix = f"ep{ep}_env{env_idx}_step{step}"
        rgb_vis = self._normalize_rgb_for_debug(rgb_img)
        cv2.imwrite(os.path.join(dump_dir, f"{prefix}_rgb.png"),
                    cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(dump_dir, f"{prefix}_depth.npy"),
                np.asarray(depth_img, dtype=np.float32))
        np.save(os.path.join(dump_dir, f"{prefix}_target.npy"),
                (np.asarray(target_mask) > 0).astype(np.uint8) * 255)
        np.save(os.path.join(dump_dir, f"{prefix}_obstacle.npy"),
                (np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255)

    @staticmethod
    def _normalize_rgb_for_debug(rgb_img):
        import cv2
        import numpy as np

        rgb_vis = rgb_img.copy()
        if rgb_vis.ndim == 2:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_GRAY2RGB)
        if rgb_vis.shape[-1] == 4:
            rgb_vis = rgb_vis[..., :3]
        if rgb_vis.dtype != np.uint8:
            if rgb_vis.max() <= 1.0:
                rgb_vis = (rgb_vis * 255).astype(np.uint8)
            else:
                rgb_vis = np.clip(rgb_vis, 0, 255).astype(np.uint8)
        return rgb_vis

    def _save_success_debug(self, env_idx, rgb_img, target_mask, obstacle_mask,
                            analysis, regions, pos_count, neg_count, success,
                            detail_str, segment_tag="whole", tag=None):
        """
        [功能]: 保存判定可视化: RGB + 目标轮廓 + 障碍物 + 锚点 + 抓取轴 + 两块清空带。
        """
        import cv2
        import numpy as np

        save_dir = self.depth_debug_dir
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)

        step = int(self.env_steps[env_idx].item()) if hasattr(self.env_steps, 'item') \
            else int(self.env_steps[env_idx])
        ep = int(self.depth_debug_episode)
        status_tag = "success" if success else "fail"
        prefix_tag = tag if tag is not None else "success_check"
        prefix = (f"{prefix_tag}_ep{ep}_env{env_idx}_step{step}_{status_tag}"
                  f"_pos{pos_count}_neg{neg_count}")

        rgb_vis = self._normalize_rgb_for_debug(rgb_img)
        target_vis = ((target_mask > 0).astype(np.uint8) * 255)
        obstacle_vis = ((obstacle_mask > 0).astype(np.uint8) * 255)
        overlay = rgb_vis.copy()

        # body / 清空带颜色叠加
        region_colors = {
            "body":      (0, 255, 80),
            "clear_pos": (255, 210, 0),
            "clear_neg": (0, 220, 255),
        }
        region_alphas = {"body": 0.25, "clear_pos": 0.45, "clear_neg": 0.45}
        for name in ("body", "clear_pos", "clear_neg"):
            region_mask = regions[name]
            mask = region_mask.astype(bool)
            if not np.any(mask):
                continue
            color = np.array(region_colors[name], dtype=np.float32)
            a = region_alphas[name]
            overlay[mask] = (overlay[mask].astype(np.float32) * (1.0 - a)
                             + color * a).astype(np.uint8)

        # 障碍物高亮 (红)
        if np.any(obstacle_mask > 0):
            overlay[obstacle_mask > 0] = (
                overlay[obstacle_mask > 0].astype(np.float32) * 0.35
                + np.array([255, 0, 0], dtype=np.float32) * 0.65
            ).astype(np.uint8)

        # 目标轮廓 (绿) / 障碍物轮廓 (红)
        target_contours, _ = cv2.findContours(target_vis, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, target_contours, -1, (0, 255, 0), 2)
        obstacle_contours, _ = cv2.findContours(obstacle_vis, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, obstacle_contours, -1, (255, 0, 0), 1)

        # 清空带轮廓白色加粗
        for name in ("clear_pos", "clear_neg"):
            band_vis = regions[name].astype(np.uint8) * 255
            band_contours, _ = cv2.findContours(band_vis, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, band_contours, -1, (255, 255, 255), 2)

        # 抓取轴 + 锚点
        au, av = analysis["anchor_uv"]
        gx, gy = analysis["grasp_axis"]
        bx, by = analysis["body_axis"]
        jaw_half_w_px = float(analysis["jaw_half_w_px"])
        anchor_pt = (int(au), int(av))

        # body 主干段 (白线): 显示主干 [trunk_lo, trunk_hi] 范围 (锚点局部坐标)
        trunk_lo = float(analysis.get("trunk_lo_px", 0.0))
        trunk_hi = float(analysis.get("trunk_hi_px", 0.0))
        if trunk_hi > trunk_lo:
            p1 = (int(round(au + bx * trunk_lo)), int(round(av + by * trunk_lo)))
            p2 = (int(round(au + bx * trunk_hi)), int(round(av + by * trunk_hi)))
            cv2.line(overlay, p1, p2, (255, 255, 255), 2)

        # grasp 短轴 (粉线), 标记 ±jaw_half_w_px
        g1 = (int(round(au - gx * jaw_half_w_px)),
              int(round(av - gy * jaw_half_w_px)))
        g2 = (int(round(au + gx * jaw_half_w_px)),
              int(round(av + gy * jaw_half_w_px)))
        cv2.line(overlay, g1, g2, (255, 0, 180), 2)
        cv2.circle(overlay, g1, 3, (255, 0, 180), -1)
        cv2.circle(overlay, g2, 3, (255, 0, 180), -1)

        cv2.circle(overlay, anchor_pt, 4, (255, 255, 255), -1)
        cv2.circle(overlay, anchor_pt, 6, (0, 0, 0), 1)

        # 双轴模式: alt 候选用淡蓝/淡黄轴线标出, 便于诊断
        # (chosen 可能是 primary 也可能是 alt; 这里画的是"另一条候选"作参考)
        is_square = bool(analysis.get("is_square_like", False))
        if is_square:
            # 找出与 chosen 对应的"另一条"轴线 (与 chosen.body_axis 垂直, 长度同主干)
            alt_grasp = (-by, bx)  # body 的垂直方向 = 另一组 grasp_axis
            alt_body = (gx, gy)    # chosen 的 grasp = 另一组 body_axis
            ag1 = (int(round(au - alt_grasp[0] * jaw_half_w_px)),
                   int(round(av - alt_grasp[1] * jaw_half_w_px)))
            ag2 = (int(round(au + alt_grasp[0] * jaw_half_w_px)),
                   int(round(av + alt_grasp[1] * jaw_half_w_px)))
            cv2.line(overlay, ag1, ag2, (180, 180, 0), 1)   # 淡黄, 细线
            ab1 = (int(round(au - alt_body[0] * 12)),
                   int(round(av - alt_body[1] * 12)))
            ab2 = (int(round(au + alt_body[0] * 12)),
                   int(round(av + alt_body[1] * 12)))
            cv2.line(overlay, ab1, ab2, (180, 180, 180), 1)  # 淡灰, 细线

        trunk_len_mm = int(round((trunk_hi - trunk_lo) / PIXELS_PER_METER * 1000))
        axis_label = ""
        if is_square:
            axis_label = " [A]" if analysis.get("is_primary", True) else " [B]"
        # 文字
        text_lines = [
            f"{status_tag}{axis_label} seg={segment_tag} pos={pos_count} neg={neg_count} thr={NOISE_THRESHOLD}",
            f"jaw={int(round(analysis['jaw_full_width_m']*1000))}mm "
            f"trunk={trunk_len_mm}mm "
            f"h={analysis['z_at_anchor_m']*1000:.0f}mm "
            f"out_of_jaw={int(analysis['out_of_jaw'])}",
        ]
        if detail_str:
            text_lines.append(detail_str[:96])
        for idx, text in enumerate(text_lines):
            cv2.putText(overlay, text, (8, 18 + idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(save_dir, f"{prefix}_rgb.png"),
                    cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_target_mask.png"), target_vis)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_obstacle_mask.png"), obstacle_vis)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
