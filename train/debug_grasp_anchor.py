"""
独立调试 success_judge 算法 (无需 Isaac Sim)。

用法:
  # 真实数据 (从仿真中 dump 出的 .npy)
  python train/debug_grasp_anchor.py \\
      --target-mask path/to/mask.npy \\
      --depth path/to/depth.npy \\
      [--rgb path/to/rgb.png] \\
      [--obstacle-mask path/to/obs.npy] \\
      [--out debug_output]

  # 合成形状 (无需任何输入)
  python train/debug_grasp_anchor.py --synthetic bottle
  python train/debug_grasp_anchor.py --synthetic scissors
  python train/debug_grasp_anchor.py --synthetic box
  python train/debug_grasp_anchor.py --synthetic l_shape

输出:
  <out>/<tag>_overlay.png        RGB + 锚点 + 抓取轴 + 两块清空带
  <out>/<tag>_score.png          z·dist² 评分热图
  <out>/<tag>_summary.txt        文本结果

可在仿真中加这几行 dump 一帧真实数据:
    np.save("dump_target.npy",   target_mask)
    np.save("dump_depth.npy",    depth_img)
    np.save("dump_obstacle.npy", obstacle_mask)
    cv2.imwrite("dump_rgb.png",  cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 让脚本能从 train/ 目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from success_judge import (  # noqa: E402
    SuccessSeparationMixin,
    PIXELS_PER_METER, NOISE_THRESHOLD,
)


# ============================================================
#  Stub host: 让 mixin 的 _analyze / _build_clear 能独立调用
# ============================================================
class _Debugger(SuccessSeparationMixin):
    def __init__(self):
        self.depth_debug_dir = None     # 禁用内部 debug 保存
        self.env_steps = [0]
        self.depth_debug_episode = 0


# ============================================================
#  合成形状 (320x320 mask + depth, table_depth = 1.25)
# ============================================================
TABLE_DEPTH_M = 1.25         # 桌面深度 (与项目相机高度一致)
SHAPE_CENTER = (160, 160)


def _empty_canvas():
    mask = np.zeros((320, 320), dtype=np.uint8)
    depth = np.full((320, 320), TABLE_DEPTH_M, dtype=np.float32)
    return mask, depth


def _set_height(depth, mask, height_m):
    """把 mask 区域的深度设为 table_depth - height_m。"""
    depth[mask > 0] = TABLE_DEPTH_M - float(height_m)


def make_synthetic(kind):
    """生成合成 (target_mask, depth, obstacle_mask)。"""
    mask, depth = _empty_canvas()
    obs = np.zeros((320, 320), dtype=np.uint8)
    cu, cv = SHAPE_CENTER

    if kind == "bottle":
        # 真实尺寸: 瓶肚 直径 ~60mm = 19px, 高 ~120mm = 38px; 瓶颈直径 ~22mm = 7px
        # 瓶肚: 椭圆 19×38px, 高度 6cm
        cv2.ellipse(mask, (cu, cv + 20), (19, 38), 0, 0, 360, 255, -1)
        body = mask.copy()
        _set_height(depth, body, 0.06)
        # 瓶颈: 7×20px, 高度 3cm
        neck_mask = np.zeros_like(mask)
        cv2.ellipse(neck_mask, (cu, cv - 35), (7, 22), 0, 0, 360, 255, -1)
        mask |= neck_mask
        _set_height(depth, neck_mask, 0.03)
        # 瓶盖: 8×6px, 高度 4cm
        cap_mask = np.zeros_like(mask)
        cv2.ellipse(cap_mask, (cu, cv - 60), (8, 6), 0, 0, 360, 255, -1)
        mask |= cap_mask
        _set_height(depth, cap_mask, 0.04)
        # 障碍: 瓶肚右侧紧挨着, 测试障碍能正确触发 fail
        cv2.rectangle(obs, (cu + 22, cv + 12), (cu + 45, cv + 30), 255, -1)
        _set_height(depth, obs, 0.05)
        return mask, depth, obs

    if kind == "scissors":
        # 真实尺寸: 总长 ~150mm = 48px, 把手处宽 ~25mm
        # 刀刃 1
        rect1 = np.zeros_like(mask)
        cv2.rectangle(rect1, (cu - 40, cv - 4), (cu + 40, cv + 4), 255, -1)
        M = cv2.getRotationMatrix2D((cu, cv), 18, 1.0)
        rect1 = cv2.warpAffine(rect1, M, (320, 320))
        # 刀刃 2
        rect2 = np.zeros_like(mask)
        cv2.rectangle(rect2, (cu - 40, cv - 4), (cu + 40, cv + 4), 255, -1)
        M = cv2.getRotationMatrix2D((cu, cv), -18, 1.0)
        rect2 = cv2.warpAffine(rect2, M, (320, 320))
        mask = ((rect1 > 0) | (rect2 > 0)).astype(np.uint8) * 255
        _set_height(depth, mask, 0.02)
        # 交叉区: 抓取应该在这附近 (高度峰值)
        cross = ((rect1 > 0) & (rect2 > 0)).astype(np.uint8) * 255
        _set_height(depth, cross, 0.045)
        # 把手 (圆环) - 缩小
        h1 = np.zeros_like(mask)
        cv2.circle(h1, (cu - 50, cv + 18), 10, 255, 4)
        h2 = np.zeros_like(mask)
        cv2.circle(h2, (cu - 50, cv - 18), 10, 255, 4)
        mask |= h1 | h2
        _set_height(depth, h1 | h2, 0.025)
        # 障碍: 上下各一块
        cv2.rectangle(obs, (cu + 25, cv - 45), (cu + 50, cv - 25), 255, -1)
        cv2.rectangle(obs, (cu + 25, cv + 25), (cu + 50, cv + 45), 255, -1)
        _set_height(depth, obs, 0.04)
        return mask, depth, obs

    if kind == "box":
        # 真实尺寸: 80×40mm = 26×13px, 高 35mm
        cv2.rectangle(mask, (cu - 26, cv - 13), (cu + 26, cv + 13), 255, -1)
        _set_height(depth, mask, 0.035)
        return mask, depth, obs

    if kind == "l_shape":
        # L 形 70×70mm 整体, 厚 30mm = 10px
        cv2.rectangle(mask, (cu - 22, cv - 5), (cu + 22, cv + 5), 255, -1)
        cv2.rectangle(mask, (cu + 12, cv - 22), (cu + 22, cv + 5), 255, -1)
        _set_height(depth, mask, 0.04)
        return mask, depth, obs

    raise ValueError(f"未知合成形状: {kind}")


# ============================================================
#  IO
# ============================================================
def _load_mask(path):
    if path.endswith(".npy"):
        m = np.load(path)
    else:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8) * 255


def _load_depth(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def _load_rgb(path, shape):
    if path is None:
        return np.full((shape[0], shape[1], 3), 80, dtype=np.uint8)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((shape[0], shape[1], 3), 80, dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
#  可视化
# ============================================================
def render_overlay(rgb, target_mask, obstacle_mask, analysis, regions,
                    pos_count, neg_count, success, segment_tag="whole"):
    overlay = rgb.copy()
    region_colors = {"body": (0, 255, 80),
                     "clear_pos": (255, 210, 0),
                     "clear_neg": (0, 220, 255)}
    region_alphas = {"body": 0.25, "clear_pos": 0.45, "clear_neg": 0.45}
    for name in ("body", "clear_pos", "clear_neg"):
        mask = regions[name]
        if not np.any(mask):
            continue
        color = np.array(region_colors[name], dtype=np.float32)
        a = region_alphas[name]
        overlay[mask] = (overlay[mask].astype(np.float32) * (1 - a)
                        + color * a).astype(np.uint8)

    if np.any(obstacle_mask > 0):
        overlay[obstacle_mask > 0] = (
            overlay[obstacle_mask > 0].astype(np.float32) * 0.35
            + np.array([255, 0, 0], dtype=np.float32) * 0.65
        ).astype(np.uint8)

    target_vis = (target_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(target_vis, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    obs_vis = (obstacle_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(obs_vis, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

    for name in ("clear_pos", "clear_neg"):
        band_vis = regions[name].astype(np.uint8) * 255
        c, _ = cv2.findContours(band_vis, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, c, -1, (255, 255, 255), 2)

    au, av = analysis["anchor_uv"]
    gx, gy = analysis["grasp_axis"]
    bx, by = analysis["body_axis"]
    jaw_half = float(analysis["jaw_half_w_px"])

    # body 主干段 (白线): trunk_lo / trunk_hi 范围 (anchor 局部坐标)
    trunk_lo = float(analysis.get("trunk_lo_px", 0.0))
    trunk_hi = float(analysis.get("trunk_hi_px", 0.0))
    if trunk_hi > trunk_lo:
        p1 = (int(round(au + bx * trunk_lo)), int(round(av + by * trunk_lo)))
        p2 = (int(round(au + bx * trunk_hi)), int(round(av + by * trunk_hi)))
        cv2.line(overlay, p1, p2, (255, 255, 255), 2)

    g1 = (int(round(au - gx * jaw_half)), int(round(av - gy * jaw_half)))
    g2 = (int(round(au + gx * jaw_half)), int(round(av + gy * jaw_half)))
    cv2.line(overlay, g1, g2, (255, 0, 180), 2)
    cv2.circle(overlay, g1, 3, (255, 0, 180), -1)
    cv2.circle(overlay, g2, 3, (255, 0, 180), -1)

    cv2.circle(overlay, (int(au), int(av)), 4, (255, 255, 255), -1)
    cv2.circle(overlay, (int(au), int(av)), 6, (0, 0, 0), 1)

    status = "SUCCESS" if success else "FAIL"
    trunk_len_mm = int(round((trunk_hi - trunk_lo) / PIXELS_PER_METER * 1000))
    lines = [
        f"{status} seg={segment_tag} pos={pos_count} neg={neg_count} thr={NOISE_THRESHOLD}",
        f"jaw={int(round(analysis['jaw_full_width_m']*1000))}mm "
        f"trunk={trunk_len_mm}mm "
        f"h={analysis['z_at_anchor_m']*1000:.0f}mm "
        f"out_of_jaw={int(analysis['out_of_jaw'])}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(overlay, t, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def render_score(analysis):
    z = analysis["z_map"]
    d = analysis["dist_map"]
    score = (np.maximum(z, 0) ** 1.0) * (np.maximum(d, 0) ** 2.0)
    if float(np.max(score)) < 1e-9:
        score = d.copy()
    s = score / max(float(np.max(score)), 1e-9)
    s_u8 = (s * 255).astype(np.uint8)
    return cv2.applyColorMap(s_u8, cv2.COLORMAP_JET)


# ============================================================
#  主流程
# ============================================================
def run(target_mask, depth, obstacle_mask, rgb, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    dbg = _Debugger()

    analysis = dbg._analyze_grasp_geometry(target_mask, depth)
    if analysis is None:
        msg = "[ERROR] 几何分析失败 (mask 太小 / 桌面估计失败 / PCA 退化)"
        print(msg)
        with open(os.path.join(out_dir, f"{tag}_summary.txt"), "w") as f:
            f.write(msg + "\n")
        return

    regions = dbg._build_clear_regions(target_mask.shape, analysis)
    # 与 _check_successful_separation 一致: 两侧清空即成功, out_of_jaw 不影响判定
    eval_result = dbg._evaluate_clear_regions(obstacle_mask, regions)
    pos_count = eval_result["pos_count"]
    neg_count = eval_result["neg_count"]
    success = eval_result["is_clear"]
    segment_tag = eval_result["segment_tag"]
    band_length_m = eval_result["band_length_m"]

    overlay = render_overlay(rgb, target_mask, obstacle_mask, analysis,
                              regions, pos_count, neg_count, success,
                              segment_tag)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"{tag}_score.png"),
                render_score(analysis))

    summary = [
        f"标签: {tag}",
        f"成功: {success}  (segment={segment_tag}, pos={pos_count}, neg={neg_count}, threshold={NOISE_THRESHOLD})",
        f"锚点: ({int(analysis['anchor_uv'][0])}, {int(analysis['anchor_uv'][1])})",
        f"夹宽: {analysis['jaw_full_width_m']*1000:.1f} mm  (上限 {0.14*1000:.0f} mm)",
        f"主干: {(analysis['trunk_hi_px']-analysis['trunk_lo_px'])/PIXELS_PER_METER*1000:.0f} mm",
        f"清空带判定长度: {band_length_m*1000:.0f} mm",
        f"锚点高度: {analysis['z_at_anchor_m']*1000:.1f} mm",
        f"超出夹爪: {analysis['out_of_jaw']}",
        f"桌面深度: {analysis['table_depth']:.4f} m",
        f"grasp_axis: ({analysis['grasp_axis'][0]:.3f}, {analysis['grasp_axis'][1]:.3f})",
        f"body_axis:  ({analysis['body_axis'][0]:.3f}, {analysis['body_axis'][1]:.3f})",
        f"PIXELS_PER_METER = {PIXELS_PER_METER:.1f}",
    ]
    text = "\n".join(summary)
    print(text)
    with open(os.path.join(out_dir, f"{tag}_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"[输出] {out_dir}/{tag}_overlay.png")
    print(f"[输出] {out_dir}/{tag}_score.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-mask", type=str, default=None)
    p.add_argument("--depth", type=str, default=None)
    p.add_argument("--obstacle-mask", type=str, default=None)
    p.add_argument("--rgb", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None,
                   choices=["bottle", "scissors", "box", "l_shape"])
    p.add_argument("--out", type=str, default="debug_grasp_output")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    if args.synthetic:
        target_mask, depth, obstacle_mask = make_synthetic(args.synthetic)
        rgb = np.full((320, 320, 3), 80, dtype=np.uint8)
        rgb[target_mask > 0] = (180, 180, 180)
        rgb[obstacle_mask > 0] = (60, 60, 60)
        tag = args.tag or f"synthetic_{args.synthetic}"
    else:
        if not args.target_mask or not args.depth:
            p.error("需要 --target-mask 和 --depth, 或使用 --synthetic")
        target_mask = _load_mask(args.target_mask)
        depth = _load_depth(args.depth)
        if args.obstacle_mask:
            obstacle_mask = _load_mask(args.obstacle_mask)
        else:
            obstacle_mask = np.zeros_like(target_mask)
        rgb = _load_rgb(args.rgb, target_mask.shape)
        tag = args.tag or os.path.splitext(os.path.basename(args.target_mask))[0]

    run(target_mask, depth, obstacle_mask, rgb, args.out, tag)


if __name__ == "__main__":
    main()
