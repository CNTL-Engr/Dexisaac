"""Pure image-space regression tests for the two-tier push geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

try:
    import cv2  # noqa: F401
    import action_primitive as ap
except Exception:  # pragma: no cover
    ap = None


@unittest.skipIf(ap is None, "当前 Python 环境缺少推点运行依赖")
class PushPointGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((320, 320), dtype=np.uint8)
        self.mask[100:161, 100:181] = 255
        self.depth = np.full((320, 320), 1.23, dtype=np.float32)
        self.depth[self.mask > 0] = 1.15  # target top z ~= 0.10 m

    def _seg_scene(self):
        seg = np.zeros((320, 320), dtype=np.int32)
        seg[self.mask > 0] = 1
        return seg

    def test_directional_band_and_window(self):
        info = {}
        result = ap._compute_migrated_edge_push(
            self.depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNotNone(result)
        self.assertEqual(info["g00_mask"].sum(), self.mask.astype(bool).sum())
        self.assertGreater(info["g01_mask"].sum(), info["g00_mask"].sum())
        self.assertTrue(np.all(info["candidate_mask"] <= info["g02_mask"]))
        self.assertTrue(np.all(info["selected_window_mask"] <= info["height_feasible_mask"]))
        self.assertFalse(np.any(info["selected_window_mask"] & info["obs_mask"]))
        self.assertGreater(info["push_z"], ap.TABLE_Z)

    def test_contact_strip_uses_configured_percentile(self):
        height = np.zeros((320, 320), dtype=np.float32)
        lateral = ap._pixel_radius(ap.GRIPPER_LATERAL_M / 2.0)
        sample_rows = range(130 - lateral, 130 + lateral + 1)
        values = np.linspace(
            0.03, 0.11, 2 * len(sample_rows), dtype=np.float32
        )
        index = 0
        for u in (102, 103):
            for v in sample_rows:
                height[v, u] = values[index]
                index += 1
        actual = ap._candidate_contact_strip(height, self.mask, 100, 130, 0)
        self.assertAlmostEqual(
            actual,
            float(np.percentile(values, ap.TARGET_HEIGHT_PERCENTILE)),
            places=6,
        )

    def test_two_cm_tier_has_priority_over_wider_second_tier_obstacle(self):
        seg = self._seg_scene()
        # Action direction 1 is image +u. A is 1 pixel from the target edge;
        # B is farther than 2 cm and wider.
        seg[130:151, 182:188] = 2
        seg[130:151, 190:202] = 3
        global_mask = (seg > 0).astype(np.uint8) * 255
        _, selected, debug = ap._find_dominant_obstacle_mask(
            seg, self.mask, global_mask, 140, 130, 1, 1, 0
        )
        self.assertEqual(selected, 2)
        self.assertEqual(debug["selected_tier"], 1)

    def test_second_tier_selects_widest_obstacle(self):
        seg = self._seg_scene()
        # Both obstacles are outside the first 2 cm tier; B is wider.
        seg[130:136, 189:195] = 2
        seg[136:151, 189:202] = 3
        global_mask = (seg > 0).astype(np.uint8) * 255
        _, selected, debug = ap._find_dominant_obstacle_mask(
            seg, self.mask, global_mask, 140, 130, 1, 1, 0
        )
        self.assertEqual(selected, 3)
        self.assertEqual(debug["selected_tier"], 2)

    def test_no_obstacle_within_five_cm_is_empty(self):
        seg = self._seg_scene()
        global_mask = (seg > 0).astype(np.uint8) * 255
        mask, selected, debug = ap._find_dominant_obstacle_mask(
            seg, self.mask, global_mask, 140, 130, 1, 1, 0
        )
        self.assertIsNone(mask)
        self.assertEqual(selected, -1)
        self.assertEqual(debug["outcome"], "empty")
        self.assertEqual(debug["empty_reason"], "no_obstacle_within_5cm")

    def test_low_pixels_remain_height_feasible_without_instance_mask(self):
        depth = self.depth.copy()
        depth[130:151, 91:97] = 1.20  # low obstacle top z ~= 0.05
        info = {}
        result = ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNotNone(result)
        low_patch = np.zeros_like(self.mask, dtype=bool)
        low_patch[130:151, 91:97] = True
        self.assertFalse(np.any(info["obs_mask"] & low_patch))

    def test_high_pixels_are_obs_without_instance_mask(self):
        depth = self.depth.copy()
        depth[130:151, 91:97] = 1.14  # obstacle top z ~= 0.11
        info = {}
        result = ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNotNone(result)
        high_patch = np.zeros_like(self.mask, dtype=bool)
        high_patch[130:151, 91:97] = True
        self.assertTrue(np.any(info["obs_mask"] & high_patch))
        self.assertFalse(np.any(info["selected_window_mask"] & info["obs_mask"]))

    def test_invalid_depth_in_g02_is_obs(self):
        depth = self.depth.copy()
        depth[130, 95] = np.nan
        info = {}
        ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertTrue(info["g02_mask"][130, 95])
        self.assertTrue(info["raw_obs_mask"][130, 95])
        self.assertTrue(info["obs_mask"][130, 95])

    def test_raw_obs_dilates_one_pixel_including_diagonals(self):
        depth = self.depth.copy()
        depth[130, 95] = np.nan
        info = {}
        ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        expected = np.zeros_like(info["g02_mask"], dtype=bool)
        expected[129:132, 94:97] = True
        expected &= info["g02_mask"]
        self.assertTrue(np.all(info["obs_mask"][expected]))
        self.assertTrue(info["obs_mask"][129, 94])

    def test_dilated_obs_is_clipped_to_g02(self):
        _, edge_points = ap._extract_rear_edge_for_direction(self.mask, 0)
        _, g02 = ap._build_directional_band(self.mask, edge_points, 0)
        row_xs = np.where(g02[130])[0]
        edge_x = int(row_xs.min())
        depth = self.depth.copy()
        depth[130, edge_x] = np.nan
        info = {}
        ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertTrue(info["raw_obs_mask"][130, edge_x])
        self.assertFalse(np.any(info["obs_mask"] & ~info["g02_mask"]))
        self.assertTrue(np.array_equal(
            info["height_feasible_mask"],
            info["g02_mask"] & ~info["obs_mask"],
        ))

    def test_every_height_feasible_pixel_satisfies_minimum_overlap_limit(self):
        depth = self.depth.copy()
        depth[130:151, 91:97] = 1.14
        info = {}
        ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        height, _ = ap._height_map(depth)
        feasible = info["height_feasible_mask"]
        pixel_z_min = np.maximum(height, ap.TABLE_Z) + ap.VERTICAL_CLEARANCE_M
        pixel_z_max = info["target_height_map_p90"] - ap.MIN_PUSH_OVERLAP_M
        self.assertTrue(np.all(pixel_z_min[feasible] <= pixel_z_max[feasible]))

    def test_window_kernel_matches_physical_footprint(self):
        lateral_radius = ap._pixel_radius(ap.GRIPPER_LATERAL_M / 2.0)
        push_radius = ap._pixel_radius(ap.GRIPPER_PUSH_AXIS_M / 2.0)
        self.assertEqual(
            ap._window_kernel(0).shape,
            (2 * lateral_radius + 1, 2 * push_radius + 1),
        )
        self.assertEqual(
            ap._window_kernel(90).shape,
            (2 * push_radius + 1, 2 * lateral_radius + 1),
        )

    def test_channel_max_height_sets_z_with_configured_clearance(self):
        depth = self.depth.copy()
        edge, points = ap._extract_rear_edge_for_direction(self.mask, 0)
        _, g02 = ap._build_directional_band(self.mask, points, 0)
        depth[g02] = 1.185  # environment height ~= 0.065 m
        info = {}
        result = ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(info["channel_max_height"], 0.065, places=3)
        self.assertAlmostEqual(
            info["push_z"],
            info["channel_max_height"] + ap.CHANNEL_Z_CLEARANCE_M,
            places=6,
        )
        self.assertFalse(np.any(info["channel_mask"] & info["g00_mask"]))

    def test_final_push_retreats_five_mm_along_negative_direction(self):
        retreat_px = 0.005 * ap.PPM
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                info = {}
                result = ap._compute_migrated_edge_push(
                    self.depth, self.mask, 140, 130, angle,
                    global_mask=self.mask, debug_info=info,
                )
                self.assertIsNotNone(result)
                center_u, center_v = info["selected_window_center_pixel"]
                du, dv = ap._cardinal_push_step(angle)
                self.assertAlmostEqual(result[0], center_u - du * retreat_px)
                self.assertAlmostEqual(result[1], center_v - dv * retreat_px)
                self.assertEqual(info["final_push_retreat_m"], 0.005)

    def test_minimum_overlap_rejects_shallow_target(self):
        shallow_depth = np.full_like(self.depth, 1.23)
        target_height = (
            ap.TABLE_Z + ap.VERTICAL_CLEARANCE_M
            + ap.MIN_PUSH_OVERLAP_M - 0.005
        )
        shallow_depth[self.mask > 0] = ap.CAMERA_Z - target_height
        info = {}
        result = ap._compute_migrated_edge_push(
            shallow_depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNone(result)
        self.assertEqual(info["geometry_invalid_reason"], "no_height_feasible_pixels")

    def test_narrow_free_band_reports_no_window_fit_after_height_filter(self):
        depth = self.depth.copy()
        # Leave only a three-row lateral corridor below the height threshold.
        depth[100:161, 87:100] = 1.14
        depth[129:132, 87:100] = 1.23
        info = {}
        result = ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNone(result)
        self.assertEqual(
            info["geometry_invalid_reason"],
            "no_window_fit_after_height_filter",
        )

    def test_robot_z_limit_reports_no_feasible_z(self):
        depth = np.full_like(self.depth, 1.23)
        depth[self.mask > 0] = 0.75  # target top z ~= 0.50 m
        _, points = ap._extract_rear_edge_for_direction(self.mask, 0)
        _, g02 = ap._build_directional_band(self.mask, points, 0)
        depth[g02] = 0.85  # channel top z ~= 0.40 m -> push z ~= 0.405 m
        info = {}
        result = ap._compute_migrated_edge_push(
            depth, self.mask, 140, 130, 0,
            global_mask=self.mask, debug_info=info,
        )
        self.assertIsNone(result)
        self.assertEqual(info["geometry_invalid_reason"], "no_feasible_z")


if __name__ == "__main__":
    unittest.main()
