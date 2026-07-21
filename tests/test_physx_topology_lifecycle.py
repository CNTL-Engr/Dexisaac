from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_scene_module():
    """加载 Scene 而不导入/启动 Isaac Lab Application。"""
    config = types.ModuleType("config")
    config.initialize_app = lambda *args, **kwargs: None
    config.configure_simulation = lambda *args, **kwargs: None
    project_paths = types.ModuleType("project_paths")
    project_paths.project_path = lambda *parts: str(ROOT.joinpath(*parts))
    project_paths.resolve_project_path = lambda path: path
    sys.modules["config"] = config
    sys.modules["project_paths"] = project_paths

    spec = importlib.util.spec_from_file_location(
        "dexisaac_scene_under_test", ROOT / "src" / "scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCENE_MODULE = _load_scene_module()


class _FakePrim:
    def __init__(self, valid=True):
        self._valid = valid

    def IsValid(self):
        return self._valid


class _FakeStage:
    def __init__(self, events):
        self.events = events
        self.paths = set()

    def GetPrimAtPath(self, path):
        return _FakePrim(path in self.paths)

    def RemovePrim(self, path):
        self.events.append(f"remove:{path}")
        self.paths.remove(path)


class _FakeSim:
    def __init__(self, events):
        self.events = events
        self.stopped = False
        self._disable_app_control_on_stop_handle = False

    def is_stopped(self):
        return self.stopped

    def is_playing(self):
        return not self.stopped

    def stop(self):
        self.events.append("stop")
        self.stopped = True

    def reset(self):
        self.events.append("reset")
        self.stopped = False
        self._disable_app_control_on_stop_handle = False

    def step(self):
        self.events.append("sim_step")

    def get_physics_dt(self):
        return 1.0 / 60.0


class _FakeTracker:
    def __init__(self, events, rebuilt_count=1):
        self.events = events
        self.rebuilt_count = rebuilt_count

    def unregister_objects(self, objects):
        self.events.append(f"unregister:{len(list(objects))}")

    def register_objects(self, objects):
        self.events.append(f"register:{len(list(objects))}")

    def rebuild_tensor_views(self):
        self.events.append("rebuild")
        return self.rebuilt_count


class _FakeRobot:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("robot_reset")

    def write(self):
        self.events.append("robot_write")

    def update(self, dt):
        self.events.append(f"robot_update:{dt:.8f}")


def _object(path):
    return SimpleNamespace(cfg=SimpleNamespace(prim_path=path))


class SceneTopologyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stage = _FakeStage(self.events)

        omni = types.ModuleType("omni")
        omni.__path__ = []
        omni_usd = types.ModuleType("omni.usd")
        omni_usd.get_context = lambda: SimpleNamespace(get_stage=lambda: self.stage)
        omni.usd = omni_usd
        sys.modules["omni"] = omni
        sys.modules["omni.usd"] = omni_usd

    def _scene(self, tracker=True, num_envs=1):
        scene = SCENE_MODULE.Scene.__new__(SCENE_MODULE.Scene)
        scene.num_envs = num_envs
        scene.sim = _FakeSim(self.events)
        scene.robots = [_FakeRobot(self.events) for _ in range(num_envs)]
        scene.robot = None
        scene._topology_update_pending = False
        scene.contact_tracker = (
            _FakeTracker(self.events, rebuilt_count=num_envs) if tracker else None
        )
        return scene

    def test_batch_delete_stops_once_before_unregister_and_remove(self):
        scene = self._scene()
        objects = [_object("/World/Scene/Target_001"), _object("/World/Scene/Obj_0_002")]
        self.stage.paths.update(obj.cfg.prim_path for obj in objects)

        scene._delete_objects(objects)

        self.assertEqual(
            self.events,
            [
                "stop",
                "unregister:2",
                "remove:/World/Scene/Target_001",
                "remove:/World/Scene/Obj_0_002",
            ],
        )
        self.assertEqual(objects, [])
        self.assertTrue(scene._topology_update_pending)
        self.assertTrue(scene.sim._disable_app_control_on_stop_handle)

    def test_reset_rebuild_order_commits_topology_update(self):
        scene = self._scene()
        scene._topology_update_pending = True
        scene.sim.stopped = True
        new_objects = [_object("/World/Scene/Target_003")]

        scene.contact_tracker.register_objects(new_objects)
        scene._reset_simulation_and_rebuild_views("test")

        self.assertEqual(
            self.events,
            ["register:1", "reset", "robot_reset", "rebuild"],
        )
        self.assertFalse(scene._topology_update_pending)
        self.assertTrue(scene.sim.is_playing())
        self.assertFalse(scene.sim._disable_app_control_on_stop_handle)

    def test_partial_delete_only_removes_requested_environment(self):
        scene = self._scene(num_envs=2)
        env0 = _object("/World/Scene_0/Scene/Target_001")
        env1 = _object("/World/Scene_1/Scene/Target_002")
        objects = [env0, env1]
        self.stage.paths.update(obj.cfg.prim_path for obj in objects)

        scene._delete_objects(objects, env_ids_to_delete=[1])

        self.assertEqual(objects, [env0])
        self.assertIn(env0.cfg.prim_path, self.stage.paths)
        self.assertNotIn(env1.cfg.prim_path, self.stage.paths)
        self.assertEqual(self.events.count("stop"), 1)
        self.assertIn("unregister:1", self.events)

    def test_lifecycle_is_safe_without_contact_tracker(self):
        scene = self._scene(tracker=False)
        objects = [_object("/World/Scene/Target_004")]
        self.stage.paths.add(objects[0].cfg.prim_path)

        scene._delete_objects(objects)
        scene._reset_simulation_and_rebuild_views("test-no-tracker")

        self.assertEqual(
            self.events,
            ["stop", "remove:/World/Scene/Target_004", "reset", "robot_reset"],
        )
        self.assertFalse(scene._topology_update_pending)

    def test_step_is_rejected_while_topology_update_is_pending(self):
        scene = self._scene()
        scene._topology_update_pending = True

        with self.assertRaisesRegex(RuntimeError, "拓扑更新尚未完成"):
            scene.step()

    def test_step_updates_every_robot_once_with_physics_dt(self):
        scene = self._scene(num_envs=3)

        scene.step()

        self.assertEqual(
            self.events,
            [
                "robot_write",
                "robot_write",
                "robot_write",
                "sim_step",
                "robot_update:0.01666667",
                "robot_update:0.01666667",
                "robot_update:0.01666667",
            ],
        )


class MultiEnvironmentControlRegressionTests(unittest.TestCase):
    def test_global_env_indices_are_not_used_as_robot_local_ik_indices(self):
        robot_source = (ROOT / "src" / "robot.py").read_text(encoding="utf-8")
        wrapper_source = (ROOT / "train" / "env_wrapper.py").read_text(encoding="utf-8")

        self.assertNotIn("ik_fail_indices", robot_source)
        self.assertNotIn("ik_fail_indices", wrapper_source)
        self.assertNotIn("env_ids=[env_idx]", wrapper_source)
        self.assertIn("self.ik_failed = False", robot_source)
        self.assertIn("if robot.ik_failed:", wrapper_source)

    def test_push_loop_does_not_advance_three_dt_per_physics_step(self):
        wrapper_source = (ROOT / "train" / "env_wrapper.py").read_text(encoding="utf-8")

        self.assertNotIn("state['elapsed_time'] += dt * 3", wrapper_source)
        self.assertNotIn("robot.update(dt * 3)", wrapper_source)
        self.assertIn("dt = float(self.scene.sim.get_physics_dt())", wrapper_source)
        self.assertIn("state['elapsed_time'] += dt", wrapper_source)


class ContactTrackerInvalidationTests(unittest.TestCase):
    def test_unregister_releases_views_but_preserves_static_registration(self):
        sys.path.insert(0, str(ROOT / "src"))
        from physx_contact_report import PhysXContactReportTracker

        tracker = PhysXContactReportTracker("/World/Scene/R_2F_140", num_envs=1)
        target = _object("/World/Scene/Target_001")
        tracker._entities_by_env[0] = {
            target.cfg.prim_path: {"entity_key": target.cfg.prim_path},
            "/World/Scene/Table/Cube": {"entity_key": "/World/Scene/Table/Cube"},
        }
        tracker._physics_sim_view = object()
        tracker._tensor_views[0] = {"regular_view": object()}
        tracker._pending_events.append({"event": "old"})
        tracker._active_pairs[(0, "finger", target.cfg.prim_path)] = {"force": 1.0}
        tracker._monitored_fingers.add("/World/Scene/R_2F_140/left_inner_finger")
        tracker._contact_subscription = object()

        tracker.unregister_objects([target])

        self.assertIsNone(tracker._physics_sim_view)
        self.assertEqual(tracker._tensor_views, {})
        self.assertEqual(tracker._pending_events, [])
        self.assertEqual(tracker._active_pairs, {})
        self.assertNotIn(target.cfg.prim_path, tracker._entities_by_env[0])
        self.assertIn("/World/Scene/Table/Cube", tracker._entities_by_env[0])
        self.assertTrue(tracker._monitored_fingers)
        self.assertIsNotNone(tracker._contact_subscription)


if __name__ == "__main__":
    unittest.main()
