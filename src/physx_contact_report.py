"""PhysX 接触追踪器（兼容 Isaac Lab 2.1.1 / Isaac Sim 4.5）。

只监控 Robotiq 左右 inner/outer 四根手指。每个物理步优先保留原始 Contact Report；
在本项目 CUDA/Fabric 管线不产生 CPU report 时，使用 PhysX GPU
``RigidContactView`` 读取真实的手指-实体成对接触力。
不创建 Isaac Lab ContactSensor 对象，也不在运行期修改任何 instanceable 几何。

返回结果按环境分流，并同时给出：
  * active: 当前仍在接触的实体；
  * new:    本物理步第一次形成有效接触的实体。

上层状态机据此实现：下降段持续检查、推动段仅检查第一次接触。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path


MONITORED_FINGER_LEAVES = (
    "left_inner_finger",
    "right_inner_finger",
    "left_outer_finger",
    "right_outer_finger",
)
MODEL_ID_RE = re.compile(r"^\d{3}$")


class PhysXContactReportTracker:
    """读取 PhysX 当前物理步的成对接触数据，并映射到场景实体。"""

    def __init__(
        self,
        gripper_prim_path: str,
        num_envs: int = 1,
    ):
        self.gripper_prim_path = gripper_prim_path
        self.num_envs = int(num_envs)

        # 默认 0 N：PhysX 只要报告真实接触就算“碰到”，避免漏掉轻触。
        # impulse / dt 仍作为诊断力输出；如需抑制数值噪声可用环境变量提高阈值。
        self.force_threshold = float(os.environ.get("PHYSX_CONTACT_FORCE_THRESHOLD", "0.0"))

        self._interface = None
        self._contact_subscription = None
        self._pending_events = []
        self._physics_sim_view = None
        self._tensor_views = {}
        self._monitored_fingers = set()
        self._entities_by_env = {env_id: {} for env_id in range(self.num_envs)}
        self._active_pairs = {}
        self._step_index = 0
        self._warned_poll_error = False
        self._debug = {
            "headers": 0,
            "finger_pairs": 0,
            "accepted_pairs": 0,
            "below_threshold": 0,
            "callback_headers": 0,
            "immediate_headers": 0,
            "tensor_pairs": 0,
        }
        self._debug_samples = []

    @staticmethod
    def _env_id_from_path(prim_path: str) -> int:
        match = re.search(r"/World/Scene_(\d+)(?:/|$)", prim_path)
        if match:
            return int(match.group(1))
        return 0

    @staticmethod
    def _model_id_from_object(obj) -> str | None:
        """从实际模型 USD 所在文件夹取得三位 ID，不依赖 prim leaf 的命名。"""
        try:
            usd_path = obj.cfg.spawn.usd_path
            model_id = Path(str(usd_path)).parent.name
            if MODEL_ID_RE.fullmatch(model_id):
                return model_id
        except Exception:
            pass

        # 仅作兼容兜底：Target_071 / Obj_0_005。
        try:
            leaf = obj.cfg.prim_path.rsplit("/", 1)[-1]
            if leaf.startswith("Target_"):
                candidate = leaf[len("Target_") :]
            elif leaf.startswith("Obj_"):
                parts = leaf.split("_", 2)
                candidate = parts[2] if len(parts) == 3 else ""
            else:
                candidate = ""
            return candidate if MODEL_ID_RE.fullmatch(candidate) else None
        except Exception:
            return None

    @staticmethod
    def _rigid_path_from_root(root_path: str) -> str:
        """解析引用资产内真正的唯一刚体路径（071 等模型的刚体位于子级）。"""
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"物体 prim 无效，无法解析刚体: {root_path}")
        # 找到刚体后不再进入其子树：USD 不允许嵌套刚体，而且继续遍历
        # instanceable 几何会展开大量 prototype，既无意义又会严重拖慢初始化。
        candidates = []
        queue = [root]
        while queue:
            prim = queue.pop(0)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                candidates.append(prim.GetPath().pathString)
                continue
            queue.extend(prim.GetChildren())
        if len(candidates) != 1:
            raise RuntimeError(
                f"{root_path} 应包含且只包含一个刚体，实际解析到 {candidates}"
            )
        return candidates[0]

    def activate_api(self) -> int:
        """验证资产中预烘焙的 API 并连接接口；运行时绝不修改 finger prim。"""
        import carb
        import isaaclab.sim as sim_utils
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, UsdPhysics

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        stage = omni.usd.get_context().get_stage()
        roots = sim_utils.find_matching_prims(self.gripper_prim_path)
        for root in roots:
            root_path = root.GetPath().pathString
            for leaf in MONITORED_FINGER_LEAVES:
                finger_path = f"{root_path}/{leaf}"
                finger = stage.GetPrimAtPath(finger_path)
                if not finger.IsValid() or not finger.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(f"[PhysXContactReport] ⚠ 跳过无效/非刚体手指: {finger_path}")
                    continue
                if not finger.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    raise RuntimeError(
                        f"{finger_path} 缺少预烘焙 PhysxContactReportAPI。"
                        "禁止运行时 Apply API；请使用 assets/env_contact_report.usda 加载场景。"
                    )
                api = PhysxSchema.PhysxContactReportAPI.Get(stage, finger.GetPath())
                threshold = api.GetThresholdAttr().Get()
                if threshold is None or float(threshold) > 0.0:
                    raise RuntimeError(
                        f"{finger_path} 的预烘焙接触阈值不是 0（当前={threshold}）。"
                        "禁止运行时 Set；请修正静态 USDA 覆盖层。"
                    )
                self._monitored_fingers.add(finger_path)

        expected_fingers = self.num_envs * len(MONITORED_FINGER_LEAVES)
        if len(self._monitored_fingers) != expected_fingers:
            raise RuntimeError(
                "四指接触检测初始化不完整："
                f"期望 {expected_fingers} 个 finger 刚体，实际 {len(self._monitored_fingers)} 个。"
            )

        self._interface = get_physx_simulation_interface()
        # SimulationContext.step() 内部还会执行框架更新；等它返回后，4.5 的
        # one-step contact buffer 可能已经清空。必须在 PhysX 回调中复制数据。
        self._contact_subscription = self._interface.subscribe_contact_report_events(
            self._on_contact_report
        )
        print(
            f"[PhysXContactReport] 已验证 {len(self._monitored_fingers)} 根手指的静态 ContactAPI，"
            f"有效接触力阈值={self.force_threshold:.3f}N"
        )
        return len(self._monitored_fingers)

    def register_objects(self, spawned_objects) -> None:
        """注册/替换传入物体所属环境的实体路径映射。"""
        objects = list(spawned_objects or [])
        affected_envs = {
            self._env_id_from_path(obj.cfg.prim_path)
            for obj in objects
            if getattr(getattr(obj, "cfg", None), "prim_path", None)
        }
        for env_id in affected_envs:
            self._entities_by_env.setdefault(env_id, {}).clear()
            self._remove_active_for_env(env_id)

        for obj in objects:
            root_path = obj.cfg.prim_path.rstrip("/")
            env_id = self._env_id_from_path(root_path)
            model_id = self._model_id_from_object(obj)
            rigid_path = self._rigid_path_from_root(root_path)
            leaf = root_path.rsplit("/", 1)[-1]
            self._entities_by_env.setdefault(env_id, {})[root_path] = {
                "entity_key": root_path,
                "root_path": root_path,
                "physics_path": rigid_path,
                "model_id": model_id,
                "kind": "object",
                "role": "target" if leaf.startswith("Target_") else "obstacle",
            }

        # 桌面不是三位模型，使用明确的 TABLE 实体，不伪造数字 ID。
        for env_id in affected_envs or range(self.num_envs):
            if self.num_envs > 1:
                table_path = f"/World/Scene_{env_id}/Scene/Table/Cube"
            else:
                table_path = "/World/Scene/Table/Cube"
            self._entities_by_env.setdefault(env_id, {})[table_path] = {
                "entity_key": table_path,
                "root_path": table_path,
                "physics_path": table_path,
                "model_id": None,
                "kind": "table",
                "role": "illegal",
            }

    def rebuild_tensor_views(self) -> int:
        """sim.reset 后按环境重建精确的 PhysX GPU 成对接触视图。"""
        from isaacsim.core.simulation_manager import SimulationManager

        # reset/PLAY 会创建一个新的全局 SimulationView。旧 view 即使已经失效，
        # 也不能继续留在 tracker 中被 poll_contacts() 访问。
        self.invalidate_tensor_views()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        if self._physics_sim_view is None:
            raise RuntimeError("PhysX SimulationView 尚未初始化，无法建立接触视图。")

        for env_id in range(self.num_envs):
            fingers = sorted(
                path for path in self._monitored_fingers
                if self._env_id_from_path(path) == env_id
            )
            entities = self._entities_by_env.get(env_id, {})
            regular_entities = [
                entity for entity in entities.values()
                if entity["kind"] == "table" or entity["physics_path"] == entity["root_path"]
            ]
            nested_entities = [
                entity for entity in entities.values()
                if entity["kind"] == "object" and entity["physics_path"] != entity["root_path"]
            ]
            expected_fingers = len(MONITORED_FINGER_LEAVES)
            if len(fingers) != expected_fingers or not regular_entities:
                continue

            regular_paths = [entity["physics_path"] for entity in regular_entities]
            print(
                f"[PhysXContactReport] Env {env_id} 创建常规视图: "
                f"{len(fingers)}指 × {len(regular_paths)} filters"
            )
            regular_view = self._physics_sim_view.create_rigid_contact_view(
                fingers,
                filter_patterns=[list(regular_paths) for _ in fingers],
            )
            if (
                regular_view.sensor_count != len(fingers)
                or regular_view.filter_count != len(regular_paths)
            ):
                raise RuntimeError(
                    f"Env {env_id} 常规接触视图维度错误: "
                    f"sensor={regular_view.sensor_count}/{len(fingers)}, "
                    f"filter={regular_view.filter_count}/{len(regular_paths)}"
                )

            state = {
                "regular_view": regular_view,
                # PhysX 4.5 的 sensor_paths/filter_paths 属性在 one-to-many
                # list-of-lists 视图上可能发生极慢枚举；API 文档保证输入顺序
                # 即矩阵顺序，因此直接保存已验证的 exact-path 输入映射。
                "fingers": fingers,
                "regular_entity_keys": [entity["entity_key"] for entity in regular_entities],
                "nested_view": None,
                "nested_entity_keys": [],
            }

            if nested_entities:
                nested_paths = [entity["physics_path"] for entity in nested_entities]
                print(
                    f"[PhysXContactReport] Env {env_id} 创建嵌套刚体视图: "
                    f"{len(nested_paths)} objects × {len(fingers)}指"
                )
                nested_view = self._physics_sim_view.create_rigid_contact_view(
                    nested_paths,
                    filter_patterns=[list(fingers) for _ in nested_paths],
                )
                if (
                    nested_view.sensor_count != len(nested_paths)
                    or nested_view.filter_count != len(fingers)
                ):
                    raise RuntimeError(
                        f"Env {env_id} 嵌套刚体接触视图维度错误: "
                        f"sensor={nested_view.sensor_count}/{len(nested_paths)}, "
                        f"filter={nested_view.filter_count}/{len(fingers)}"
                    )
                state["nested_view"] = nested_view
                state["nested_entity_keys"] = [
                    entity["entity_key"] for entity in nested_entities
                ]

            self._tensor_views[env_id] = state

        print(
            f"[PhysXContactReport] 已重建 {len(self._tensor_views)} 个 GPU 接触视图 "
            f"(普通: 4指×物体/桌面；嵌套刚体: 物体×4指)"
        )
        return len(self._tensor_views)

    def invalidate_tensor_views(self) -> None:
        """释放依赖当前 PhysX SimulationView 的全部瞬态状态。

        该方法只释放会随 STOP/PLAY 失效的 native tensor views 和接触边沿状态；
        静态手指路径、Contact Report subscription 及实体注册表会保留，以便
        ``sim.reset()`` 后调用 :meth:`rebuild_tensor_views` 原地恢复。
        """
        self._tensor_views.clear()
        self._physics_sim_view = None
        self._pending_events.clear()
        self._active_pairs.clear()

    def unregister_objects(self, objects) -> None:
        """物体从 stage 删除前同步清理注册表，避免路径复用时残留状态。"""
        for obj in list(objects or []):
            try:
                root_path = obj.cfg.prim_path.rstrip("/")
            except Exception:
                continue
            env_id = self._env_id_from_path(root_path)
            self._entities_by_env.setdefault(env_id, {}).pop(root_path, None)
        # 任一刚体拓扑变化都会使全局 SimulationView 失效，而不只是被删除
        # 环境对应的 contact view；下一次 sim.reset 后必须统一重建全部环境。
        self.invalidate_tensor_views()

    def begin_action(self, env_ids) -> None:
        """开始一次推动前清除旧动作的接触边沿状态。"""
        self._pending_events.clear()
        for env_id in env_ids:
            self._remove_active_for_env(int(env_id))

    def _remove_active_for_env(self, env_id: int) -> None:
        for key in list(self._active_pairs):
            if key[0] == env_id:
                self._active_pairs.pop(key, None)

    def _resolve_finger(self, actor_path: str, collider_path: str) -> str | None:
        for finger_path in self._monitored_fingers:
            if actor_path == finger_path or actor_path.startswith(finger_path + "/"):
                return finger_path
            if collider_path == finger_path or collider_path.startswith(finger_path + "/"):
                return finger_path
        return None

    def _resolve_entity(self, env_id: int, actor_path: str, collider_path: str):
        entities = self._entities_by_env.get(env_id, {})
        # 最长前缀优先，兼容 /Target_071/Root/textured 等嵌套刚体/碰撞体。
        for root_path in sorted(entities, key=len, reverse=True):
            if (
                actor_path == root_path
                or actor_path.startswith(root_path + "/")
                or collider_path == root_path
                or collider_path.startswith(root_path + "/")
            ):
                return entities[root_path]
        return None

    @staticmethod
    def _decode_path(encoded_path) -> str:
        from pxr import PhysicsSchemaTools

        return str(PhysicsSchemaTools.intToSdfPath(encoded_path))

    @staticmethod
    def _contact_impulse(header, contact_data):
        impulse_sum = 0.0
        min_separation = float("inf")
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        for index in range(start, end):
            point = contact_data[index]
            impulse = point.impulse
            impulse_sum += math.sqrt(sum(float(impulse[axis]) ** 2 for axis in range(3)))
            min_separation = min(min_separation, float(point.separation))
        return impulse_sum, min_separation

    def _snapshot_report(self, contact_headers, contact_data):
        """在 PhysX 缓冲仍有效时复制出纯 Python 数据。"""
        snapshots = []
        for header in contact_headers:
            impulse, separation = self._contact_impulse(header, contact_data)
            snapshots.append({
                "actor0": self._decode_path(header.actor0),
                "actor1": self._decode_path(header.actor1),
                "collider0": self._decode_path(header.collider0),
                "collider1": self._decode_path(header.collider1),
                "event_type": header.type.name,
                "impulse": impulse,
                "separation": separation,
            })
        return snapshots

    def _on_contact_report(self, contact_headers, contact_data) -> None:
        """PhysX step 回调：只复制数据，不在回调里执行训练状态机。"""
        try:
            snapshots = self._snapshot_report(contact_headers, contact_data)
            self._debug["callback_headers"] += len(snapshots)
            self._pending_events.extend(snapshots)
        except Exception as exc:
            if not self._warned_poll_error:
                print(f"[PhysXContactReport] ⚠ 接触回调复制失败: {exc}")
                self._warned_poll_error = True

    def poll_contacts(self, dt: float):
        """物理步之后立即调用；返回 env_id -> {active, new, ended}。"""
        result = {
            env_id: {"active": [], "new": [], "ended": [], "step": self._step_index}
            for env_id in range(self.num_envs)
        }
        self._step_index += 1
        if self._interface is None or not self._monitored_fingers:
            return result

        snapshots = list(self._pending_events)
        self._pending_events.clear()

        try:
            # 兼容低层 step 后缓冲仍可见的运行模式；与回调副本按完整事件去重。
            contact_headers, contact_data = self._interface.get_contact_report()
            immediate = self._snapshot_report(contact_headers, contact_data)
            self._debug["immediate_headers"] += len(immediate)
            snapshots.extend(immediate)
        except Exception as exc:
            if not self._warned_poll_error:
                print(f"[PhysXContactReport] ⚠ get_contact_report 失败: {exc}")
                self._warned_poll_error = True

        unique_snapshots = []
        seen = set()
        for event in snapshots:
            key = (
                event["actor0"], event["actor1"], event["collider0"], event["collider1"],
                event["event_type"], event["impulse"], event["separation"],
            )
            if key not in seen:
                unique_snapshots.append(event)
                seen.add(key)

        new_records = {env_id: [] for env_id in range(self.num_envs)}
        ended_records = {env_id: [] for env_id in range(self.num_envs)}
        self._debug["headers"] += len(unique_snapshots)

        for event in unique_snapshots:
            actor0 = event["actor0"]
            actor1 = event["actor1"]
            collider0 = event["collider0"]
            collider1 = event["collider1"]

            if len(self._debug_samples) < 12:
                sample = (actor0, actor1, collider0, collider1, event["event_type"])
                if sample not in self._debug_samples:
                    self._debug_samples.append(sample)

            finger_path = self._resolve_finger(actor0, collider0)
            if finger_path is not None:
                other_actor, other_collider = actor1, collider1
            else:
                finger_path = self._resolve_finger(actor1, collider1)
                if finger_path is None:
                    continue
                other_actor, other_collider = actor0, collider0

            env_id = self._env_id_from_path(finger_path)
            entity = self._resolve_entity(env_id, other_actor, other_collider)
            if entity is None:
                # 手指自碰或未注册实体不属于本任务的非法物体。
                continue

            self._debug["finger_pairs"] += 1
            pair_key = (env_id, finger_path, entity["entity_key"])
            if event["event_type"] == "CONTACT_LOST":
                old = self._active_pairs.pop(pair_key, None)
                if old is not None:
                    ended_records.setdefault(env_id, []).append(old)
                continue

            impulse = event["impulse"]
            separation = event["separation"]
            force = impulse / max(float(dt), 1.0e-8)
            if force < self.force_threshold:
                self._debug["below_threshold"] += 1
                # 低于判定阈值等同于本任务中的有效接触结束，不能把上一步的
                # 高力记录继续留在 active 中造成下降阶段误报。
                old = self._active_pairs.pop(pair_key, None)
                if old is not None:
                    ended_records.setdefault(env_id, []).append(old)
                continue

            record = {
                **entity,
                "env_id": env_id,
                "finger_path": finger_path,
                "actor0": actor0,
                "actor1": actor1,
                "collider0": collider0,
                "collider1": collider1,
                "other_actor": other_actor,
                "other_collider": other_collider,
                "force": force,
                "impulse": impulse,
                "separation": separation,
                "event_type": event["event_type"],
                "step": self._step_index - 1,
            }
            if pair_key not in self._active_pairs:
                new_records.setdefault(env_id, []).append(record)
            self._active_pairs[pair_key] = record
            self._debug["accepted_pairs"] += 1

        for env_id in range(self.num_envs):
            tensor_state = self._tensor_views.get(env_id)
            if tensor_state is not None:
                current_tensor_keys = set()
                # GPU force matrix 的无接触项严格为 0；只排除非正值，确保默认
                # 0 N 配置不会漏掉任意幅值的真实轻触。
                tensor_threshold = self.force_threshold
                entities = self._entities_by_env.get(env_id, {})

                def accept_pair(finger_path, entity_key, force):
                    if (
                        not math.isfinite(force)
                        or force <= 0.0
                        or force < tensor_threshold
                    ):
                        return
                    entity = entities.get(entity_key)
                    if entity is None:
                        return
                    pair_key = (env_id, finger_path, entity_key)
                    current_tensor_keys.add(pair_key)
                    record = {
                        **entity,
                        "env_id": env_id,
                        "finger_path": finger_path,
                        "actor0": finger_path,
                        "actor1": entity["root_path"],
                        # Tensor API 精确到两个 rigid body；不伪造其内部 shape 路径。
                        "collider0": None,
                        "collider1": None,
                        "other_actor": entity["root_path"],
                        "other_collider": None,
                        "force": force,
                        "impulse": force * float(dt),
                        "separation": float("nan"),
                        "event_type": "TENSOR_CONTACT",
                        "backend": "physx_tensor",
                        "step": self._step_index - 1,
                    }
                    if pair_key not in self._active_pairs:
                        new_records.setdefault(env_id, []).append(record)
                    self._active_pairs[pair_key] = record
                    self._debug["tensor_pairs"] += 1

                regular_view = tensor_state["regular_view"]
                regular_forces = regular_view.get_contact_force_matrix(dt).reshape(
                    regular_view.sensor_count, regular_view.filter_count, 3
                ).norm(dim=-1).detach().cpu().tolist()
                for finger_index, finger_path in enumerate(tensor_state["fingers"]):
                    for filter_index, entity_key in enumerate(
                        tensor_state["regular_entity_keys"]
                    ):
                        force = float(regular_forces[finger_index][filter_index])
                        accept_pair(finger_path, entity_key, force)

                nested_view = tensor_state["nested_view"]
                if nested_view is not None:
                    nested_forces = nested_view.get_contact_force_matrix(dt).reshape(
                        nested_view.sensor_count, nested_view.filter_count, 3
                    ).norm(dim=-1).detach().cpu().tolist()
                    for object_index, entity_key in enumerate(
                        tensor_state["nested_entity_keys"]
                    ):
                        for finger_index, finger_path in enumerate(tensor_state["fingers"]):
                            force = float(nested_forces[object_index][finger_index])
                            accept_pair(finger_path, entity_key, force)

                for key, old in list(self._active_pairs.items()):
                    if (
                        key[0] == env_id
                        and old.get("backend") == "physx_tensor"
                        and key not in current_tensor_keys
                    ):
                        self._active_pairs.pop(key, None)
                        ended_records.setdefault(env_id, []).append(old)

            result[env_id]["active"] = [
                record for key, record in self._active_pairs.items() if key[0] == env_id
            ]
            result[env_id]["new"] = new_records.get(env_id, [])
            result[env_id]["ended"] = ended_records.get(env_id, [])
        return result

    def reset_debug(self) -> None:
        for key in self._debug:
            self._debug[key] = 0
        self._debug_samples.clear()

    def get_debug(self) -> dict:
        return dict(self._debug)

    def get_debug_samples(self) -> list:
        """返回少量原始 header 路径，用于诊断 collider 层级而不刷屏。"""
        return list(self._debug_samples)

    @property
    def is_ready(self) -> bool:
        return (
            self._interface is not None
            and bool(self._monitored_fingers)
            and len(self._tensor_views) == self.num_envs
        )
