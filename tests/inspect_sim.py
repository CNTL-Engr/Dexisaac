"""
统一仿真信息验证脚本。

用途: 需要从 Isaac Sim 里确认某个运行期信息 (body 名 / 关节名 / 传感器读数 /
      推点几何 / 接触力等) 时, 都往这里加一个 "检查项", 用 --check 选择运行。
      避免每次为了确认一个信息临时写一个一次性脚本。

用法:
  cd <Dexisaac_root>
  python inspect_sim.py --check body_names      # 打印 articulation 的 body/joint 名, 定位 finger pad body
  python inspect_sim.py --check push_point --target_object 005 --num_objects_min 4 --num_objects_max 7 --action 0 1 2 3
  python inspect_sim.py --check list            # 列出所有可用检查项

新增检查项:
  1. 写一个 def check_xxx(ctx): ... 函数 (ctx 见 InspectContext)
  2. 在 CHECKS 字典里注册 "xxx": check_xxx
  3. 用 python inspect_sim.py --check xxx 运行
"""

import os
import sys
import random

# 脚本位于 Dexisaac/tests/ 下；项目模块位于其父目录的 src/ 和 train/。
# 之前以 tests/ 作为基准拼接路径，导致 train/action_primitive.py 无法导入。
_CUR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CUR)
_SRC = os.path.join(_ROOT, "src")
_TRAIN = os.path.join(_ROOT, "train")
for _p in (_SRC, _TRAIN, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class InspectContext:
    """检查项拿到的上下文。惰性启动 Isaac, 只有真正需要时才拉起仿真。"""

    def __init__(self, num_envs=1, args=None):
        self.num_envs = num_envs
        self.args = args
        self._scene = None

    @property
    def scene(self):
        if self._scene is None:
            from scene import Scene
            print("[inspect] 启动 Isaac Sim (首次访问 scene 时惰性启动)...")
            self._scene = Scene(description="Inspect Sim", num_envs=self.num_envs)
        return self._scene

    @property
    def robot0(self):
        return self.scene.robots[0]

    def close(self):
        if self._scene is not None:
            try:
                self._scene.close()
            except Exception as e:
                print(f"[inspect] 关闭 app 时出错 (可忽略): {e}")


def check_body_names(ctx):
    """
    打印 articulation 的 body 名和 joint 名, 并高亮 finger pad body。
    目的: 为接触传感器 (点1) 确定 finger pad 的 body/link 名和 body 索引。
    """
    art = ctx.robot0.articulation

    body_names = list(art.body_names)
    joint_names = list(art.joint_names)

    print("\n" + "=" * 70)
    print(f"[body_names] 共 {len(body_names)} 个 body:")
    print("=" * 70)
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")

    print("\n" + "-" * 70)
    print(f"[joint_names] 共 {len(joint_names)} 个 joint:")
    print("-" * 70)
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name}")

    # 高亮候选 finger pad body (接触传感器要挂的目标)
    kw = ("finger_pad", "inner_finger", "finger", "pad")
    print("\n" + "=" * 70)
    print("[候选 finger body] (接触传感器 prim_path 用这些 body 名):")
    print("=" * 70)
    hits = [(i, n) for i, n in enumerate(body_names)
            if any(k in n.lower() for k in kw)]
    if hits:
        for i, n in hits:
            print(f"  ✓ body_idx={i:2d}  name='{n}'")
        print(f"\n  robot prim_path 模板: {ctx.scene.gripper_prim_path}")
        print("  → 传感器 prim_path 形如: "
              f"{ctx.scene.gripper_prim_path}/<上面的 body 名>")
    else:
        print("  ⚠ 未匹配到明显的 finger/pad body, 请人工从上面完整列表里挑。")
    print("=" * 70 + "\n")


def check_table_depth(ctx):
    """
    测固定台面深度 (相机到桌面台面的距离, 米)。用于固定尺度 depth 归一化的 DEPTH_FAR。
    相机和桌子都是静态的, 所以这是一个常数, 测一次即可。
    """
    import numpy as np

    scene = ctx.scene
    # 让物理稳定几步 (此时未 spawn 物体, 画面 = 台面 + 地面)
    for _ in range(10):
        scene.step()

    cam = scene.cameras[0]
    images = cam.get_images(hide_robot=True)
    if images is None:
        print("[table_depth] ⚠ 无法获取相机图像")
        return
    depth = images.get('distance_to_image_plane')
    if depth is None:
        print("[table_depth] ⚠ 无深度通道")
        return
    depth = depth.cpu().numpy() if hasattr(depth, 'cpu') else np.asarray(depth)
    depth = np.squeeze(depth)

    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        print("[table_depth] ⚠ 无有效深度")
        return

    vals, counts = np.unique(valid, return_counts=True)
    order = np.argsort(counts)[::-1]

    print("\n" + "=" * 70)
    print("[table_depth] 最频繁的深度值 (Isaac 深度常为离散值):")
    print("=" * 70)
    for i in order[:6]:
        print(f"  depth = {vals[i]:.4f} m    count = {counts[i]}")

    top2 = np.sort(vals[order[:2]])
    table_d = float(top2[0])   # 较小 = 较近 = 台面
    floor_d = float(top2[-1])  # 较大 = 较远 = 地面
    print("-" * 70)
    print(f"  近平面(台面) 深度 ≈ {table_d:.4f} m   (灰度 0 应对应此值)")
    print(f"  远平面(地面) 深度 ≈ {floor_d:.4f} m")
    print(f"  台面离地高度      ≈ {floor_d - table_d:.4f} m   (= 桌子厚度+腿高)")
    print(f"\n  ==> 固定归一化 DEPTH_FAR (台面) = {table_d:.4f} m")
    print("=" * 70 + "\n")


def check_object_max_height(ctx):
    """
    遍历物体池 (meshdata_CH + meshdata_target) 读每个模型 bbox, ×0.01 缩放后取最长边。
    最长边 = 物体翻转竖起的最坏高度, 用于确定固定归一化的 MAX_HEIGHT 常数。
    """
    import os
    from project_paths import resolve_project_path

    SCALE = 0.01  # scene.py spawn 时的 scale=(0.01,0.01,0.01)

    # bbox 读取只依赖 pxr(USD), 不需要跑物理; 但 pxr 可能要 app 起来才可导入。
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        _ = ctx.scene  # 惰性启动 Isaac 后再导
        from pxr import Usd, UsdGeom

    pools = {
        "obstacle(meshdata_CH)": resolve_project_path("meshdata/meshdata_CH"),
        "target(meshdata_target)": resolve_project_path("meshdata/meshdata_target"),
    }

    global_max = 0.0
    for label, root in pools.items():
        print("\n" + "=" * 70)
        print(f"[{label}]  {root}")
        print("=" * 70)
        if not os.path.exists(root):
            print(f"  ⚠ 路径不存在")
            continue
        models = [d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "textured.usd"))]
        print(f"  共 {len(models)} 个模型")
        pool_max = 0.0
        for m in sorted(models):
            usd_path = os.path.join(root, m, "textured.usd")
            try:
                stage = Usd.Stage.Open(usd_path)
                bbox_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                )
                prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
                rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                size = rng.GetSize()
                dims = (size[0] * SCALE, size[1] * SCALE, size[2] * SCALE)
                longest = max(dims)
                pool_max = max(pool_max, longest)
                print(f"  {m:28s} dims(m)=({dims[0]:.3f},{dims[1]:.3f},{dims[2]:.3f}) "
                      f"longest={longest:.3f}")
            except Exception as e:
                print(f"  {m:28s} ⚠ bbox 读取失败: {e}")
        print(f"  → [{label}] 池内最长边最大值 = {pool_max:.3f} m")
        global_max = max(global_max, pool_max)

    print("\n" + "=" * 70)
    print(f"  ==> 所有物体最长边最大值 = {global_max:.3f} m (翻转竖起最坏情况)")
    print(f"  ==> 建议 MAX_HEIGHT = {global_max * 1.1:.3f} m (留 10% 余量)")
    print("=" * 70 + "\n")


def check_gripper_prims(ctx):
    """只读遍历夹爪 prim 层级, 报告完整路径/是否 instanceable/是否有刚体API/接触API。用于诊断脱件。"""
    # 用 DISABLE_GRIPPER_CONTACT=1 拉起 pristine stage (不装接触传感器), 纯观察原始结构
    import os
    os.environ.setdefault("DISABLE_GRIPPER_CONTACT", "1")
    scene = ctx.scene  # 必须先启动 Isaac, omni.usd 才可导入
    import omni.usd
    from pxr import UsdPhysics, PhysxSchema, Usd
    stage = omni.usd.get_context().get_stage()

    root_path = scene.gripper_prim_path  # 单环境无通配符
    print(f"\n[gripper_prims] 遍历 {root_path} 下的所有 prim:\n")

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"  ⚠ 夹爪根 prim 无效: {root_path}")
        return

    # 手动递归遍历 (含 instance proxy), 比 PrimRange 的谓词签名更可移植
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    def _walk(prim):
        yield prim
        for child in prim.GetFilteredChildren(predicate):
            yield from _walk(child)

    n_rigid = 0
    for prim in _walk(root):
        path = prim.GetPath().pathString
        is_inst = prim.IsInstance()
        is_proxy = prim.IsInstanceProxy()
        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_cr = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        # 只打印有意义的层 (刚体 / instance / finger 相关)
        name = path.rsplit("/", 1)[-1]
        interesting = has_rb or is_inst or is_proxy or ("finger" in name.lower()) or ("knuckle" in name.lower())
        if interesting:
            tags = []
            if is_inst: tags.append("INSTANCE")
            if is_proxy: tags.append("proxy")
            if has_rb: tags.append("RigidBody");
            if has_cr: tags.append("ContactAPI")
            if has_rb:
                n_rigid += 1
            print(f"  {path}   [{', '.join(tags) if tags else '-'}]")

    print(f"\n  刚体总数 = {n_rigid}")
    print("  → 若 finger 刚体带 INSTANCE/proxy 标志, 则给它加 ContactAPI 会破坏实例(脱件根因)。")
    print("    解法: 加 API 前先对该 prim 设 instanceable=False, 或只在最内层非实例刚体上加。")


def _model_folder_from_prim_path(prim_path):
    """
    [点1-展示] 从物体 prim_path 提取模型文件夹名 (读取模型时所在的文件夹, 如 071 / 005)。
      目标物体 prim leaf: 'Target_{model}'   → 去 'Target_' 前缀
      障碍物   prim leaf: 'Obj_{idx}_{model}' → 去 'Obj_{idx}_' 前缀
    model 名本身可能含下划线 (如 003_cracker), 故用带 maxsplit 的切分保住完整名。
    """
    leaf = prim_path.split('/')[-1]
    if leaf.startswith('Target_'):
        return leaf[len('Target_'):]
    if leaf.startswith('Obj_'):
        parts = leaf.split('_', 2)  # ['Obj', idx, model]
        if len(parts) == 3:
            return parts[2]
    return leaf


def _print_contact_events(env):
    """
    [点1-展示] 消费 env._contact_events (prim_path), 翻成模型文件夹序号并打印。
    push_start: 报"准备推谁"。first_contact: 报"意图 vs 第一下实际碰到"。
    """
    def fmt(p):
        if not p:
            return "未知"
        if p == 'table':
            return "桌面(table)"
        return f"{_model_folder_from_prim_path(p)}[{p.split('/')[-1]}]"

    events = getattr(env, "_contact_events", []) or []
    saw_first = False
    for ev in events:
        if ev['type'] == 'push_start':
            print(f"  🎯 [Env {ev['env_idx']}] 准备推{ev['kind']}: "
                  f"目标模型ID={ev.get('intended_model_id') or '未知'} "
                  f"prim={fmt(ev.get('intended_prim_path'))} "
                  f"(direction={ev['direction']})")
        elif ev['type'] == 'first_contact':
            saw_first = True
            verdict = '✅合法' if not ev['illegal'] else '🚫非法'
            phase = {'descent': '下降段', 'push': '推动段'}.get(ev.get('phase'), ev.get('phase', '?'))
            print(f"  {verdict} [Env {ev['env_idx']}] 推{ev['kind']} [{phase}] | "
                  f"目标ID={ev.get('intended_model_id') or '未知'} "
                  f"prim={fmt(ev.get('intended_prim_path'))} | "
                  f"实际ID={ev.get('hit_model_id') or 'TABLE'} "
                  f"prim={fmt(ev.get('hit_prim_path'))} ({ev.get('first_hit')}) | "
                  f"手指={str(ev.get('finger_prim_path', '?')).split('/')[-1]} | "
                  f"力={ev.get('force', 0.0):.3f}N step={ev.get('step', '?')}")
            if ev.get('actor0') or ev.get('actor1'):
                print(f"      actor:    {ev.get('actor0')}  <->  {ev.get('actor1')}")
                print(f"      collider: {ev.get('collider0')}  <->  {ev.get('collider1')}")
    if not saw_first:
        print("  (本次推动四指未碰到任何刚体 → 空推/未接触)")


def check_contact_force(ctx):
    """驱动真实推动，用 PhysX 成对接触力打印双方刚体、模型 ID 与阶段判定。"""
    scene = ctx.scene  # 触发 Isaac 启动 + PhysxContactReportAPI 激活

    class _Args:
        device = "cuda:0"
        episode_max_steps = 8
        num_objects_min = 5
        num_objects_max = 8

    _sys_argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0]]  # 防止 PushEnv 内部再解析命令行
    from env_wrapper import PushEnv
    env = PushEnv(scene=scene, args=_Args())
    sys.argv = _sys_argv_backup
    env.contact_debug = True        # 开启接触事件原始记录 (供本脚本翻译打印)

    tracker = getattr(scene, "contact_tracker", None)
    print(f"\n[contact_force] PhysX contact tracker={'已启用' if tracker else 'None(未启用)'}")

    n_pushes = 4
    print(f"[contact_force] 将执行 {n_pushes} 次推动, 打印接触判定 (意图 vs 第一下实际碰到)...\n")

    import random
    for i in range(n_pushes):
        # 每次推动前重置场景: step() 的同步重置只标 done、真正重置交给调用方,
        # 所以这里必须自己 reset, 否则第 1 次推动后 env 已 done → 后续推动 0 步。
        states, spawned = env.reset()
        action_idx = random.randint(0, 7)  # 0-3 推目标, 4-7 推障碍
        actions = [action_idx] + [0] * (env.num_envs - 1)

        if tracker is not None:
            tracker.reset_debug()

        env.step(actions, spawned)

        kind = "推目标" if action_idx <= 3 else "推障碍"
        print(f"\n  ── 推动{i} (action={action_idx} {kind}) ──")
        if tracker is not None:
            dbg = tracker.get_debug()
            print(
                f"  [诊断] tracker就绪={tracker.is_ready} "
                f"PhysX headers={dbg['headers']} 手指相关={dbg['finger_pairs']} "
                f"有效接触={dbg['accepted_pairs']} 低于阈值={dbg['below_threshold']} "
                f"回调/即时={dbg['callback_headers']}/{dbg['immediate_headers']} "
                f"GPU成对接触={dbg['tensor_pairs']} "
                f"阈值={tracker.force_threshold:.3f}N"
            )
            if dbg['accepted_pairs'] == 0 and dbg['headers'] > 0:
                print("  [诊断] PhysX 有报告但未归属到四指/物体，原始样本:")
                for actor0, actor1, collider0, collider1, event_type in tracker.get_debug_samples():
                    print(f"      {event_type}: actor {actor0} <-> {actor1}")
                    print(f"                    collider {collider0} <-> {collider1}")

        _print_contact_events(env)

    print("\n  → 判读: '第一下实际碰到' 应等于 '目标物体序号' 才合法; 碰到别的物体/桌面即非法。")
    print("    GPU 接触矩阵直接对应两侧刚体；原始 report 可用时另附 collider 路径，均不依赖分割图判断实际接触物。")


def check_table_collider(ctx):
    """检查 Table / Ground 是否有 collider (CollisionAPI + collisionEnabled), 以及是否刚体。
       回答'桌子有没有碰撞网格'——若无 collider, 则'碰桌面'的净力根本不来自桌子。"""
    import os
    os.environ.setdefault("DISABLE_GRIPPER_CONTACT", "1")
    scene = ctx.scene
    import omni.usd
    from pxr import UsdPhysics, Usd
    stage = omni.usd.get_context().get_stage()

    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for name in ("Table", "Ground"):
        root_path = f"/World/Scene/{name}"
        root = stage.GetPrimAtPath(root_path)
        print(f"\n[table_collider] {root_path}: valid={root.IsValid()}")
        if not root.IsValid():
            continue
        n_coll = 0
        for prim in Usd.PrimRange(root, pred):
            path = prim.GetPath().pathString
            ptype = prim.GetTypeName()
            has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            enabled = None
            if has_coll:
                ca = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
                attr = ca.GetCollisionEnabledAttr()
                enabled = attr.Get() if attr and attr.HasAuthoredValue() else "(默认True)"
            tags = []
            if has_rb: tags.append("RigidBody")
            if has_coll: tags.append(f"Collision(enabled={enabled})"); n_coll += 1
            if ptype == "Mesh" or tags:
                rel = path[len(root_path):] or "/(self)"
                approx = ""
                if has_coll:
                    ca2 = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
                    if ca2:
                        aa = ca2.GetApproximationAttr()
                        approx = f" approx={aa.Get()}" if aa and aa.HasAuthoredValue() else ""
                print(f"    {rel:36s} <{ptype}> [{', '.join(tags) if tags else '-'}]{approx}")
        print(f"  → {name} 带 CollisionAPI 的 prim 数 = {n_coll}"
              + ("  ⚠ 无 collider!" if n_coll == 0 else ""))
    print("\n  判读: 若 Table 无 collider, 夹爪物理上不会与桌子接触, '碰桌面'的净力另有来源")
    print("        (自碰夹爪其它部件 / Ground 平面 / 未过滤物体), 需改判定逻辑。")


def check_object_collider(ctx):
    """生成一批物体, 检查每个物体的 CollisionAPI/RigidBodyAPI 挂在 root 还是子 prim。
       回答'为何某些物体接触力归不到其 root filter 列'(root filter 匹配不到子 collider)。"""
    scene = ctx.scene

    class _Args:
        device = "cuda:0"; episode_max_steps = 8
        num_objects_min = 5; num_objects_max = 8

    _bak = list(sys.argv); sys.argv = [sys.argv[0]]
    from env_wrapper import PushEnv
    env = PushEnv(scene=scene, args=_Args())
    sys.argv = _bak
    _states, spawned = env.reset()

    import omni.usd
    from pxr import UsdPhysics, Usd
    stage = omni.usd.get_context().get_stage()
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    print(f"\n[object_collider] 检查 {len(spawned)} 个物体的 collider/rigidbody 挂载层级:\n")
    for obj in spawned:
        root_path = obj.cfg.prim_path
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"  {root_path}: ⚠ 无效"); continue
        root_rb = root.HasAPI(UsdPhysics.RigidBodyAPI)
        root_coll = root.HasAPI(UsdPhysics.CollisionAPI)
        coll_locs = []
        for prim in Usd.PrimRange(root, pred):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                rel = prim.GetPath().pathString[len(root_path):] or "/(self)"
                coll_locs.append(rel)
        leaf = root_path.split('/')[-1]
        print(f"  {leaf}: rootRigidBody={root_rb} rootCollision={root_coll} "
              f"| CollisionAPI位置={coll_locs}")
    print("\n  判读: 若 CollisionAPI 在子 prim (非 /(self)), 则用 root 精确路径做 filter")
    print("        匹配不到该 collider → 接触力归不到该列 → 误判'空推'。解法: filter 用 root/.* 通配。")


def check_push_point(ctx):
    """按正式训练流程生成杂乱场景，保存候选区/推点并真实执行指定动作。"""
    import re
    from datetime import datetime
    from pathlib import Path

    import cv2
    import numpy as np

    from action_primitive import (
        compute_push_point_from_action,
    )
    from env_wrapper import PushEnv
    from project_paths import project_path, resolve_project_path

    target_value = ctx.args.target_object
    if target_value is None and ctx.args.model_path:
        if len(ctx.args.model_path) != 1:
            raise ValueError("正式场景模式一次只指定一个目标；请使用 --target_object")
        target_value = ctx.args.model_path[0]
    if not target_value:
        raise ValueError("push_point 需要 --target_object <模型ID/模型目录/textured.usd>")
    if not ctx.args.action:
        raise ValueError("push_point 需要 --action，后接一个或多个 0..7 动作序号")
    invalid_actions = [a for a in ctx.args.action if not 0 <= a <= 7]
    if invalid_actions:
        raise ValueError(f"动作序号必须在 0..7，收到非法值: {invalid_actions}")
    if ctx.args.num_objects_min < 1:
        raise ValueError("--num_objects_min 至少为 1（包含目标物体）")
    if ctx.args.num_objects_max < ctx.args.num_objects_min:
        raise ValueError("--num_objects_max 不能小于 --num_objects_min")

    raw_target = Path(target_value).expanduser()
    if not raw_target.is_absolute() and not raw_target.exists():
        if len(raw_target.parts) == 1:
            raw_target = Path(resolve_project_path(
                f"meshdata/meshdata_target/{target_value}"
            ))
        else:
            raw_target = Path(resolve_project_path(str(raw_target)))
    else:
        raw_target = raw_target.resolve()
    target_dir = raw_target.parent if raw_target.is_file() else raw_target
    target_usd = target_dir / "textured.usd"
    if not target_usd.is_file():
        raise FileNotFoundError(f"目标模型不存在或缺少 textured.usd: {target_dir}")

    target_model = target_dir.name
    safe_target_model = re.sub(r"[^0-9A-Za-z_-]+", "_", target_model)
    force_task_config = {
        "target_model_dir": str(target_dir.parent),
        "target_model": target_model,
    }

    scene = ctx.scene

    class _PushArgs:
        device = "cuda:0"
        episode_max_steps = max(1, len(ctx.args.action))
        num_objects_min = ctx.args.num_objects_min
        num_objects_max = ctx.args.num_objects_max

    # 与正式训练一致地通过 PushEnv.reset 生成场景、缓存首帧观测并初始化执行状态。
    env = PushEnv(scene=scene, args=_PushArgs())
    env.contact_debug = True
    _states, spawned_objects = env.reset(force_task_config=force_task_config)
    if not spawned_objects:
        raise RuntimeError("正式场景生成失败：没有生成任何物体")

    # 保存首次稳定后的完整动力学状态。后续动作不重新随机生成场景，而是
    # 精确写回同一批刚体和机器人的状态，保证每个动作从相同初始条件开始。
    initial_object_states = {
        obj.cfg.prim_path: obj.data.root_state_w.clone()
        for obj in spawned_objects
    }
    initial_robot_states = [
        (
            robot.articulation.data.joint_pos.clone(),
            robot.articulation.data.joint_vel.clone(),
        )
        for robot in scene.robots
    ]

    def restore_initial_scene():
        """恢复首次动作前的物体/机器人/PushEnv 状态并重建同帧观测。"""
        for obj in spawned_objects:
            obj.write_root_state_to_sim(initial_object_states[obj.cfg.prim_path])

        for robot, (joint_pos, joint_vel) in zip(scene.robots, initial_robot_states):
            articulation = robot.articulation
            articulation.write_joint_state_to_sim(joint_pos, joint_vel)
            articulation.set_joint_position_target(joint_pos)
            articulation.write_data_to_sim()
            robot.current_joint_targets = joint_pos.clone()
            robot.reset_ik_status()

        # forward 只同步写入，不推进物理时间，避免恢复后场景再次漂移。
        scene.sim.forward()

        env.current_step = 0
        env.env_dones.fill_(False)
        env.env_steps.fill_(0)
        env.previous_actions = [None for _ in range(env.num_envs)]
        env.opposite_action_streaks = [0 for _ in range(env.num_envs)]
        env.previous_empty_pushes = [False for _ in range(env.num_envs)]
        env.ik_failed_blacklist.clear()
        env.illegal_contact_envs.clear()
        env.action_contact_results.clear()
        env.action_push_measurements.clear()
        env.dynamics_explosion_envs.clear()
        env.dynamics_explosion_peaks.clear()
        env.dynamics_explosion_monitor_step = 0
        env.dynamics_explosion_speed_streaks.clear()
        env.dynamics_explosion_speed_streak_starts.clear()
        env.dynamics_explosion_max_speed_streaks.clear()
        env.dynamics_explosion_speed_intervals.clear()
        env.dynamics_explosion_acceleration_events.clear()
        env._contact_events = []

        env.initial_obj_positions = {
            obj.cfg.prim_path.rsplit('/', 1)[-1]: obj.data.root_pos_w[0].clone()
            for obj in spawned_objects
        }
        env.initial_target_pos = env._get_target_position(spawned_objects)
        env.previous_target_pos = env.initial_target_pos.clone()
        env._precompute_target_grasp_specs(
            spawned_objects, env_ids=range(env.num_envs)
        )
        return env._get_observations(spawned_objects)

    output_base_dir = Path(ctx.args.output_dir or project_path("debug"))
    if not output_base_dir.is_absolute():
        output_base_dir = Path(resolve_project_path(str(output_base_dir)))
    save_time = datetime.now().strftime("%H%M%S")
    output_dir = output_base_dir / (
        f"{safe_target_model}_{ctx.args.num_objects_min}_"
        f"{ctx.args.num_objects_max}_{ctx.args.seed}_{save_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    direction_names = {0: "X+", 1: "Y+", 2: "X-", 3: "Y-"}

    print("\n" + "=" * 78)
    print(
        f"[push_point] 正式场景: target={target_model}, "
        f"objects=[{ctx.args.num_objects_min}, {ctx.args.num_objects_max}], "
        f"envs={ctx.num_envs}, actions={ctx.args.action}"
    )
    print(f"[push_point] 图片目录: {output_dir}")
    print("=" * 78)

    env_objects_by_id = []
    for env_idx, state in enumerate(scene.states):
        env_objects = [obj for obj in spawned_objects if state._is_obj_in_env(obj)]
        if not env_objects:
            raise RuntimeError(f"Env {env_idx}: 未找到生成物体")
        env_objects_by_id.append(env_objects)
        spawned_names = [obj.cfg.prim_path.rsplit('/', 1)[-1] for obj in env_objects]
        print(f"  [Env {env_idx}] 总物体数={len(env_objects)}: {spawned_names}")

    for step_idx, action_idx in enumerate(ctx.args.action):
        action_idx = int(action_idx)
        if step_idx > 0:
            _states = restore_initial_scene()
            print(f"[push_point] 已恢复完全相同的初始场景，准备执行 action={action_idx}")

        # 每个环境先从训练正在使用的动作前观测保存候选区和实际选中推点。
        for env_idx, state in enumerate(scene.states):
            env_objects = env_objects_by_id[env_idx]
            total_objects = len(env_objects)
            debug_info = {}
            push_point, direction_idx, contact_spec = compute_push_point_from_action(
                action_idx, env_idx, state, scene, env_objects,
                debug_info=debug_info,
            )
            base_rgb = np.asarray(debug_info["rgb"]).copy()
            if base_rgb.dtype != np.uint8:
                scale = 255.0 if base_rgb.size and float(np.nanmax(base_rgb)) <= 1.0 else 1.0
                base_rgb = np.clip(base_rgb * scale, 0, 255).astype(np.uint8)
            if base_rgb.ndim != 3 or base_rgb.shape[2] < 3:
                base_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_GRAY2RGB)
            base_rgb = base_rgb[..., :3]

            g02_mask = np.asarray(
                debug_info.get(
                    "g02_mask",
                    np.zeros(base_rgb.shape[:2], dtype=bool),
                ),
                dtype=bool,
            )
            window_center_mask = np.asarray(
                debug_info.get(
                    "window_center_mask", np.zeros(base_rgb.shape[:2], dtype=bool)
                ),
                dtype=bool,
            )
            obs_mask = np.asarray(
                debug_info.get("obs_mask", np.zeros(base_rgb.shape[:2], dtype=bool)),
                dtype=bool,
            )
            height_feasible_mask = np.asarray(
                debug_info.get(
                    "height_feasible_mask",
                    np.zeros(base_rgb.shape[:2], dtype=bool),
                ),
                dtype=bool,
            )
            overlay = base_rgb.copy()
            overlay[g02_mask] = np.array([0, 0, 255], dtype=np.uint8)
            overlay[obs_mask] = np.array([255, 0, 0], dtype=np.uint8)
            push_pixel = debug_info.get("push_pixel")
            geometry_valid = bool(
                debug_info.get("geometry_valid", push_pixel is not None)
            )
            selected_window_bounds = debug_info.get("selected_window_bounds")
            if geometry_valid and selected_window_bounds is not None:
                u0, u1, v0, v1 = (int(value) for value in selected_window_bounds)
                cv2.rectangle(overlay, (u0, v0), (u1, v1), (255, 255, 255), 1)
            else:
                invalid_reason = debug_info.get("geometry_invalid_reason")
                if debug_info.get("outcome") == "empty":
                    invalid_reason = debug_info.get(
                        "empty_reason", "empty_push"
                    )
                cv2.putText(
                    overlay,
                    f"NO VALID PUSH: {invalid_reason or 'unknown'}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 0, 0), 1, cv2.LINE_AA,
                )

            output_path = output_dir / (
                f"push_point_target{safe_target_model}_n{total_objects}_"
                f"env{env_idx}_step{step_idx}_action{action_idx}.png"
            )
            if not cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"图像保存失败: {output_path}")

            point = push_point.detach().cpu().tolist() if push_point is not None else None
            print("-" * 78)
            print(
                f"  step={step_idx}, action={action_idx} "
                f"({'目标' if action_idx <= 3 else '障碍'}, "
                f"{direction_names[direction_idx]}) intended={contact_spec.get('intended_model_id')}"
            )
            print(
                f"  G02像素={int(np.count_nonzero(g02_mask))}, "
                f"Obs像素={int(np.count_nonzero(obs_mask))}, "
                f"高度可行像素={int(np.count_nonzero(height_feasible_mask))}, "
                f"合法窗口中心={int(np.count_nonzero(window_center_mask))}, "
                f"geometry_valid={debug_info.get('geometry_valid', False)}"
            )
            obstacle_selection = debug_info.get("obstacle_selection") or {}
            if obstacle_selection:
                print(
                    f"  障碍射线: outcome={obstacle_selection.get('outcome')}, "
                    f"tier={obstacle_selection.get('selected_tier')}, "
                    f"selected_seg={obstacle_selection.get('selected_seg_id')}"
                )
            selected_segment = debug_info.get("selected_segment_index")
            selected_range = debug_info.get("selected_segment_range")
            print(
                f"  线段总数={int(debug_info.get('segment_count', 0))}, "
                f"宽度合格={int(debug_info.get('wide_segment_count', 0))}, "
                f"选中线段={selected_segment}, 范围={selected_range}, "
                f"断段阈值={float(debug_info.get('segment_gap_threshold_px', 0.0)):.2f}px"
            )
            if push_pixel is None:
                print(
                    f"  几何无效: reason={debug_info.get('geometry_invalid_reason', 'unknown')}"
                )
            else:
                print(f"  实际推点: pixel={push_pixel}, world={point} m")
                print(
                    f"  窗口={selected_window_bounds}, "
                    f"hT_P90={debug_info.get('h_target_p90')}, "
                    f"hChannelMax={debug_info.get('channel_max_height')}, "
                    f"channelClearance="
                    f"{debug_info.get('channel_z_clearance_m')}"
                )
            print(f"  输出: {output_path}")

        # PushEnv 会从同一缓存帧再次解析推点并驱动所有未结束环境的机械臂执行动作。
        actions = [action_idx for _ in range(ctx.num_envs)]
        _next_states, rewards, dones, infos = env.step(actions, spawned_objects)
        print(f"[push_point] step={step_idx}, action={action_idx} 执行完成")
        for env_idx in range(ctx.num_envs):
            info = infos[env_idx]
            print(
                f"  [Env {env_idx}] reward={float(rewards[env_idx]):.4f}, "
                f"done={bool(dones[env_idx])}, success={info.get('success', False)}, "
                f"empty_push={info.get('empty_push', info.get('is_empty_push', False))}, "
                f"ik_failed={info.get('ik_failed', False)}"
            )
        _print_contact_events(env)
    print("=" * 78 + "\n")


CHECKS = {
    "body_names": check_body_names,
    "table_depth": check_table_depth,
    "object_max_height": check_object_max_height,
    "gripper_prims": check_gripper_prims,
    "contact_force": check_contact_force,
    "table_collider": check_table_collider,
    "object_collider": check_object_collider,
    "push_point": check_push_point,
}


def _apply_seed(seed):
    """给 random / numpy / torch 播种, 返回实际使用的种子。
       未指定时随机生成一个并返回, 以便打印复现。"""
    if seed is None:
        seed = random.randrange(2 ** 31)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="统一仿真信息验证脚本")
    parser.add_argument("--check", default="list",
                        help="要运行的检查项名 (--check list 查看全部)")
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int,
                        help="随机种子; 不指定则随机生成并打印, 指定后可复现同一场景+动作")
    parser.add_argument("--target_object", "--target_model", default=None,
                        help="push_point: 目标模型ID、模型目录或 textured.usd 路径")
    parser.add_argument("--model_path", default=None, nargs="+",
                        help="push_point: 旧参数兼容；正式场景模式只允许传一个目标")
    parser.add_argument("--num_objects_min", default=4, type=int,
                        help="push_point: 最少总物体数（含1个目标）")
    parser.add_argument("--num_objects_max", default=7, type=int,
                        help="push_point: 最大总物体数（含1个目标）")
    parser.add_argument("--output_dir", default=None,
                        help="push_point: 输出目录，默认 Dexisaac/debug")
    parser.add_argument("--action", "--action_idx", dest="action", default=None,
                        nargs="+", type=int,
                        help="push_point: 在同一初始场景中分别执行动作 0..7，如 --action 0 1 2 3")
    # 只解析已知参数, 其余留给 AppLauncher (与 config.initialize_app 一致)
    args, _ = parser.parse_known_args()

    if args.check == "list":
        print("可用检查项:")
        for name in CHECKS:
            fn = CHECKS[name]
            doc = (fn.__doc__ or "").strip().splitlines()
            summary = doc[0] if doc else ""
            print(f"  {name:16s} - {summary}")
        return

    if args.check not in CHECKS:
        print(f"未知检查项: {args.check}")
        print(f"可用: {list(CHECKS.keys())}  (或 --check list)")
        sys.exit(1)

    num_envs = args.num_envs

    # 播种必须在 Isaac 启动/场景生成/动作采样之前 (Isaac 启动不会重置 python random/numpy)。
    seed = _apply_seed(args.seed)
    # 保存实际使用的种子（包括未显式传 --seed 时生成的随机种子），供检查项
    # 输出命名和复现实验使用。
    args.seed = seed
    print("=" * 60)
    print(f"[inspect_sim] 随机种子 = {seed}")
    print(f"             复现命令: --check {args.check} --num_envs {num_envs} --seed {seed}")
    print("=" * 60)

    ctx = InspectContext(num_envs=num_envs, args=args)
    try:
        CHECKS[args.check](ctx)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
