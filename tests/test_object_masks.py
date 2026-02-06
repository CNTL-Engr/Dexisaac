import sys
sys.path.insert(0, "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/src")
from scene import Scene

# === 创建场景 ===
print("[TEST] Creating scene with 1 environment...")
scene = Scene(description="Object Mask Test", num_envs=1)

# === 生成杂乱环境 ===
print("[TEST] Generating clutter environment...")
objects = scene.create_clutter_environment(num_objects_range=(10, 10))

# === 重置相机 ===
print("[TEST] Resetting cameras...")
scene.reset_cameras()

# === 运行几帧让物理稳定 ===
print("[TEST] Running simulation for 50 frames...")
for i in range(50):
    scene.sim.step()

# === 获取图像并提取物体掩膜 ===
print("\n[TEST] Capturing images and extracting object masks...")
state = scene.states[0]

# 获取图像
rgb, depth, seg = state.get_img()

# 保存原始图像
save_path = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs/mask_test"
state.save_img(rgb, depth, seg, save_path)

# 提取物体掩膜
print("\n" + "="*60)
print("提取物体掩膜")
print("="*60)
object_masks = state.extract_object_masks(seg, exclude_floor=True)

# 保存物体掩膜
masks_dir = f"{save_path}/object_masks"
state.save_object_masks(object_masks, masks_dir)

print("\n" + "="*60)
print(f"✓ 测试完成! 请查看以下目录:")
print(f"  - 原始图像: {save_path}")
print(f"  - 物体掩膜: {masks_dir}")
print("="*60)
