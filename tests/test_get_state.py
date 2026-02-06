import sys
sys.path.insert(0, "/home/wyq/xc/equi/IsaacLab/scripts/workspace/src")
from scene import Scene
import torch

# 创建场景
print("[TEST] Creating scene...")
scene = Scene(description="Get State Test", num_envs=1)

# 生成环境
print("[TEST] Generating clutter environment...")
objects = scene.create_clutter_environment(num_objects_range=(10, 10))

# 重置相机
print("[TEST] Resetting cameras...")
scene.reset_cameras()

# 运行模拟
print("[TEST] Running simulation...")
for i in range(50):
    scene.sim.step()

# 测试get_state方法
print("\n" + "="*60)
print("测试 get_state() 方法")
print("="*60)

state = scene.states[0]
state_tensor = state.get_state(objects)

if state_tensor is not None:
    print(f"\n✓ 成功生成状态tensor")
    print(f"  - Shape: {state_tensor.shape}")
    print(f"  - Dtype: {state_tensor.dtype}")
    print(f"  - Device: {state_tensor.device}")
    print(f"\n通道信息:")
    print(f"  - Channel 0 (Depth):    min={state_tensor[0,0].min():.3f}, max={state_tensor[0,0].max():.3f}")
    print(f"  - Channel 1 (Target):   min={state_tensor[0,1].min():.3f}, max={state_tensor[0,1].max():.3f}")
    print(f"  - Channel 2 (Obstacle): min={state_tensor[0,2].min():.3f}, max={state_tensor[0,2].max():.3f}")
    
    # 验证格式
    assert state_tensor.shape == torch.Size([1, 3, 448, 448]), "Shape mismatch!"
    assert state_tensor.dtype == torch.float32, "Dtype mismatch!"
    print("\n✓ 所有验证通过!")
else:
    print("\n✗ 未能生成状态tensor")
