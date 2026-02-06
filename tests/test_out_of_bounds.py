"""
测试出界检测功能

验证基于全局掩膜形状的出界检测逻辑
"""

import os
import sys

# 添加 src 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_path)

# 确保启用相机
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

from scene import Scene
import cv2
import numpy as np


def test_out_of_bounds_detection():
    """测试出界检测功能"""
    print("=" * 80)
    print("测试出界检测功能")
    print("=" * 80)
    
    # 创建场景
    print("\n[1/3] 创建场景...")
    scene = Scene(description="Out of Bounds Test", num_envs=2)
    
    # 生成杂乱环境
    print("[2/3] 生成杂乱环境...")
    spawned_objects = scene.create_clutter_environment(num_objects_range=(10, 15))
    
    # 等待物理稳定
    print("  等待物理稳定...")
    for _ in range(100):
        scene.step()
    
    # 测试每个环境
    print("\n[3/3] 测试出界检测...")
    for env_idx in range(scene.num_envs):
        print(f"\n{'='*60}")
        print(f"环境 {env_idx}")
        print('='*60)
        
        state = scene.states[env_idx]
        
        # 1. 获取原始掩膜
        raw_mask = state.get_raw_global_mask(include_table=True)
        
        if raw_mask is not None:
            print(f"  ✓ 成功获取原始掩膜: {raw_mask.shape}")
            print(f"  - 非零像素数: {np.sum(raw_mask > 0)}")
            
            # 保存掩膜图像
            save_path = f"/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs/raw_mask_env_{env_idx}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, raw_mask)
            print(f"  ✓ 原始掩膜已保存: {save_path}")
        else:
            print("  ✗ 无法获取原始掩膜")
            continue
        
        # 2. 执行出界检测
        out_of_bounds, info = state.check_out_of_bounds(verbose=True)
        
        print(f"\n  检测结果: {'出界' if out_of_bounds else '正常'}")
        print(f"  详细信息: {info}")
        
        # 3. 可视化连通域（用于调试）
        if raw_mask is not None:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                raw_mask, connectivity=8
            )
            
            # 创建彩色可视化
            vis_img = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            
            # 为每个连通域绘制边界框
            for i in range(1, num_labels):  # 跳过背景
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                color = (0, 255, 0) if num_labels == 2 else (0, 0, 255)
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis_img, f"#{i}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 保存可视化
            vis_path = f"/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs/vis_mask_env_{env_idx}.png"
            cv2.imwrite(vis_path, vis_img)
            print(f"  ✓ 可视化已保存: {vis_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 关闭仿真
    scene.simulation_app.close()


if __name__ == "__main__":
    test_out_of_bounds_detection()
