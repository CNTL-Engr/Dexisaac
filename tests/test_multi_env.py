
import sys
sys.path.insert(0, "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/src")
from scene import Scene
# === 创建2个场景 ==
# print("[TEST] Creating 2 environments...")
scene = Scene(description="Multi-Env Test", num_envs=2)
# 生成杂乱环境
# print("[TEST] Generating clutter environments...")
objects = scene.create_clutter_environment(num_objects_range=(15,15))
# Reset相机
# print("[TEST] Resetting cameras...")
scene.reset_cameras()


def main():
    # print("[TEST] Entering main loop...")
    frame_count = 0
    max_frames = 10000
    captured = False
    while frame_count < max_frames:
        if frame_count % 20 == 0:
            # print(f"[TEST] Frame {frame_count}")
            pass
        scene.sim.step()


        if frame_count == 50 and not captured:
            # print("[TEST] Capturing images...")


            for env_idx, state in enumerate(scene.states):
                print(f" Environment {env_idx}...")
                rgb, depth, seg = state.get_img()
                save_path = f"/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs/Scene_{env_idx}"
                state.save_img(rgb, depth, seg, save_path)
                
                # 1. 提取并保存目标物体掩膜
                target_mask = state.extract_target_mask(seg, objects)
                if target_mask is not None:
                    target_mask_path = f"{save_path}/target_mask.png"
                    state.save_target_mask(target_mask, target_mask_path)
                
                # 2. 提取并保存全局掩膜(所有物体)
                global_mask = state.extract_global_mask(seg, exclude_floor=True)
                if global_mask is not None:
                    global_mask_path = f"{save_path}/global_mask.png"
                    state.save_global_mask(global_mask, global_mask_path)
                
                # 3. 获取并保存原始全局掩膜（基于深度图）- 仅用于调试
                raw_global_mask = state.get_raw_global_mask(include_table=True)
                if raw_global_mask is not None:
                    import cv2
                    import os
                    save_path = f"/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs/Scene_{env_idx}"
                    os.makedirs(save_path, exist_ok=True)
                    raw_mask_path = f"{save_path}/raw_global_mask.png"
                    cv2.imwrite(raw_mask_path, raw_global_mask)
                    # print(f"✓ 原始全局掩膜已保存: {raw_mask_path}")
                    
                    # 同时执行出界检测并打印结果
                    out_of_bounds, info = state.check_out_of_bounds(verbose=True)
                    # print(f"  出界检测结果: {'出界' if out_of_bounds else '正常'}")
                    # print(f"  详细信息: {info}")

                captured = True

        frame_count += 1

        if not captured or frame_count != 51:

            scene.update_cameras(dt=scene.sim.get_physics_dt())

 

if __name__ == "__main__":

    main()