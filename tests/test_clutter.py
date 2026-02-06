import os
import sys
 

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from scene import Scene

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

# === 三明治结构第1层：加载舞台 ===
print("[TEST] Step 1: Loading stage...")
scene = Scene(description="Camera Test", num_envs=1)
objects = scene.create_clutter_environment(num_objects_range=(15,15))
scene.state.reset()


def main():
    print("[TEST] Entering main loop...")
    frame_count = 0
    while scene.is_app_running():
        scene.step()
        
        captured = False  # Flag to skip update if we captured
        
        if frame_count % 20 == 0:
            print(f"[TEST] Frame {frame_count}")
        
        # Capture and save images at frame 50
        if frame_count == 50:
            rgb, depth, seg = scene.state.get_img(hide_robot=True)  # This calls camera.update() internally!
            scene.state.save_img(rgb, depth, seg, "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs")
            captured = True  # Skip the update below
            
        frame_count += 1
        
        # Only update if we didn't capture (to avoid double-rendering)
        if not captured:
            scene.state.update(dt=scene.sim.get_physics_dt())
        
    print("[TEST] Main loop completed!")

if __name__ == "__main__":
    main()

