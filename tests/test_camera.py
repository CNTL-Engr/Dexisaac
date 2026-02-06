import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from scene import Scene

def main():
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    scene = Scene(description="Camera Test", num_envs=1)
    object = scene.create_clutter_environment(num_objects_range=(3,5))
    # from camera import Camera 
    # exclude_paths = [scene.robot_prim_path, scene.gripper_prim_path]
    # camera = Camera(prim_path="/World/Camera", exclude_prim_paths=exclude_paths)
    
 
    print("Camera initialized.")
 
    i= 1
    while i:

        scene.step()
        i+=1
        if i == 10:
            print("Capturing image...")
            imgs= scene.state.camera.get_images(hide_robot=True)
            scene.state.camera.save_images(imgs,"/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/test_imgs")
            break
    # print("Test complete.")
    scene.simulation_app.close()

if __name__ == "__main__":
    main()
