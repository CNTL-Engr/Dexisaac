import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from scene import Scene

def main():
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    scene = Scene(description="Robot State Test", num_envs=1)
    import torch
    robot = scene.robot
    target_pos = torch.tensor([[1.125, 0.375, 0.02]], device=robot.device)
 
    target_quat = robot.fixed_ee_orientation.unsqueeze(0)
    robot.move_to(scene, target_pos, target_quat)
    while scene.is_app_running():
        scene.step()
 
    scene.simulation_app.close()

if __name__ == "__main__":
    main()
