import os
import sys
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from scene import Scene

def main():
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    scene = Scene(description="State Test", num_envs=1)
    objects = scene.create_clutter_environment(num_objects_range=(3,5))
    # State is initialized in Scene
    state = scene.state
    
    print("Stepping simulation...")
    for _ in range(20):
        scene.step()
        

    processed_rgb, processed_depth = state.get_img()
    
    if processed_rgb is not None and processed_depth is not None:
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        
        print(f"Captured images via state.get_img. RGB Shape: {processed_rgb.shape}")
    else:
        print("Failed to capture image.")

    print("Image captured. keeping simulation running...")
    while scene.is_app_running():
        scene.step()

    scene.simulation_app.close()

if __name__ == "__main__":
    main()
