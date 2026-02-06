import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scene import Scene
# from state import State # Delayed import
import numpy as np
import cv2
import time
from sam2_wrapper import SAM2Wrapper

def test_sam2_multipoint():
    # Enable cameras
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    # Initialize Scene (State is now instantiated inside Scene)
    scene = Scene(description="SAM2 Multipoint Test", num_envs=1)
    state = scene.state
    
    # Initialize SAM2 Wrapper (via State)
    print("Using SAM2 Wrapper from State...")
    sam2 = state.sam

    # Create clutter environment
    print("Creating cluttered environment...")
    objects = scene.create_clutter_environment(num_objects_range=5)
    
    # Step physics to settle objects
    print("Stepping simulation to settle physics...")
    for _ in range(50):
        scene.step()
    
    # Get current RGB and Depth image
    rgb_img, depth_img = state.get_img()
    
    # Save full RGB for debug
    output_dir = "/home/wyq/xc/equi/IsaacLab/scripts/workspace/test_imgs_multipoint"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv2.imwrite(os.path.join(output_dir, "full_scene_rgb.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    # Get object poses and segment each
    object_states = scene.get_object_poses(objects)
    print(object_states)
    print(f"\nProcessing {len(object_states)} objects...")
    
    for i, obj_state in enumerate(object_states):
        name = obj_state["name"]
        pos = obj_state["position"]

        # Convert world position to pixel coordinates
        u, v = state.world_to_pixel(pos[:2]) # Using x, y
        
        print(f"Object: {name}, World: {pos[:2]}, Pixel: ({u}, {v})")
        
        # 使用SAM2的增强方法(自动生成高斯点+后处理),并获取生成的点和box
        mask_binary, points, box = sam2.SAMpredict_enhanced(
            rgb_img, 
            [u, v], 
            depth_img=depth_img,
            use_gaussian=True,       # 启用高斯点
            use_postprocess=True,    # 启用后处理
            min_area_threshold=30,   # 后处理阈值
            return_points=True,      # 返回生成的点用于可视化
            return_box=True          # 返回生成的box用于可视化
        )
        
        print(f"  -> Generated {len(points)} points for segmentation")
        
        # 3. Apply mask to RGB image
        masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_binary)
        masked_img_bgr = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        
        # 4. Visualization: Draw box (if generated)
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(masked_img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
            print(f"  -> Drew bounding box: [{x1}, {y1}, {x2}, {y2}]")
        
        # 5. Visualization: Draw ALL points
        # Center point (first): Red
        # Gaussian points: Blue
        for idx, pt in enumerate(points):
            px, py = int(pt[0]), int(pt[1])
            if idx == 0:
                color = (0, 0, 255)  # Red for center
            else:
                color = (255, 0, 0)  # Blue for gaussian points
                
            cv2.circle(masked_img_bgr, (px, py), 4, color, -1) 
            cv2.circle(masked_img_bgr, (px, py), 5, (255, 255, 255), 1)
        
        save_path = os.path.join(output_dir, f"{name}_{i}_multipoint.png")
        cv2.imwrite(save_path, masked_img_bgr)
        
        print(f"Saved multipoint segmented image to: {save_path}")
        
    while scene.is_app_running():
        scene.step()
    scene.simulation_app.close()

if __name__ == "__main__":
    test_sam2_multipoint()
