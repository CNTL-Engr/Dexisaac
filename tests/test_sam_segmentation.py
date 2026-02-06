import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scene import Scene
# from state import State # Delayed import
import numpy as np
import cv2
import time

def test_sam_segmentation():
    # Enable cameras
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    # Initialize Scene (State is now instantiated inside Scene)
    scene = Scene(description="SAM Test", num_envs=1)
    state = scene.state
    
    # Create clutter environment
    print("Creating cluttered environment...")
    objects = scene.create_clutter_environment(num_objects_range=8)
    
    # Step physics to settle objects
    print("Stepping simulation to settle physics...")
    for _ in range(50):
        scene.step()
    
    # Get current RGB image
    rgb_img, _ = state.get_img()
    
    # Save full RGB for debug
    cv2.imwrite("test_imgs/full_scene_rgb.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    # Create output directory
    output_dir = "test_imgs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
        
        # Segment using SAM
        # SAMpredict expects a list of points
        mask_binary = state.sam.SAMpredict(rgb_img, [[u, v]])
        
        # Apply mask to RGB image
        # Create a masked image where background is black
        masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_binary)
        
        # Save the result
        # Convert RGB back to BGR for cv2 saving
        masked_img_bgr = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        
        # Draw the prompt point for visualization
        # Green filled circle with white outline for high contrast
        cv2.circle(masked_img_bgr, (u, v), 4, (0, 255, 0), -1) 
        cv2.circle(masked_img_bgr, (u, v), 5, (255, 255, 255), 1)
        
        save_path = os.path.join(output_dir, f"{name}_{i}_segmented.png")
        cv2.imwrite(save_path, masked_img_bgr)
        
        print(f"Saved segmented image to: {save_path}")
    while scene.is_app_running():
        scene.step()
    scene.simulation_app.close()

if __name__ == "__main__":
    test_sam_segmentation()
