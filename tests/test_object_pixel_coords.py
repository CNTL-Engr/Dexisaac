import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from scene import Scene

def main():
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    print("Initializing Scene...")
  
    scene = Scene(description="Pixel Coord Test", num_envs=1)
    
    # State is initialized in Scene
    state = scene.state
    
    print("Generating clutter environment...")
    objects = scene.create_clutter_environment(num_objects_range=(3,5))
    
    print("Stepping simulation to settle physics...")
    for _ in range(50):
        scene.step()
 
    
    object_states = scene.get_object_poses(objects)
    rgb_img, depth_img = state.get_img()
    
    state.save_img(rgb_img, depth_img)
    for obj_state in object_states:
        name = obj_state["name"]
        pos = obj_state["position"]
        
        u, v = state.world_to_pixel(pos)
        pos_str = f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
        pixel_str = f"({u}, {v})"
        
        print(f"{name:<30} | {pos_str:<30} | {pixel_str:<15}")
        
    print("-" * 80)
    while scene.is_app_running():
        scene.step()
    scene.simulation_app.close()

if __name__ == "__main__":
    main()
