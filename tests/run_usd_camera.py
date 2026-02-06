# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

It has been modified to load custom meshes and specific camera settings.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p scripts/workspace/tests/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p scripts/workspace/tests/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import quat_from_euler_xyz

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # Position: 0.75, 0, 1. Orientation: 0, 0, 90 (degrees)
    # Note: quat_from_euler_xyz takes radians.
    
    # Create Xform for the camera to attach to
    prim_utils.create_prim("/World/CameraXform", "Xform")
    
    # Convert orientation
    rot_euler = torch.tensor([180.0 * np.pi / 180.0, 0.0, -90.0 * np.pi / 180.0])
    rot_quat = quat_from_euler_xyz(rot_euler[0], rot_euler[1], rot_euler[2])
    
    # In Isaac Lab, offset in CameraCfg is relative to the prim it spawns on OR the parent prim.
    # Here we can just set the offset directly relative to origin if we don't spawn it on a robot.
    # However, CameraCfg requires a prim_path.
    
    camera_cfg = CameraCfg(
        prim_path="/World/CameraXform/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # Set pose directly via offset here since parent is at origin
        offset=CameraCfg.OffsetCfg(
            pos=(0.75, 0.0, 1.0),
            rot=rot_quat.tolist() # (w, x, y, z)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene with clutter environment (from scene.py logic)."""
    # Create a dictionary for the scene entities
    scene_entities = {}
    # Meshdata paths
    mesh_root = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/meshdata"
    mesh_target_root = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/meshdata_target"
    
    # Get available models
    def get_models(root):
        if os.path.exists(root):
            return [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "textured.usd"))]
        return []
    
    available_models = get_models(mesh_root)
    available_target_models = get_models(mesh_target_root)
    
    if not available_models or not available_target_models:
        print("Warning: No models found!")
        return scene_entities
    
    # Number of objects to spawn
    num_objects = random.randint(3, 5)
    
    # Spawn target object
    target_model_name = random.choice(available_target_models)
    target_usd_path = os.path.join(mesh_target_root, target_model_name, "textured.usd")
    
    # Target position and rotation
    target_pos = [random.uniform(0.7, 0.8), random.uniform(-0.05, 0.05), random.uniform(0.1, 0.15)]
    target_euler = torch.tensor([random.uniform(-np.pi, np.pi) for _ in range(3)])
    target_quat = quat_from_euler_xyz(target_euler[0], target_euler[1], target_euler[2]).tolist()
    
    # Create target object
    target_cfg = RigidObjectCfg(
        prim_path=f"/World/Objects/Target_{target_model_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=target_usd_path,
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            semantic_tags=[("class", target_model_name)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=target_pos,
            rot=target_quat
        ),
    )
    scene_entities["target"] = RigidObject(cfg=target_cfg)
    
    # Spawn obstacle objects
    for i in range(num_objects):
        model_name = random.choice(available_models)
        usd_path = os.path.join(mesh_root, model_name, "textured.usd")
        
        # Random position in workspace
        position = [
            random.uniform(0.55, 0.95),
            random.uniform(-0.2, 0.2),
            random.uniform(0.05, 0.15)
        ]
        
        # Random rotation
        rot_euler = torch.tensor([random.uniform(-np.pi, np.pi) for _ in range(3)])
        rot_quat = quat_from_euler_xyz(rot_euler[0], rot_euler[1], rot_euler[2]).tolist()
        
        # Create object
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}_{model_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                semantic_tags=[("class", model_name)],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position,
                rot=rot_quat
            ),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)
    
    print(f"[INFO] Created {len(scene_entities)} objects in the scene")

    # Sensors - create camera after scene is fully set up
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera_test")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Simulate physics
    frame_count = 0
    camera.reset() # Initialize camera buffers
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())
       
        # Print info periodically
        if frame_count % 20 == 0:
            print(f"Frame {frame_count}")
            # print(camera) # avoid spamming if camera str is long

        # Extract camera data
        if args_cli.save and frame_count > 10: # Wait for a few frames to settle
            print("Saving images...")
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)
            
            # Run for a bit then exit
            if frame_count > 30:
                break

        # Draw pointcloud if there is a GUI and --draw has been passed
        if sim.has_gui() and args_cli.draw and "distance_to_image_plane" in camera.data.output.keys():
            # Derive pointcloud from camera at camera_index
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                depth=camera.data.output["distance_to_image_plane"][camera_index],
                position=camera.data.pos_w[camera_index],
                orientation=camera.data.quat_w_ros[camera_index],
                device=sim.device,
            )

            # In the first few steps, things are still being instanced and Camera.data
            # can be empty. If we attempt to visualize an empty pointcloud it will crash
            # the sim, so we check that the pointcloud is not empty.
            if pointcloud.size()[0] > 0:
                pc_markers.visualize(translations=pointcloud)
        
        frame_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera (viewer)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
