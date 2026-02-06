import numpy as np
import torch
import cv2
import sys
import os

# Ensure sam2 is in path if not installed globally (it is installed as 'SAM-2' package but imports as 'sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Wrapper:
    def __init__(self, 
                 checkpoint_path="/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac/sam_model/sam2.1_hiera_large.pt", 
                 config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                 device="cuda"):
        self.device = device
        
        # build_sam2 expects config file path relative to sam2 root or absolute. 
        # Since we have the absolute path, we can pass it directly or handle it.
        # Note: build_sam2 signature: (config_file, checkpoint, device, mode, apply_postprocessing, ...)
        
        print(f"Loading SAM2 model from {checkpoint_path} with config {config_file}...")
        self.sam_model = build_sam2(config_file, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam_model)

    def set_image(self, image):
        """
        [功能]: 设置图像
        [输入]: image: RGB image (H, W, 3) or (H, W) or BGR (if standardized)
                SAM2 expects RGB 0-255 uint8 or float 0-1.
        """
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        """
        [功能]: 内部预测包装器
        [输入]: point_coords: (N, 2)
                point_labels: (N,)
        [输出]: masks, scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
        # Sort by score to return best
        # Note: masks shape is (N, H, W) usually for single prompt.
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            return masks[best_idx : best_idx + 1], scores[best_idx : best_idx + 1], logits[best_idx : best_idx + 1]
        else:
            return masks, scores, logits

    def SAMpredict(self, image, point_coords):
        """
        [功能]: 简单的预测接口
        [输入]: image: RGB image as numpy array.
                point_coords: List of points or numpy array [[x, y], ...].
        [输出]: binary_mask: uint8 numpy array (0 for background, 255 for foreground).
        """
        self.set_image(image)
        
        point_coords = np.array(point_coords)
        point_labels = np.ones(point_coords.shape[0]) # Assume all points are foreground (1)

        # Use multimask_output=True to get candidate masks, then select best in predict()
        masks, _, _ = self.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        
        # masks is shape (1, H, W) boolean/float
        # SAM2 masks might be bool or float. safe cast.
        if masks.dtype != np.bool_:
            masks = masks > 0.0 # Threshold if logits
            
        binary_mask = (masks[0] * 255).astype(np.uint8)
        return binary_mask

if __name__ == "__main__":
    # Simple test logic
    print("Testing SAM2Wrapper...")
    try:
        wrapper = SAM2Wrapper()
        print("SAM2 Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize SAM2: {e}")
