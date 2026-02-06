import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMWrapper:
    def __init__(self, checkpoint_path="/home/wyq/xc/equi/IsaacLab/scripts/workspace/sam_model/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
        best_idx = np.argmax(scores)
        return masks[best_idx : best_idx + 1], scores[best_idx : best_idx + 1], logits[best_idx : best_idx + 1]

    def SAMpredict(self, image, point_coords):
        """
        Simple interface for prediction.
        Args:
            image: RGB image as numpy array.
            point_coords: List of points or numpy array [[x, y], ...].
        Returns:
            binary_mask: uint8 numpy array (0 for background, 255 for foreground).
        """
        self.set_image(image)
        
        point_coords = np.array(point_coords)
        point_labels = np.ones(point_coords.shape[0]) # Assume all points are foreground (1)

        masks, _, _ = self.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        
        # masks is shape (1, H, W) boolean
        binary_mask = (masks[0] * 255).astype(np.uint8)
        return binary_mask





'''
    test code 
'''
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

if __name__ == "__main__":
    # Load image
    image = cv2.imread('/home/wyq/xc/equi/IsaacLab/scripts/workspace/src/image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Wrapper (defaults)
    sam_wrapper = SAMWrapper()

    # Define input points
    input_point = [[250, 187]] # List format allowed
    
    # Predict using simplified interface
    mask = sam_wrapper.SAMpredict(image, input_point)
    
    # Visualize
    plt.figure(figsize=(10,10))
    plt.imshow(mask, cmap='gray') # Display the binary mask directly
    
    input_label = np.array([1])
    # show_points(np.array(input_point), input_label, plt.gca())
    
    plt.title("Binary Mask Result (Black/White)", fontsize=18)
    plt.axis('off')
    plt.show()
