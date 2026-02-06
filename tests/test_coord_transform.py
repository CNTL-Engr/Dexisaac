import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from unittest.mock import MagicMock
sys.modules["omni"] = MagicMock()
sys.modules["omni.usd"] = MagicMock()
sys.modules["omni.kit.app"] = MagicMock()
sys.modules["pxr"] = MagicMock()
sys.modules["isaaclab.sensors"] = MagicMock()

sys.modules["camera"] = MagicMock()

from state import State

def test_transformation():
    original_init = State.__init__
    State.__init__ = lambda self, prim_path="/World/Camera", exclude_prim_paths=None: None
    
    state = State()
    
    state = State()

    center_w = (0.75, 0.0)
    expected_center_p = (160, 160)
    res_center = state.world_to_pixel(center_w)
    print(f"Center {center_w} -> Pixel {res_center}. Expected {expected_center_p}")
    assert res_center == expected_center_p
    
    res_center = state.world_to_pixel(center_w)
    print(f"Center {center_w} -> Pixel {res_center}. Expected {expected_center_p}")
    assert res_center == expected_center_p
    
    corner_w = (1.125, 0.375)
    expected_corner_p = (0, 0)
    res_corner = state.world_to_pixel(corner_w)
    print(f"Corner {corner_w} -> Pixel {res_corner}. Expected {expected_corner_p}")
    assert abs(res_corner[0] - expected_corner_p[0]) <= 1
    assert abs(res_corner[1] - expected_corner_p[1]) <= 1
    

    corner_min = (0.375, -0.375)
    expected_corner_min = (448, 448)
    res_min = state.world_to_pixel(corner_min)
    print(f"Corner Min {corner_min} -> Pixel {res_min}. Expected {expected_corner_min}")
    assert abs(res_min[0] - expected_corner_min[0]) <= 1
    assert abs(res_min[1] - expected_corner_min[1]) <= 1
    

    tensor_input = torch.tensor([0.75, 0.0])
    res_tensor = state.world_to_pixel(tensor_input)
    print(f"Tensor {tensor_input} -> Pixel {res_tensor}")
    assert res_tensor == expected_center_p

    print("All tests passed!")

if __name__ == "__main__":
    test_transformation()
