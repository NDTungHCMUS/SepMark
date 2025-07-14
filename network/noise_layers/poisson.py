import torch
import torch.nn as nn
import random
from . import *

class PoissonNoise(nn.Module):
    """
    Poisson noise layer following the exact logic provided.
    """
    
    def __init__(self):
        super(PoissonNoise, self).__init__()
        
    def forward(self, image_cover_mask):
        """
        Apply Poisson noise to the input image.
        
        Args:
            image_cover_mask (list): [encoded_image, cover_image, mask]
        
        Returns:
            torch.Tensor: Noisy image with Poisson noise
        """
        # Extract the encoded image (first element)
        y_forw = image_cover_mask[0]
        
        # Convert from [-1, 1] to [0, 1] range for processing
        y_forw = (y_forw + 1) / 2
        
        # Apply Poisson noise logic
        vals = 10**4
        if random.random() < 0.5:
            noisy_img_tensor = torch.poisson(y_forw * vals) / vals
        else:
            img_gray_tensor = torch.mean(y_forw, dim=1, keepdim=True)
            noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
            noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

        y_forw = torch.clamp(noisy_img_tensor, 0, 1)
        
        # Convert back to [-1, 1] range
        y_forw = y_forw * 2 - 1
        
        return y_forw