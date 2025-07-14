import torch
import torch.nn as nn

from .JPEG_utils import diff_round, quality_to_factor, Quantization
from .compression import compress_jpeg
from .decompression import decompress_jpeg


class DiffJPEG(nn.Module):    
    def __init__(self, differentiable=True, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
            # rounding = Quantization()
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor)
        self.quality = quality

    def forward(self, image_cover_mask):
        '''
        Apply JPEG compression to the input image.
        Compatible with Random_Noise interface.
        
        Args:
            image_cover_mask (list): [encoded_image, cover_image, mask]
                                   Following the Random_Noise interface
        
        Returns:
            torch.Tensor: JPEG compressed image
        '''
        # Extract the encoded image (first element)
        x = image_cover_mask[0]
        
        # Ensure compress and decompress modules are on the same device as input
        device = x.device
        self.compress = self.compress.to(device)
        self.decompress = self.decompress.to(device)
        
        # Convert from [-1, 1] to [0, 1] range for JPEG processing
        x_normalized = (x + 1) / 2
        
        # Get original dimensions
        org_height = x_normalized.shape[2]
        org_width = x_normalized.shape[3]
        
        # Apply JPEG compression
        y, cb, cr = self.compress(x_normalized)
        
        # Apply JPEG decompression
        recovered = self.decompress(y, cb, cr, org_height, org_width)
        
        # Convert back to [-1, 1] range
        recovered = recovered * 2 - 1
        
        # Clamp to ensure values are in valid range
        recovered = torch.clamp(recovered, -1, 1)
        
        return recovered
    
    def to(self, device):
        """Override to method to ensure all submodules are moved to device."""
        super().to(device)
        self.compress = self.compress.to(device)
        self.decompress = self.decompress.to(device)
        return self
    
    def cuda(self, device=None):
        """Override cuda method to ensure all submodules are moved to GPU."""
        super().cuda(device)
        self.compress = self.compress.cuda(device)
        self.decompress = self.decompress.cuda(device)
        return self
    
    def cpu(self):
        """Override cpu method to ensure all submodules are moved to CPU."""
        super().cpu()
        self.compress = self.compress.cpu()
        self.decompress = self.decompress.cpu()
        return self
    
    def __repr__(self):
        return f"DiffJPEG(differentiable=True, quality={self.quality})"


# Convenience function for creating JPEG with specific quality
def create_diff_jpeg(quality=75, differentiable=True):
    """
    Create DiffJPEG with specified quality.
    
    Args:
        quality (int): JPEG quality (1-100)
        differentiable (bool): Whether to use differentiable version
        
    Returns:
        DiffJPEG: Configured JPEG compression layer
    """
    return DiffJPEG(differentiable=differentiable, quality=quality)