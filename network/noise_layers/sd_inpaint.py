import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import random
from . import *
from diffusers import StableDiffusionInpaintPipeline

class SDInpaint(nn.Module):
    """
    Stable Diffusion Inpainting noise layer.
    """
    
    def __init__(self, pipe=None, mask_path="/workspace/SepMark/dataset/128_bits_5_images_mask", image_size=128):
        """
        Initialize SD Inpainting layer.
        
        Args:
            pipe: Stable diffusion inpainting pipeline
            mask_path (str): Path to mask images directory
            image_size (int): Image size for resizing masks
            image_id (int): Image ID for mask selection
        """
        super(SDInpaint, self).__init__()
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            ).to("cuda")
        self.mask_path = mask_path
        self.image_size = image_size
        
    def forward(self, image_cover_mask, image_id):
        """
        Apply SD inpainting to the input image.
        
        Args:
            image_cover_mask (list): [encoded_image, cover_image, mask]
        
        Returns:
            torch.Tensor: Inpainted image
        """
        print("SDInpaint for image_id:", image_id)
        # Extract the encoded image (first element)
        y_forw = image_cover_mask[0]
        
        # Empty prompt for inpainting
        prompt = ""
        
        # Get batch size
        b, _, _, _ = y_forw.shape
        
        # Convert from [-1, 1] to [0, 1] range and prepare for PIL
        image_batch = ((y_forw + 1) / 2).permute(0, 2, 3, 1).detach().cpu().numpy()
        forw_list = []
        
        for j in range(b):
            # Use self.image_id instead of global variables
            i = image_id + 1
            image_size = self.image_size
            
            # Load mask image
            mask_path = self.mask_path + str(i).zfill(4) + ".png"
            try:
                mask_image = Image.open(mask_path).convert("L")
            except:
                # If specific mask doesn't exist, create a random mask or use a default
                mask_image = Image.new("L", (image_size, image_size), 128)
            
            mask_image = mask_image.resize((image_size, image_size))
            h, w = mask_image.size
            
            # Prepare input image
            image = image_batch[j, :, :, :]
            image_init = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")
            
            image_inpaint = self.pipe(
                prompt=prompt, 
                image=image_init, 
                mask_image=mask_image, 
                height=w, 
                width=h
            ).images[0]
            
            # Convert back to numpy
            image_inpaint = np.array(image_inpaint) / 255.
            mask_image = np.array(mask_image)
            mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
            mask_image = mask_image.astype(np.float32)
            
            # Fuse original and inpainted images using mask
            image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
            forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
        
        # Stack batch and convert back to [-1, 1] range
        y_forw = torch.stack(forw_list, dim=0).float().cuda()
        y_forw = y_forw * 2 - 1
        
        return y_forw


class SDInpaintConfigurable(nn.Module):
    """
    Configurable Stable Diffusion Inpainting noise layer with more options.
    """
    
    def __init__(self, pipe=None, mask_path="../dataset/valAGE-Set-Mask/", 
                 image_size=512, prompt="", strength=1.0, guidance_scale=7.5, image_id=0):
        """
        Initialize configurable SD Inpainting layer.
        
        Args:
            pipe: Stable diffusion inpainting pipeline
            mask_path (str): Path to mask images directory
            image_size (int): Image size for resizing masks
            prompt (str): Text prompt for inpainting
            strength (float): Inpainting strength
            guidance_scale (float): Guidance scale for generation
            image_id (int): Image ID for mask selection
        """
        super(SDInpaintConfigurable, self).__init__()
        self.pipe = pipe
        self.mask_path = mask_path
        self.image_size = image_size
        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.image_id = image_id
        
    def forward(self, image_cover_mask):
        """
        Apply configurable SD inpainting to the input image.
        
        Args:
            image_cover_mask (list): [encoded_image, cover_image, mask]
        
        Returns:
            torch.Tensor: Inpainted image
        """
        # Extract the encoded image (first element)
        y_forw = image_cover_mask[0]
        
        # Get batch size
        b, _, _, _ = y_forw.shape
        
        # Convert from [-1, 1] to [0, 1] range and prepare for PIL
        image_batch = ((y_forw + 1) / 2).permute(0, 2, 3, 1).detach().cpu().numpy()
        forw_list = []
        
        for j in range(b):
            # Use self.image_id instead of sequential numbering
            i = self.image_id + 1
            mask_path = self.mask_path + str(i).zfill(4) + ".png"
            
            try:
                mask_image = Image.open(mask_path).convert("L")
            except:
                # Create a random circular mask as fallback
                mask_image = Image.new("L", (self.image_size, self.image_size), 0)
                # You could add more sophisticated mask generation here
            
            mask_image = mask_image.resize((self.image_size, self.image_size))
            h, w = mask_image.size
            
            # Prepare input image
            image = image_batch[j, :, :, :]
            image_init = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")
            
            # Apply inpainting with configurable parameters
            if self.pipe is not None:
                image_inpaint = self.pipe(
                    prompt=self.prompt,
                    image=image_init,
                    mask_image=mask_image,
                    height=h,
                    width=w,
                    strength=self.strength,
                    guidance_scale=self.guidance_scale
                ).images[0]
            else:
                # Fallback if no pipeline available
                image_inpaint = image_init
            
            # Convert back to numpy and fuse
            image_inpaint = np.array(image_inpaint) / 255.
            mask_image = np.array(mask_image)
            mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
            mask_image = mask_image.astype(np.float32)
            
            # Fuse original and inpainted images using mask
            image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
            forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
        
        # Stack batch and convert back to [-1, 1] range
        y_forw = torch.stack(forw_list, dim=0).float().cuda()
        y_forw = y_forw * 2 - 1
        
        return y_forw