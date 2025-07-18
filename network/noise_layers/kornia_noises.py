import torch.nn as nn
import kornia
import numpy as np
import torch
# Kornia based noises

class GaussianNoiseEditGuard(nn.Module):
    """
    Gaussian noise layer adapted from EditGuard implementation.
    """
    
    def __init__(self, noisesigma=25.5):
        """
        Initialize Gaussian noise layer.
        
        Args:
            noisesigma (float): Noise sigma parameter (default: 25.5)
        """
        super(GaussianNoiseEditGuard, self).__init__()
        self.noisesigma = noisesigma
        
    def forward(self, image_cover_mask):
        """
        Apply Gaussian noise to the input image.
        
        Args:
            image_cover_mask (list): [encoded_image, cover_image, mask]
        
        Returns:
            torch.Tensor: Noisy image
        """
        # Extract the encoded image (first element)
        y_forw = image_cover_mask[0]
        
        # Calculate noise level
        NL = self.noisesigma / 255.0
        
        # Generate Gaussian noise
        noise = np.random.normal(0, NL, y_forw.shape)
        
        # Convert to torch tensor and move to GPU
        torchnoise = torch.from_numpy(noise).cuda().float()
        
        # Add noise to the image
        y_forw = y_forw + torchnoise
        
        return y_forw



# intensity
class GaussianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3), sigma=(2,2), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image)


class GaussianNoise(nn.Module):

    def __init__(self, mean=0, std=0.1, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, image_cover_mask):
        image, mask = image_cover_mask[0], image_cover_mask[2]
        #mask = mask[:, 0: 3, :, :]
        return self.transform(image) #image * mask + self.transform(image) * (1 - mask)


class MedianBlur(nn.Module):

    def __init__(self, kernel_size=(3,3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image)


class Brightness(nn.Module):

    def __init__(self, brightness=0.5, p=1):
        super(Brightness, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(brightness=brightness, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1 ) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Contrast(nn.Module):

    def __init__(self, contrast=0.5, p=1):
        super(Contrast, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(contrast=contrast, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Saturation(nn.Module):

    def __init__(self, saturation=0.5, p=1):
        super(Saturation, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(saturation=saturation, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Hue(nn.Module):

    def __init__(self, hue=0.1, p=1):
        super(Hue, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(hue=hue, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


# geometric
class Rotation(nn.Module):

    def __init__(self, degrees=180, p=1):
        super(Rotation, self).__init__()
        self.transform = kornia.augmentation.RandomRotation(degrees=degrees, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image)


class Affine(nn.Module):

    def __init__(self, degrees=0, translate=0.1, scale=[0.7,0.7], shear=30, p=1):
        super(Affine, self).__init__()
        self.transform = kornia.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, p=p)

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        return self.transform(image)

