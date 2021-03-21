import math
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class Resize:
    """Resize the input image to the given size."""

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        self.size = size

    def __call__(self, sample):
        t = transforms.Resize(self.size)
        return {'original_image': t(sample['original_image']),
                'downsampled_image': t(sample['downsampled_image']),
                'label': sample['label']}


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            return {'original_image': F.hflip(sample['original_image']),
                    'downsampled_image': F.hflip(sample['downsampled_image']),
                    'label': sample['label']}
        return sample


class RandomCrop:
    """Crop the given image at a random location."""

    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding

    def __call__(self, sample):
        original_image = sample['original_image']
        downsampled_image = sample['downsampled_image']

        if self.padding is not None:
            original_image = F.pad(original_image, self.padding)
            downsampled_image = F.pad(downsampled_image, self.padding)

        i, j, h, w = transforms.RandomCrop.get_params(original_image, self.size)

        return {'original_image': F.crop(original_image, i, j, h, w),
                'downsampled_image': F.crop(downsampled_image, i, j, h, w),
                'label': sample['label']}


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""

    def __call__(self, sample):
        return {'original_image': F.to_tensor(sample['original_image']),
                'downsampled_image': F.to_tensor(sample['downsampled_image']),
                'label': sample['label']}


class Normalize:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        t = transforms.Normalize(self.mean, self.std, self.inplace)
        return {'original_image': t(sample['original_image']),
                'downsampled_image': t(sample['downsampled_image']),
                'label': sample['label']}


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        if random.uniform(0, 1) > self.probability:
            return sample

        attempts = 100
        for _ in range(attempts):
            original_image = sample['original_image']
            downsampled_image = sample['downsampled_image']
            channel, height, width = original_image.size()
            target_area = random.uniform(self.sl, self.sh) * height * width
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)
                if channel == 3:
                    original_image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    original_image[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    original_image[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    downsampled_image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    downsampled_image[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    downsampled_image[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    original_image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    downsampled_image[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return sample

        return sample
