import pdb
import torch
from torchvision.transforms import functional as F

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data, qpos_data, action_data, is_pad):
        for t in self.transforms:
            image_data, qpos_data, action_data, is_pad = t(image_data, qpos_data, action_data, is_pad)
        return image_data, qpos_data, action_data, is_pad

class ToTensor():
    def __call__(self, image_data, qpos_data, action_data, is_pad):
        return F.to_tensor(image_data), F.to_tensor(qpos_data), F.to_tensor(action_data), F.to_tensor(action_data)

class Normalize():
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image_data, qpos_data, action_data, is_pad):
        image_data = F.normalize(image_data, mean=self.mean, std=self.std)

        if self.to_bgr:
            image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data, qpos_data, action_data, is_pad
