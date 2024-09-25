import pdb
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def get_detector(cfg,):
    if cfg["POLICY"]["EXTERNAL_DET"] == "AllColorFilter":
        return AllColorFilterDetector(cfg)
    elif cfg["POLICY"]["EXTERNAL_DET"] == "SingleColorFilter":
        return SingleColorFilter(cfg)
    elif cfg["POLICY"]["EXTERNAL_DET"] == "LanguageMultiColorFilter":
        return LanguageMultiColorFilter(cfg)

class SingleColorFilter():
    def __init__(self, cfg):
        self.cfg = cfg
        norm_mean = cfg['DATA']['IMG_NORM_MEAN']
        norm_std = cfg['DATA']['IMG_NORM_STD']
        self.norm_mean = torch.Tensor(norm_mean)[None, None, :, None, None].cuda()
        self.norm_std = torch.Tensor(norm_std)[None, None, :, None, None].cuda()

        filter_cfg = {
            'red': [(0.2, -0.01, -0.01), (1.01, 0.1, 0.1)],
            'green': [(-0.01, 0.2, -0.01), (0.1, 1.01, 0.1)],
            'blue': [(-0.01, -0.01, 0.2), (0.1, 0.1, 1.2)],
            'purple': [(0.2, -0.01, 0.2), (1.01, 0.1, 1.01)],
            'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, img, task_instruction, status = None):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, num_cam, 3, img_h, img_w)
        '''

        bs, num_cam, ch, img_h, img_w = img.shape
        num_color = 1

        denorm_img = img * self.norm_std + self.norm_mean # Left shape: (bs, num_cam, 3, img_h, img_w). Range: 0~1
        denorm_img = denorm_img[:, :, None, :, :, :] # Left shape: (bs, num_cam, 1, 3, img_h, img_w)

        color_thre = []
        for ele in task_instruction:
            color_thre.append(self.filter_cfg[ele])
        color_thre = torch.stack(color_thre, dim = 0)   # Left shape: (bs, 2, 3)
        color_thre_min = color_thre[:, 0][:, None, None, :, None, None] # Left shape: (bs, 1, 1, 3, 1, 1)
        color_thre_max = color_thre[:, 1][:, None, None, :, None, None] # Left shape: (bs, 1, 1, 3, 1, 1)

        obj_mask = (denorm_img >= color_thre_min) & (denorm_img <= color_thre_max) # Left shape: (bs, num_cam, num_color, 3, img_h, img_w)
        obj_mask = obj_mask.view(bs * num_cam * num_color, ch, img_h, img_w) # Left shape: (bs * num_cam * num_color, 3, img_h, img_w)
        obj_mask = torch.all(obj_mask, dim = 1) # Left shape: (bs * num_cam * num_color, img_h, img_w)

        det_box = masks_to_boxes(obj_mask)  # Left shape: (bs * num_cam * num_color, 4)
        rescale_det_box = rescale_box(det_box, (img.shape[4], img.shape[3]), scale_ratio = self.cfg['POLICY']['EXTERNAL_DET_SCALE_FACTOR']) # Left shape: (bs * num_cam * num_color, 4)
        idxs = torch.arange(bs * num_cam).to(rescale_det_box.device)[:, None, None].expand(-1, num_color, -1).reshape(bs * num_cam * num_color, 1)   # Left shape: (bs * num_cam * num_color, 1)
        rescale_det_box = torch.cat([idxs, rescale_det_box], dim = 1) # Left shape: (bs * num_cam * num_color, 5)
        resize_scale = (self.cfg['DATA']['ROI_RESIZE_SHAPE'][1], self.cfg['DATA']['ROI_RESIZE_SHAPE'][0])
        view_img = img.view(bs * num_cam, ch, img_h, img_w)   # Left shape: (bs * num_cam, 3, img_h, img_w)
        sample_img = torchvision.ops.roi_align(input = view_img, boxes = rescale_det_box, output_size = resize_scale)   # Left shape: (bs * num_cam * num_color, ch, img_h, img_w)
        sample_img = sample_img.view(bs, num_cam, num_color, ch, sample_img.shape[-2], sample_img.shape[-1])    # Left shape: (bs, num_cam, num_color, ch, img_h, img_w)

        det_box[:, 0] = det_box[:, 0] / img_w
        det_box[:, 1] = det_box[:, 1] / img_h
        det_box[:, 2] = det_box[:, 2] / img_w
        det_box[:, 3] = det_box[:, 3] / img_h
        det_box = det_box.view(bs, num_cam, num_color, 4)
        
        return det_box, sample_img
    
class LanguageMultiColorFilter():
    def __init__(self, cfg):
        self.cfg = cfg
        norm_mean = cfg['DATA']['IMG_NORM_MEAN']
        norm_std = cfg['DATA']['IMG_NORM_STD']
        self.norm_mean = torch.Tensor(norm_mean)[None, None, :, None, None].cuda()
        self.norm_std = torch.Tensor(norm_std)[None, None, :, None, None].cuda()

        filter_cfg = {
            'red': [(0.2, -0.01, -0.01), (1.01, 0.1, 0.1)],
            'green': [(-0.01, 0.2, -0.01), (0.1, 1.01, 0.1)],
            'blue': [(-0.01, -0.01, 0.2), (0.1, 0.1, 1.2)],
            'purple': [(0.2, -0.01, 0.2), (1.01, 0.1, 1.01)],
            'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, img, task_instruction, status = None):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, num_cam, 3, img_h, img_w)
        '''

        bs, num_cam, ch, img_h, img_w = img.shape
        num_color = 1

        denorm_img = img * self.norm_std + self.norm_mean # Left shape: (bs, num_cam, 3, img_h, img_w). Range: 0~1
        denorm_img = denorm_img[:, :, None, :, :, :] # Left shape: (bs, num_cam, 1, 3, img_h, img_w)

        color_thre = []
        for cnt, ele in enumerate(task_instruction):
            word_list = ele.split(' ')
            color1, color2 = word_list[3], word_list[11]
            this_status = status[cnt].item()
            if this_status == 0:
                color = color1
            elif this_status == 1:
                color = color2
            elif this_status == 2:
                color = color2
            else:
                raise Exception("Unsupported status: {}".format(this_status))
            color_thre.append(self.filter_cfg[color])
        color_thre = torch.stack(color_thre, dim = 0)   # Left shape: (bs, 2, 3)
        color_thre_min = color_thre[:, 0][:, None, None, :, None, None] # Left shape: (bs, 1, 1, 3, 1, 1)
        color_thre_max = color_thre[:, 1][:, None, None, :, None, None] # Left shape: (bs, 1, 1, 3, 1, 1)

        obj_mask = (denorm_img >= color_thre_min) & (denorm_img <= color_thre_max) # Left shape: (bs, num_cam, num_color, 3, img_h, img_w)
        obj_mask = obj_mask.view(bs * num_cam * num_color, ch, img_h, img_w) # Left shape: (bs * num_cam * num_color, 3, img_h, img_w)
        obj_mask = torch.all(obj_mask, dim = 1) # Left shape: (bs * num_cam * num_color, img_h, img_w)

        det_box = masks_to_boxes(obj_mask)  # Left shape: (bs * num_cam * num_color, 4)
        rescale_det_box = rescale_box(det_box, (img.shape[4], img.shape[3]), scale_ratio = self.cfg['POLICY']['EXTERNAL_DET_SCALE_FACTOR']) # Left shape: (bs * num_cam * num_color, 4)
        idxs = torch.arange(bs * num_cam).to(rescale_det_box.device)[:, None, None].expand(-1, num_color, -1).reshape(bs * num_cam * num_color, 1)   # Left shape: (bs * num_cam * num_color, 1)
        rescale_det_box = torch.cat([idxs, rescale_det_box], dim = 1) # Left shape: (bs * num_cam * num_color, 5)
        resize_scale = (self.cfg['DATA']['ROI_RESIZE_SHAPE'][1], self.cfg['DATA']['ROI_RESIZE_SHAPE'][0])
        view_img = img.view(bs * num_cam, ch, img_h, img_w)   # Left shape: (bs * num_cam, 3, img_h, img_w)
        sample_img = torchvision.ops.roi_align(input = view_img, boxes = rescale_det_box, output_size = resize_scale)   # Left shape: (bs * num_cam * num_color, ch, img_h, img_w)
        sample_img = sample_img.view(bs, num_cam, num_color, ch, sample_img.shape[-2], sample_img.shape[-1])    # Left shape: (bs, num_cam, num_color, ch, img_h, img_w)

        det_box[:, 0] = det_box[:, 0] / img_w
        det_box[:, 1] = det_box[:, 1] / img_h
        det_box[:, 2] = det_box[:, 2] / img_w
        det_box[:, 3] = det_box[:, 3] / img_h
        det_box = det_box.view(bs, num_cam, num_color, 4)
        
        return det_box, sample_img

class AllColorFilterDetector():
    def __init__(self, cfg):
        self.cfg = cfg
        norm_mean = cfg['DATA']['IMG_NORM_MEAN']
        norm_std = cfg['DATA']['IMG_NORM_STD']
        self.norm_mean = torch.Tensor(norm_mean)[None, None, :, None, None].cuda()
        self.norm_std = torch.Tensor(norm_std)[None, None, :, None, None].cuda()

        filter_cfg = {
            'red': [(0.2, -0.01, -0.01), (1.01, 0.1, 0.1)],
            'green': [(-0.01, 0.2, -0.01), (0.1, 1.01, 0.1)],
            'blue': [(-0.01, -0.01, 0.2), (0.1, 0.1, 1.2)],
            'purple': [(0.2, -0.01, 0.2), (1.01, 0.1, 1.01)],
            'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, img, task_instruction, status = None):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, num_cam, 3, img_h, img_w)
        '''
        bs, num_cam, ch, img_h, img_w = img.shape
        num_color = len(self.filter_cfg)
        denorm_img = img * self.norm_std + self.norm_mean # Left shape: (bs, num_cam, 3, img_h, img_w). Range: 0~1
        denorm_img = denorm_img[:, :, None, :, :, :].expand(-1, -1, num_color, -1, -1, -1) # Left shape: (bs, num_cam, num_color, 3, img_h, img_w)

        color_thre_min = torch.stack([ele[0] for ele in self.filter_cfg.values()], dim = 0) # Left shape: (num_color, 3)
        color_thre_min = color_thre_min[None, None, :, :, None, None] # Left shape: (1, 1, num_color 3, 1, 1)
        color_thre_max = torch.stack([ele[1] for ele in self.filter_cfg.values()], dim = 0) # Left shape: (num_color, 3)
        color_thre_max = color_thre_max[None, None, :, :, None, None] # Left shape: (1, 1, num_color, 3, 1, 1)

        obj_mask = (denorm_img >= color_thre_min) & (denorm_img <= color_thre_max) # Left shape: (bs, num_cam, num_color, 3, img_h, img_w)
        obj_mask = obj_mask.view(bs * num_cam * num_color, ch, img_h, img_w) # Left shape: (bs * num_cam * num_color, 3, img_h, img_w)
        obj_mask = torch.all(obj_mask, dim = 1) # Left shape: (bs * num_cam * num_color, img_h, img_w)

        det_box = masks_to_boxes(obj_mask)  # Left shape: (bs * num_cam * num_color, 4)
        rescale_det_box = rescale_box(det_box, (img.shape[4], img.shape[3]), scale_ratio = self.cfg['POLICY']['EXTERNAL_DET_SCALE_FACTOR']) # Left shape: (bs * num_cam * num_color, 4)
        idxs = torch.arange(bs * num_cam).to(rescale_det_box.device)[:, None, None].expand(-1, num_color, -1).reshape(bs * num_cam * num_color, 1)   # Left shape: (bs * num_cam * num_color, 1)
        rescale_det_box = torch.cat([idxs, rescale_det_box], dim = 1) # Left shape: (bs * num_cam * num_color, 5)
        resize_scale = (self.cfg['DATA']['ROI_RESIZE_SHAPE'][1], self.cfg['DATA']['ROI_RESIZE_SHAPE'][0])
        view_img = img.view(bs * num_cam, ch, img_h, img_w)   # Left shape: (bs * num_cam, 3, img_h, img_w)
        sample_img = torchvision.ops.roi_align(input = view_img, boxes = rescale_det_box, output_size = resize_scale)   # Left shape: (bs * num_cam * num_color, ch, img_h, img_w)
        sample_img = sample_img.view(bs, num_cam, num_color, ch, sample_img.shape[-2], sample_img.shape[-1])    # Left shape: (bs, num_cam, num_color, ch, img_h, img_w)
        sample_img_mask = torch.zeros((bs, num_cam, num_color, sample_img.shape[-2] // 14, sample_img.shape[-1] // 14), dtype = torch.bool, device = sample_img.device) # Left shape: (bs, num_cam, num_color, img_h // 14, img_w // 14)

        det_box[:, 0] = det_box[:, 0] / img_w
        det_box[:, 1] = det_box[:, 1] / img_h
        det_box[:, 2] = det_box[:, 2] / img_w
        det_box[:, 3] = det_box[:, 3] / img_h
        det_box = det_box.view(bs, num_cam, num_color, 4)

        return det_box, sample_img, sample_img_mask

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x) == 0:
            bounding_boxes[index, 0] = 0
            bounding_boxes[index, 1] = 0
            bounding_boxes[index, 2] = 0
            bounding_boxes[index, 3] = 0
        else:
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def rescale_box(det_box, img_shape, scale_ratio = 1.0):
    '''
    Input:
        det_box: torch tensor with the shape of (bs, 4)
        img_shape: (img_w, img_h)
    Output:
        det_box: torch tensor with the shape of (bs, 4)
    '''
    bs = det_box.shape[0]
    img_w, img_h = img_shape

    det_box = det_box.view(bs, 2, 2)
    det_box_center = det_box.mean(dim = 1)[:, None]   # Left shape: (bs, 1, 2)
    det_box_wh = (det_box[:, 1] - det_box[:, 0])[:, None] # Left shape: (bs, 1, 2)
    det_box = torch.cat((det_box_center - det_box_wh / 2 * scale_ratio, det_box_center + det_box_wh / 2 * scale_ratio), dim = 1)    # Left shape: (bs, 2, 2)

    det_box[:, :, 0] = torch.clamp(det_box[:, :, 0], min = 0, max = img_w - 1)
    det_box[:, :, 1] = torch.clamp(det_box[:, :, 1], min = 0, max = img_h - 1)
    return det_box.view(bs, 4)