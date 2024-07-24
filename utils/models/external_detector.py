import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

class ColorFilterDetector():
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
            'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)]
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, img, task_instruction):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, num_cam, 3, img_h, img_w)
        '''
        denorm_img = img * self.norm_std + self.norm_mean # Left shape: (bs, num_cam, 3, img_h, img_w). Range: 0~1

        color_thre = []
        if self.cfg['TASK_NAME'] != 'isaac_multicolorbox':
            for ele in task_instruction:
                color_thre.append(self.filter_cfg[ele])
        else:
            raise NotImplementedError
        color_thre = torch.stack(color_thre, dim = 0)   # Left shape: (bs, 2, 3)
        color_min_thre = color_thre[:, 0][:, None, :, None, None] # Left shape: (bs, 1, 3, 1, 1)
        color_max_thre = color_thre[:, 1][:, None, :, None, None] # Left shape: (bs, 1, 3, 1, 1)

        obj_mask = (denorm_img >= color_min_thre) & (denorm_img <= color_max_thre) # Left shape: (bs, num_cam, 3, img_h, img_w)
        bs, num_cam, ch, img_h, img_w = obj_mask.shape
        obj_mask = obj_mask.view(bs * num_cam, ch, img_h, img_w) # Left shape: (bs * num_cam, 3, img_h, img_w)
        obj_mask = torch.all(obj_mask, dim = 1) # Left shape: (bs * num_cam, img_h, img_w)

        det_box = masks_to_boxes(obj_mask)  # Left shape: (bs * num_cam, 4)
        det_box[:, 0] = det_box[:, 0] / img_w
        det_box[:, 1] = det_box[:, 1] / img_h
        det_box[:, 2] = det_box[:, 2] / img_w
        det_box[:, 3] = det_box[:, 3] / img_h

        det_box = det_box.view(bs, num_cam, 4)

        '''vis_img = (denorm_img[0][1].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        vis_img = np.ascontiguousarray(vis_img)
        det_box = det_box[0][1].cpu().numpy()
        cv2.rectangle(vis_img, (int(det_box[0] * img_w), int(det_box[1] * img_h)), (int(det_box[2] * img_w), int(det_box[3] * img_h)), (255, 255, 0), 2)
        vis_mask = obj_mask.view(bs, num_cam, img_h, img_w)[0][1].cpu().numpy()
        vis_mask = np.concatenate((vis_mask[:, :, None].astype(np.uint8) * 255, np.zeros((vis_mask.shape[0], vis_mask.shape[1], 2), dtype = np.uint8)), axis = 2)
        vis = np.concatenate((vis_img, vis_mask), axis = 1)
        plt.imshow(vis)
        plt.savefig('vis.png')
        pdb.set_trace()'''

        return det_box

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