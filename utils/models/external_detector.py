import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

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
            'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, img, task_instruction):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, num_cam, 3, img_h, img_w)
        '''
        bs, num_cam, ch, img_h, img_w = img.shape
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
        obj_mask = obj_mask.view(bs * num_cam, ch, img_h, img_w) # Left shape: (bs * num_cam, 3, img_h, img_w)
        obj_mask = torch.all(obj_mask, dim = 1) # Left shape: (bs * num_cam, img_h, img_w)

        det_box = masks_to_boxes(obj_mask)  # Left shape: (bs * num_cam, 4)
        rescale_det_box = rescale_box(det_box, (img.shape[4], img.shape[3]), scale_ratio = self.cfg['POLICY']['EXTERNAL_DET_SCALE_FACTOR']) # Left shape: (bs * num_cam, 4)
        idxs = torch.arange(rescale_det_box.shape[0]).to(rescale_det_box.device)[:, None]
        rescale_det_box = torch.cat([idxs, rescale_det_box], dim = 1) # Left shape: (bs * num_cam, 5)
        resize_scale = (self.cfg['DATA']['IMG_RESIZE_SHAPE'][1], self.cfg['DATA']['IMG_RESIZE_SHAPE'][0])
        view_img = img.view(bs * num_cam, ch, img_h, img_w)   # Left shape: (bs * num_cam, 3, img_h, img_w)
        sample_img = torchvision.ops.roi_align(input = view_img, boxes = rescale_det_box, output_size = resize_scale)   # Left shape: (bs * num_cam, ch, img_h, img_w)
        sample_img = sample_img.view(bs, num_cam, ch, img_h, img_w)

        # vis
        '''vis_img = ((sample_img * self.norm_std + self.norm_mean)[0, 0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        vis_img = np.ascontiguousarray(vis_img[:, :, ::-1])
        cv2.imwrite('vis.png', vis_img)'''

        det_box[:, 0] = det_box[:, 0] / img_w
        det_box[:, 1] = det_box[:, 1] / img_h
        det_box[:, 2] = det_box[:, 2] / img_w
        det_box[:, 3] = det_box[:, 3] / img_h

        det_box = det_box.view(bs, num_cam, 4)

        return det_box, sample_img

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

if __name__ == '__main__':
    video_path = '/home/cvte/twilight/code/act/datasets/isaac_singlecolorbox/exterior_camera1/episode_0.mp4'
    filter_cfg = {
        'red': [(0.2, -0.01, -0.01), (1.01, 0.1, 0.1)],
        'green': [(-0.01, 0.2, -0.01), (0.1, 1.01, 0.1)],
        'blue': [(-0.01, -0.01, 0.2), (0.1, 0.1, 1.2)],
        'purple': [(0.2, -0.01, 0.2), (1.01, 0.1, 1.01)],
        'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
    }
    color_thre = filter_cfg['yellow']

    read_cap = cv2.VideoCapture(video_path)
    write_cap = cv2.VideoWriter('vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 240))

    while True:
        ret, frame = read_cap.read()
        if not ret:
            break
        rgb_img = frame[:, :, ::-1]
        norm_img = torch.Tensor(rgb_img.astype(np.float32) / 255)
        color_min_thre = torch.Tensor(color_thre[0])[None][None]
        color_max_thre = torch.Tensor(color_thre[1])[None][None]
        obj_mask = (norm_img >= color_min_thre) & (norm_img <= color_max_thre) # Left shape: (img_h, img_w, 3)
        ch, img_h, img_w = obj_mask.shape
        obj_mask = torch.all(obj_mask, dim = 2) # Left shape: (img_h, img_w)
        det_box = masks_to_boxes(obj_mask[None])[0]  # Left shape: (4,)

        vis_img = frame
        cv2.rectangle(vis_img, (int(det_box[0]), int(det_box[1])), (int(det_box[2]), int(det_box[3])), (255, 255, 0), 1)
        vis_mask = obj_mask.numpy()
        vis_mask = np.concatenate((vis_mask[:, :, None].astype(np.uint8) * 255, np.zeros((vis_mask.shape[0], vis_mask.shape[1], 2), dtype = np.uint8)), axis = 2)
        vis = np.concatenate((vis_img, vis_mask), axis = 1)
        write_cap.write(vis)
        
    write_cap.release()
    write_cap.release()