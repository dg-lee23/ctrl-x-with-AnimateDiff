import torch
import torch.nn.functional as F
import kornia

def shift_tensor(tensor, shift_dir='left'):
    '''
    expects input shape (hw, x, y, z)
        the axes x, y, z are irrelevant and may be anything
    '''
    # setup
    hw, n_head, f1, f2 = tensor.shape
    res = h = w = int(hw**0.5)

    if shift_dir == 'left': shift = (0, w//4)
    elif shift_dir == 'right': shift = (0, -w//4)
    elif shift_dir == 'up': shift = (h//4, 0)
    elif shift_dir == 'down': shift = (-h//4, 0)
    shift_x, shift_y = shift

    # no circular shifting; creates 'blank' pixels instead
    tensor = tensor.view(res, res, n_head, f1, f2)
    if shift_x != 0:
        if shift_x > 0:
            tensor = torch.cat((tensor[shift_x:, :, :, :, :], torch.zeros_like(tensor[:shift_x, :, :, :, :])), dim=0)
        else:
            shift_x = -shift_x
            tensor = torch.cat((torch.zeros_like(tensor[-shift_x:, :, :, :, :]), tensor[:-shift_x, :, :, :, :]), dim=0)

    if shift_y != 0:
        if shift_y > 0:
            tensor = torch.cat((tensor[:, shift_y:, :, :, :], torch.zeros_like(tensor[:, :shift_y, :, :, :])), dim=1)
        else:
            shift_y = -shift_y
            tensor = torch.cat((torch.zeros_like(tensor[:, -shift_y:, :, :, :]), tensor[:, :-shift_y, :, :, :]), dim=1)

    # Reshape back 
    shifted_tensor = tensor.view(hw, n_head, f1, f2)
    
    return shifted_tensor


def pad_tensor(tensor, target_size, mode='center'):

    # ? check expected input shape: (B, C, H, W)
    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    if mode=='center':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode=='top-left':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = 0
        pad_right = target_W - W
    
    elif mode=='top-right':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = target_W - W
        pad_right = 0

    elif mode=='bottom-left':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'bottom-right':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-left':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'center-right':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-top':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode == 'center-bottom':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)

    return F.pad(tensor, padding, mode='constant', value=0)


def crop_tensor(tensor, target_size, mode='center'):

    # Extract current and target dimensions
    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    # Compute cropping coordinates based on mode
    if mode == 'center':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'top-left':
        crop_top = 0
        crop_bottom = target_H
        crop_left = 0
        crop_right = target_W
    
    elif mode == 'top-right':
        crop_top = 0
        crop_bottom = target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'bottom-left':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = 0
        crop_right = target_W

    elif mode == 'bottom-right':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-left':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = 0
        crop_right = target_W

    elif mode == 'center-right':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-top':
        crop_top = 0
        crop_bottom = target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'center-bottom':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    # Perform the cropping operation
    cropped_tensor = tensor[:, :, crop_top:crop_bottom, crop_left:crop_right]

    return cropped_tensor


def perspective_warp_tensor(tensor, warp_dir, warp_scale=1):
    
    # ? expects shape (f, c, h, w)
    assert len(tensor.shape) == 4, "tensor shape must be (f,c,h,w)"

    # adjust tensor shape
    f, dim, h, w = tensor.shape
    tensor = tensor.contiguous().view(f * dim, 1, h, w)

    d = int(h * 0.1)
    if warp_dir == 'right':
        dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
        src_pts = [(0, 0), (h-1, d), (h-1, h-1-d), (0, h-1)]
    elif warp_dir == 'left':
        dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
        src_pts = [(0, d), (h-1, 0), (h-1, h-1), (0, h-1 - d)]
    elif warp_dir == 'down':
        dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
        src_pts = [(0, 0), (h-1, 0), (h-1-d, h-1-d), (d, h-1 - d)]

    dst_pts = torch.tensor(dst_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)
    src_pts = torch.tensor(src_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)
    H = kornia.geometry.get_perspective_transform(src_pts, dst_pts).float()

    # apply H
    H_batch = H.repeat(f * dim, 1, 1)  
    for _ in range(int(warp_scale)):
        tensor = kornia.geometry.warp_perspective(tensor.float(), H_batch, dsize=(h,w), mode='bicubic', align_corners=True)

    # reshape back
    tensor = tensor.view(f, dim, h, w).to(torch.float16)

    return tensor