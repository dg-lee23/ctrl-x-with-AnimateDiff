import math

import torch.nn.functional as F

from .utils import *
from .my_utils import *


def get_schedule(timesteps, schedule):
    end = round(len(timesteps) * schedule)
    timesteps = timesteps[:end]
    return timesteps


def get_elem(l, i, default=0.0):
    if i >= len(l):
        return default
    return l[i]


def pad_list(l_1, l_2, pad=0.0):
    max_len = max(len(l_1), len(l_2))
    l_1 = l_1 + [pad] * (max_len - len(l_1))
    l_2 = l_2 + [pad] * (max_len - len(l_2))
    return l_1, l_2
    
    
def normalize(x, dim):
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x_normalized = (x - x_mean) / x_std
    return x_normalized


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def appearance_mean_std(q_c_normed, k_s_normed, v_s):  # c: content, s: style
    q_c = q_c_normed  # q_c and k_s must be projected from normalized features
    k_s = k_s_normed
    mean = F.scaled_dot_product_attention(q_c, k_s, v_s)  # Use scaled_dot_product_attention for efficiency
    std = (F.scaled_dot_product_attention(q_c, k_s, v_s.square()) - mean.square()).relu().sqrt()
    
    return mean, std
    

# for resize
def feature_injection_(features, batch_order, scale=0.8):
    assert features.shape[0] % len(batch_order) == 0
    features_dict = batch_tensor_to_dict(features, batch_order)

    # * ==============
    shape = features_dict["structure_cond"].shape
    s = scale**0.5

    # print(shape)

    # (f, n_heads, hw, head_dim)
    if shape[2] != shape[3]:
        f, n_heads, hw, head_dim = features_dict["structure_cond"].shape
        h = w = int(hw**0.5)
        target = features_dict["structure_cond"].permute(0, 1, 3, 2).view(f, n_heads*head_dim, h*w)
        target = target.view(f, n_heads*head_dim, h, w)
        target = F.interpolate(target, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
        target = pad_tensor(target, (f, n_heads*head_dim, h, w), 'center')
        target = target.view(f, n_heads, head_dim, h, w).view(f, n_heads, head_dim, h*w)
        target = target.permute(0, 1, 3, 2)
        features_dict["cond"][target != 0] = target[target != 0]

    # (f, dim, h, w)
    else:
        f, dim, h, w  = features_dict["structure_cond"].shape
        target = features_dict["structure_cond"]
        target = F.interpolate(target, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
        target = pad_tensor(target, (f, dim, h, w), 'center')
        features_dict["cond"][target != 0] = target[target != 0]

    # ? original code
    # * ==============
    # features_dict["cond"] = features_dict["structure_cond"]
    # * ==============
    features = batch_dict_to_tensor(features_dict, batch_order)
    return features


# for shift
def feature_injection_(features, batch_order, shift_dir='left'):
    assert features.shape[0] % len(batch_order) == 0
    features_dict = batch_tensor_to_dict(features, batch_order)
    
    # ? shift feature 
    
    # * ==============
    shape = features_dict["structure_cond"].shape

    # (f, n_heads, hw, head_dim)
    if shape[2] != shape[3]:
        f, n_heads, hw, head_dim = features_dict["structure_cond"].shape
        target = features_dict["structure_cond"].permute(2, 1, 3, 0).view(hw, n_heads*head_dim, f, 1)
        target = shift_tensor(target, shift_dir).view(hw, n_heads, head_dim, f, 1).squeeze().permute(3, 1, 0, 2)
        features_dict["cond"][target != 0] = target[target != 0]

    # (f, dim, h, w)
    else:
        f, dim, h, w  = features_dict["structure_cond"].shape
        target = features_dict["structure_cond"].view(f, dim, h*w).permute(2, 1, 0)
        target = target.unsqueeze(dim=-1)
        target = shift_tensor(target, shift_dir).squeeze().permute(2, 1, 0).view(f, dim, h, w)
        features_dict["cond"][target != 0] = target[target != 0]
    # * ============== 

    # ? original code
    # * ==============
    # features_dict["cond"] = features_dict["structure_cond"]
    # * ==============

    features = batch_dict_to_tensor(features_dict, batch_order)
    return features


# for default
def feature_injection(features, batch_order):
    assert features.shape[0] % len(batch_order) == 0
    features_dict = batch_tensor_to_dict(features, batch_order)
    features_dict["cond"] = features_dict["structure_cond"]
    features = batch_dict_to_tensor(features_dict, batch_order)
    return features


# for perspective
def feature_injection_(features, batch_order, warp_dir='right'):
    assert features.shape[0] % len(batch_order) == 0
    features_dict = batch_tensor_to_dict(features, batch_order)

    # * ==============
    shape = features_dict["structure_cond"].shape

    # (f, n_heads, hw, head_dim)
    if shape[2] != shape[3]:
        f, n_heads, hw, head_dim = features_dict["structure_cond"].shape
        h = w = int(hw**0.5)
        target = features_dict["structure_cond"].permute(0, 1, 3, 2).view(f, n_heads*head_dim, h*w)
        target = target.view(f, n_heads*head_dim, h, w)

        target = perspective_warp_tensor(target, warp_dir)

        target = target.view(f, n_heads, head_dim, h, w).view(f, n_heads, head_dim, h*w)
        target = target.permute(0, 1, 3, 2)
        features_dict["cond"][target != 0] = target[target != 0]

    # (f, dim, h, w)
    else:
        f, dim, h, w  = features_dict["structure_cond"].shape
        target = features_dict["structure_cond"]
        target = perspective_warp_tensor(target, warp_dir)
        features_dict["cond"][target != 0] = target[target != 0]

    features = batch_dict_to_tensor(features_dict, batch_order)
    return features




def appearance_transfer(features, q_normed, k_normed, batch_order, v=None, reshape_fn=None):
    assert features.shape[0] % len(batch_order) == 0

    features_dict = batch_tensor_to_dict(features, batch_order)
    q_normed_dict = batch_tensor_to_dict(q_normed, batch_order)
    k_normed_dict = batch_tensor_to_dict(k_normed, batch_order)
    v_dict = features_dict
    if v is not None:
        v_dict = batch_tensor_to_dict(v, batch_order)
    
    mean_cond, std_cond = appearance_mean_std(
        q_normed_dict["cond"], k_normed_dict["appearance_cond"], v_dict["appearance_cond"],
    )

    if reshape_fn is not None:
        mean_cond = reshape_fn(mean_cond)
        std_cond = reshape_fn(std_cond)

    features_dict["cond"] = std_cond * normalize(features_dict["cond"], dim=-2) + mean_cond
    
    features = batch_dict_to_tensor(features_dict, batch_order)
    return features
