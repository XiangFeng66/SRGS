#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from models.network_swinir import SwinIR as net
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,image_ori, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
        pretrained_model = torch.load("./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
        self.SR_model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)
        self.SR_model.eval()
        self.SR_model = self.SR_model.to(device)

        try:
            self.data_device = torch.device(data_device)
            self.test_data_device = torch.device("cpu")
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_test = image_ori.clamp(0.0, 1.0).to(self.test_data_device)
        with torch.no_grad():
        # pad input image to be a multiple of window_size
            window_size = 8
            self.SR_image = self.original_image.unsqueeze(dim=0)
            _, _, h_old, w_old = self.SR_image.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            self.SR_image = torch.cat([self.SR_image, torch.flip(self.SR_image, [2])], 2)[:, :, :h_old + h_pad, :]
            self.SR_image = torch.cat([self.SR_image, torch.flip(self.SR_image, [3])], 3)[:, :, :, :w_old + w_pad]
            self.SR_image = self.SR_model(self.SR_image)
            self.SR_image = self.SR_image[..., :h_old * 4, :w_old * 4]
            print(self.SR_image.shape)
        self.SR_image = self.SR_image.cpu()
        self.image_width = self.original_image.shape[2] * 4
        self.image_height = self.original_image.shape[1] * 4

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height // 4, self.image_width // 4), device=self.data_device)
            self.image_test *= torch.ones((1, self.image_height, self.image_width),
                                              device=self.test_data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

