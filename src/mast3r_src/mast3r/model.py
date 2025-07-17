# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import os

from torch import nn

from .catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa
from ..dust3r.dust3r.patch_embed import get_patch_embed as dust3r_patch_embed # 我们有了自己的patch_embed用于处理深度图mlp，这里要换个名字
from .patch_embed import get_patch_embed # pow3r设计的


inf = float('inf')


class DepthFusionModule(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 主路径：三层卷积+非线性
        main = self.relu(self.conv1(x))
        main = self.relu(self.conv2(main))
        main = self.conv3(main)

        # 残差路径：1x1卷积
        residual = self.residual(x)

        # 融合并激活
        return self.relu(main + residual)

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, use_offsets=False, sh_degree=1,
                 patch_embed_cls='PatchEmbedDust3R', **kwargs): # 嗯，这里也需要cls名称了。

        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree
        self.max_depth = 100
        self.patch_embed_cls = patch_embed_cls


        super().__init__(**kwargs)
        self.patch_ln = nn.Identity()

        # 更强大的深度融合模块
        self.depth_fusion = DepthFusionModule(in_channels=4, out_channels=3)

        # 可选的深度特征增强
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = dust3r_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # 这个get_patch_embed是pow3r实现的，一个用于深度图的方法。
        # self.patch_embed_depth = get_patch_embed(self.patch_embed_cls + '_Mlp', img_size, patch_size, enc_embed_dim,
        #                                          in_chans=2)

    # def _encode_symmetrized(self, view1):
    #     """重写编码方法：使用单视图的RGB和深度图作为双输入""" 这里已经暂时被废弃了。现在用的是pow3r方法
    #     # 从单视图字典中提取RGB和深度图
    #     img1 = view1['img']  # RGB图像 [B, 3, H, W]
    #
    #     # 获取深度图并确保是三通道
    #     depthmap = view1['depthmap']
    #     if depthmap.dim() == 3:  # 如果是单通道 [B, H, W]
    #         depthmap = depthmap.unsqueeze(1)  # 添加通道维度 [B, 1, H, W]
    #
    #     # 如果深度图是单通道，转换为三通道伪RGB
    #     if depthmap.size(1) == 1:
    #         # 标准化深度值：截断 + 归一化
    #         depthmap = torch.clamp(depthmap, 0, self.max_depth) / self.max_depth
    #         # 复制为三通道
    #         depthmap = depthmap.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    #
    #     # 获取图像真实尺寸（考虑数据增强后的裁剪）
    #     B = img1.shape[0]
    #     shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
    #     shape2 = shape1.clone()  # 深度图与RGB同尺寸
    #
    #     # 编码图像对（RGB + 深度伪RGB）
    #     feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, depthmap, shape1, shape2)
    #
    #     return (shape1, shape2), (feat1, feat2), (pos1, pos2)



    # 从pow3r那里搞来的encoder图像处理，包括了深度。
    def _encode_image(self, image, true_shape, depth=None):
        """改进的图像编码方法：使用多阶段融合"""
        if depth is not None:
            # 分离深度和掩码
            depth_map = depth[:, 0:1]  # [B,1,H,W]
            mask = depth[:, 1:2]       # [B,1,H,W]

            # 应用掩码
            depth_map = depth_map * mask

            # 可选：深度特征增强
            depth_feat = self.depth_encoder(depth)

            # 拼接RGB和增强后的深度特征
            image_with_depth = torch.cat([image, depth_feat], dim=1)

            # 通过深度融合模块
            fused_image = self.depth_fusion(image_with_depth)
        else:
            fused_image = image

        # 使用融合后的图像进行嵌入
        x, pos = self.patch_embed(fused_image, true_shape=true_shape)
        x = self.patch_ln(x)

        # 添加位置嵌入（无cls token）
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)  # 这里可能有问题，完全没找到这个depth参数的来源。

        x = self.enc_norm(x)
        return x, pos

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        # privileged information
        # rays1 = view1.get('known_rays', None)
        # rays2 = view2.get('known_rays', None)
        depth1 = view1.get('depthmap', None)
        depth2 = view2.get('depthmap', None)
        mask1 = view1.get('valid_mask', None)
        mask2 = view2.get('valid_mask', None)

        # 准备深度输入：将深度图和掩码堆叠为2通道张量
        depth1_input = torch.stack([depth1, mask1], dim=1) if depth1 is not None else None
        depth2_input = torch.stack([depth2, mask2], dim=1) if depth2 is not None else None

        # 使用修改后的_encode_image方法
        feat1, pos1 = self._encode_image(img1, shape1, depth=depth1_input)
        feat2, pos2 = self._encode_image(img2, shape2, depth=depth2_input)
        # print("depth形状",depth1_two_channel.shape)

        # if is_symmetrized(view1, view2):
        #     # computing half of forward pass!'
        #     def hsub(x):
        #         return None if x is None else x[::2]
        #
        #     feat1, pos1 = self._encode_image(img1[::2], shape1[::2], rays=hsub(rays1), depth=hsub(depth1))
        #     feat2, pos2 = self._encode_image(img2[::2], shape2[::2], rays=hsub(rays2), depth=hsub(depth2))
        #
        #     feat1, feat2 = interleave(feat1, feat2)
        #     pos1, pos2 = interleave(pos1, pos2)
        # else: 对于对称样本的冗余处理，暂时没条件实现。直接走else吧.

        # feat1, pos1 = self._encode_image(img1, shape1, depth=depth1_two_channel)
        # feat2, pos2 = self._encode_image(img2, shape2, depth=depth2_two_channel)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), use_offsets=self.use_offsets, sh_degree=self.sh_degree)
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), use_offsets=self.use_offsets, sh_degree=self.sh_degree)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def forward(self, view1):
        """处理单视图输入（包含RGB和深度图）"""
        # 编码单视图的RGB和深度图
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1)

        # 解码器处理双特征流
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        # 下游头生成预测结果
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)  # RGB分支
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)  # 深度图分支

        # 调整输出格式
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # 重命名
        res2['from_depth'] = True  # 标记结果来自深度图

        # 添加原始深度信息（可选，用于损失计算）
        res2['raw_depth'] = view1['depthmap'].clone()

        return res1, res2
