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
from functools import partial


inf = float('inf')

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

# 深度编码器，用到了一点稀疏CNN？
# class SparseDepthEncoder(nn.Module):
#     """增强的稀疏深度编码器，显式处理掩码"""
#
#     def __init__(self, in_channels=2, out_channels=32):
#         super().__init__()
#         # 显式处理稀疏性的编码器
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
#         self.mask_pool = nn.MaxPool2d(3, stride=1, padding=1)  # 掩码下采样
#
#     def forward(self, x):
#         # 分离深度和掩码 [B,2,H,W]
#         depth_map, mask = x.chunk(2, dim=1)
#
#         # 初始处理 (保留掩码信息)
#         x = torch.cat([depth_map * mask, mask], dim=1)  # [B,2,H,W]
#         x = F.relu(self.conv1(x))
#
#         # 下采样掩码并应用
#         mask = self.mask_pool(mask)
#         x = x * mask  # 掩码引导的特征选择
#
#         x = F.relu(self.conv2(x))
#         mask = self.mask_pool(mask)
#         x = x * mask
#
#         return self.conv3(x)  # [B, out_channels, H, W]
#
# # gpt建议的后融合策略，避免前融合导致特征提取失败
# class FeatureFusionGate(nn.Module):
#     """门控特征融合模块"""
#
#     def __init__(self, rgb_channels, depth_channels):
#         super().__init__()
#         # 通道对齐
#         self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
#         # 门控机制
#         self.gate = nn.Sequential(
#             nn.Conv2d(rgb_channels * 2, rgb_channels, 3, padding=1),
#             nn.Sigmoid()  # 生成0-1融合权重
#         )
#
#     def forward(self, rgb_feat, depth_feat):
#         # 对齐深度特征维度
#         depth_feat = self.depth_proj(depth_feat)
#
#         # 生成门控权重
#         gate = self.gate(torch.cat([rgb_feat, depth_feat], dim=1))
#
#         # 门控融合
#         return rgb_feat * gate + depth_feat * (1 - gate)

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

        # 我们增加的内容
        # 稀疏深度编码器 (输出32通道)
        # self.depth_encoder = SparseDepthEncoder(in_channels=2, out_channels=32)

    # def _reshape_to_2d(self, tokens, true_shape):
    #     """将ViT输出token序列转换为2D特征图"""
    #     B, N, C = tokens.shape
    #
    #     # 计算特征图尺寸 (考虑非方形patch)
    #     H = true_shape[:, 0] // self.patch_size
    #     W = true_shape[:, 1] // self.patch_size
    #
    #     # 重塑为2D特征图 [B, C, H, W]
    #     feat_maps = []
    #     for i in range(B):
    #         h, w = H[i].item(), W[i].item()
    #         feat = tokens[i].view(h, w, C).permute(2, 0, 1)  # [C, H, W]
    #         feat_maps.append(feat)
    #
    #     return torch.stack(feat_maps)  # [B, C, H, W]

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        self.patch_embed = dust3r_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # 这个get_patch_embed是pow3r实现的，一个用于深度图的方法。
        # self.patch_embed_depth = get_patch_embed(self.patch_embed_cls + '_Mlp', img_size, patch_size, enc_embed_dim,
        #                                          in_chans=2)
        # 保存关键尺寸信息
        self.patch_size = patch_size
        self.img_size = img_size
        self.enc_norm = norm_layer(enc_embed_dim)
        # 计算特征图尺寸 (ViT输出维度)
        self.feat_height = img_size[0] // patch_size
        self.feat_width = img_size[1] // patch_size
        self.feat_channels = self.enc_norm.normalized_shape[0]  # 从LayerNorm获取通道数

        # 特征融合门 (RGB特征通道来自ViT，深度特征32通道)
        # self.fusion_gate = FeatureFusionGate(
        #     rgb_channels=self.feat_channels,
        #     depth_channels=32
        # )

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
    def _encode_image(self, image, true_shape):
        # """改进的图像编码方法：使用多阶段融合"""
        # if depth is not None:
        #     # 分离深度和掩码
        #     depth_map = depth[:, 0:1]  # [B,1,H,W]
        #     mask = depth[:, 1:2]       # [B,1,H,W]
        #
        #     # 应用掩码
        #     depth_map = depth_map * mask
        #
        #     # 深度特征增强
        #     depth_feat = self.depth_encoder(depth)
        #
        #     # 拼接RGB和增强后的深度特征
        #     image_with_depth = torch.cat([image, depth_feat], dim=1)
        #
        #     # 通过深度融合模块
        #     fused_image = self.depth_fusion(image_with_depth)
        # else:
        #     fused_image = image
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        tokens = self.enc_norm(x)  # [B, N, C]

        # 深度处理
        # depth_feat = None
        # if depth is not None:
        #     depth_feat = self.depth_encoder(depth)  # [B, 32, H, W]

        # 返回token序列和深度特征
        return tokens, pos

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # 准备深度输入
        # depth1 = view1.get('depthmap', None)
        # mask1 = view1.get('valid_mask', None)
        # depth1_input = torch.stack([depth1, mask1], dim=1) if depth1 is not None else None

        # depth2 = view2.get('depthmap', None)
        # mask2 = view2.get('valid_mask', None)
        # depth2_input = torch.stack([depth2, mask2], dim=1) if depth2 is not None else None

        # 编码图像
        tokens1, pos1 = self._encode_image(img1, shape1)
        tokens2, pos2= self._encode_image(img2, shape2)

        return (shape1, shape2), (tokens1, tokens2), (pos1, pos2)

    # def _fuse_features(self, tokens, true_shape, depth_feat):
    #     """融合ViT特征和深度特征"""
    #     # 转换token为2D特征图
    #     rgb_feat = self._reshape_to_2d(tokens, true_shape)  # [B, C, H, W]
    #
    #     if depth_feat is not None:
    #         # 调整深度特征分辨率 (与ViT特征图匹配)
    #         depth_feat = F.interpolate(
    #             depth_feat,
    #             size=(rgb_feat.shape[2], rgb_feat.shape[3]),
    #             mode='bilinear',
    #             align_corners=False
    #         )
    #         # 门控融合
    #         fused_feat = self.fusion_gate(rgb_feat, depth_feat)
    #     else:
    #         fused_feat = rgb_feat
    #
    #     # 重新展平为token序列 [B, N, C]
    #     B, C, H, W = fused_feat.shape
    #     return fused_feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

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

    def forward(self, view1, view2):
        """处理单视图输入（包含RGB和深度图）"""
        # 编码单视图的RGB和深度图
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

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
