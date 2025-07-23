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


class CrossModalFusionBlock(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=4):
        super().__init__()
        # 深度特征有效性检测器（备用方案）
        # self.depth_validity = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim // reduction_ratio, 1),  # 输出单通道有效性概率
        #     nn.Sigmoid()  # 压缩到[0,1]范围
        # )

        # 深度注意力生成器（主方案）
        self.attention_gen = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1),  # 1x1卷积降维
            nn.ReLU(),
            nn.Conv2d(embed_dim // reduction_ratio, 1, 1),  # 输出单通道注意力图
            nn.Sigmoid()  # 获得0-1的注意力权重
        )

        # 原始深度值存储
        self.original_depth = None

    # 存储原始深度值（在特征进入Transformer前保存）
    # def store_original_depth(self, depth):
    #     """存储原始深度值用于后续掩码生成
    #     参数:
    #         depth: 原始深度图 [B, C, H, W]
    #     """
    #     self.original_depth = depth

    def forward(self, rgb, depth, original_depth):
        # 输入转换: 序列(B,L,C) -> 空间(B,C,H,W)
        B, L, C = rgb.shape
        H = W = int(L ** 0.5)  # 假设特征图是正方形

        rgb_spatial = rgb.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        depth_spatial = depth.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]

        """ 掩码生成策略（双重保障）"""
        # 方案1: 基于学习到的特征预测有效性
        # feature_validity = self.depth_validity(depth).view(B, H, W, 1)
        # feature_mask = feature_validity.permute(0, 3, 1, 2)  # [B,1,H,W]

        # 方案2: 优先使用原始深度信息（更可靠）
        # if original_depth is not None:
            # 创建二进制掩码（深度>0.1视为有效）
        binary_mask = (original_depth > 0.1).float()
            # 调整掩码尺寸到当前特征图大小（最近邻保持边界清晰）
        mask = F.interpolate(binary_mask, size=(H, W), mode='nearest')
        # else:
        #     mask = feature_mask  # 回退到学习方案

        """ 注意力与融合 """
        # 生成深度注意力图（与掩码相乘过滤无效区）
        raw_attn = self.attention_gen(depth_spatial)  # [B,1,H,W]
        depth_attn = raw_attn * mask  # 应用有效性掩码

        # 门控融合公式
        # RGB权重 = (1 - 深度权重)，深度权重 = depth_attn
        fused = rgb_spatial * (1 - depth_attn) + depth_spatial * depth_attn

        # 输出转换: 空间 -> 序列
        return fused.permute(0, 2, 3, 1).reshape(B, L, C)


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
        self.fusion_stage = 4   # 在第几层之后融合

        super().__init__(**kwargs)
        self.patch_ln = nn.Identity()
        # 添加可训练的融合模块
        self.fusion_blocks = CrossModalFusionBlock(embed_dim=self.enc_embed_dim)


    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        self.patch_embed = dust3r_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # 这个get_patch_embed是pow3r实现的，一个用于深度图的方法。
        self.patch_embed_depth = get_patch_embed(self.patch_embed_cls + '_Mlp', img_size, patch_size, enc_embed_dim,
                                                 in_chans=3)
        # 保存关键尺寸信息
        self.patch_size = patch_size
        self.img_size = img_size
        self.enc_norm = norm_layer(enc_embed_dim)
        # 计算特征图尺寸 (ViT输出维度)
        self.feat_height = img_size[0] // patch_size
        self.feat_width = img_size[1] // patch_size
        self.feat_channels = self.enc_norm.normalized_shape[0]  # 从LayerNorm获取通道数


    # 从pow3r那里搞来的encoder图像处理，包括了深度。
    def _encode_image(self, image, true_shape, depth=None):
        rgb_emb, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        # for blk in self.enc_blocks:
        #     x = blk(x, pos)

        if depth is not None:
            # 关键保存: 原始深度值(在Transformer前)
            original_depth = depth.clone()  # [B,1,H原,W原]
            depth_emb, _ = self.patch_embed_depth(depth, true_shape=true_shape)

            # 独立编码阶段(浅层处理)4层
            for i in range(self.fusion_stage):
                rgb_emb = self.enc_blocks[i](rgb_emb, pos)   #是原本的好还是重新训练好？
                depth_emb = self.enc_blocks[i](depth_emb, pos)  # 位置编码？？

            # 融合当前层特征
            fused_emb = self.fusion_blocks(rgb_emb, depth_emb, original_depth)
            # 融合后阶段8层
            for j in range(len(self.enc_blocks)-self.fusion_stage-1):
                # 传递融合结果给后续Transformer
                fused_emb = self.enc_blocks[self.fusion_stage + j](fused_emb, pos)

            tokens = self.enc_norm(fused_emb)
        else:
            # 纯RGB分支
            for blk in self.enc_blocks:
                rgb_emb = blk(rgb_emb, pos)
            tokens = self.enc_norm(rgb_emb)

        # 返回token序列和深度特征
        return tokens, pos

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # 准备深度输入
        depth1 = view1.get('depth', None)
        depth2 = view2.get('depth', None)

        # 编码图像
        # tokens1, pos1, depth_feat1 = self._encode_image(img1, shape1, depth=depth1)
        # tokens2, pos2, depth_feat2 = self._encode_image(img2, shape2, depth=depth2)

        # return (shape1, shape2), (tokens1, tokens2), (pos1, pos2), (depth_feat1, depth_feat2)


        # 直接相加的融合方法不需要两个encoder输出都返回，只要返回相加后的tokens
        tokens1, pos1 = self._encode_image(img1, shape1, depth=depth1)
        tokens2, pos2 = self._encode_image(img2, shape2, depth=depth2)
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
