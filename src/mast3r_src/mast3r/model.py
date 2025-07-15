# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa
from dust3r.patch_embed import get_patch_embed as dust3r_patch_embed # 我们有了自己的patch_embed用于处理深度图mlp，这里要换个名字
from patch_embed import get_patch_embed # pow3r设计的


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
        self.patch_embed = dust3r_patch_embed(patch_embed_cls, img_size, patch_size, self.enc_embed_dim)
        # 这个get_patch_embed是pow3r实现的，一个用于深度图的方法。
        self.patch_embed_depth = get_patch_embed(patch_embed_cls + '_Mlp', img_size, patch_size, self.enc_embed_dim,
                                                 in_chans=2)
        super().__init__(**kwargs)

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
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if depth is not None: # B,2,H,W ？为什么是2？
            # 包含稀疏深度图和掩码（形状为(2, H, W)），第一维分别是归一化后的稀疏深度图和对应的掩码。
            # 这里注意要修改深度图获取那里。pow3r的稀疏深度图是采样获取的，而不是由雷达得到。
            depth_emb, pos2 = self.patch_embed_depth(depth, true_shape=true_shape)
            assert (pos == pos2).all()
            # if self.mode.startswith('embed'): 不知道在干啥，这里理论上没有别的
            x = x + depth_emb # 这一步增加了深度信息，看上去没有形状上的改变
        else:
            depth_emb = None

        x = self.patch_ln(x)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos, depth=depth_emb)  # 这里可能有问题，完全没找到这个depth参数的来源。

        x = self.enc_norm(x)
        return x, pos

    def encode_symmetrized(self, view1, view2):
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

        feat1, pos1 = self._encode_image(img1, shape1, depth=depth1)
        feat2, pos2 = self._encode_image(img2, shape2, depth=depth2)

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
