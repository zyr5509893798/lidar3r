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
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, use_offsets=False, sh_degree=1, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree
        self.max_depth = 100
        super().__init__(**kwargs)

    def _encode_symmetrized(self, view1):
        """重写编码方法：使用单视图的RGB和深度图作为双输入"""
        # 从单视图字典中提取RGB和深度图
        img1 = view1['img']  # RGB图像 [B, 3, H, W]

        # 获取深度图并确保是三通道
        depthmap = view1['depthmap']
        if depthmap.dim() == 3:  # 如果是单通道 [B, H, W]
            depthmap = depthmap.unsqueeze(1)  # 添加通道维度 [B, 1, H, W]

        # 如果深度图是单通道，转换为三通道伪RGB
        if depthmap.size(1) == 1:
            # 标准化深度值：截断 + 归一化
            depthmap = torch.clamp(depthmap, 0, self.max_depth) / self.max_depth
            # 复制为三通道
            depthmap = depthmap.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # 获取图像真实尺寸（考虑数据增强后的裁剪）
        B = img1.shape[0]
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = shape1.clone()  # 深度图与RGB同尺寸

        # 编码图像对（RGB + 深度伪RGB）
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, depthmap, shape1, shape2)

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
