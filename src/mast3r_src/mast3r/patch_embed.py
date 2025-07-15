# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import torch.nn.functional as F

import pow3r.tools.path_to_dust3r
from dust3r.patch_embed import PatchEmbedDust3R, ManyAR_PatchEmbed  # noqa
from croco.models.blocks import Mlp


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):
    assert patch_embed_cls in ['PatchEmbedDust3R_Mlp', 'ManyAR_PatchEmbed_Mlp']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, in_chans, enc_embed_dim)

    # TODO remove
    # compute grid size and attach it, useful for init of positional encoding 
    # assert all(img_size[i] % ps == 0 for i,ps in enumerate(patch_size))
    # patch_embed.grid_size = tuple(img_size[i] // ps for i,ps in enumerate(patch_size))

    return patch_embed


class Permute(torch.nn.Module):
    dims: tuple[int, ...]
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        return f"Permute{self.dims}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)


class PixelUnshuffle (nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        if input.numel() == 0:
            # this is not in the original torch implementation
            C,H,W = input.shape[-3:]
            assert H and W and H % self.downscale_factor == W%self.downscale_factor == 0
            return input.view(*input.shape[:-3], C*self.downscale_factor**2, H//self.downscale_factor, W//self.downscale_factor)
        else:
            return F.pixel_unshuffle(input, self.downscale_factor)

class PatchEmbed_Mlp (PatchEmbedDust3R):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0,2,3,1)),
            Mlp(in_chans * patch_size**2, 4*embed_dim, embed_dim),
            Permute((0,3,1,2)),
            )
    

class ManyAR_PatchEmbed_Mlp (ManyAR_PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
    
        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size), 
            Permute((0,2,3,1)),
            Mlp(in_chans * patch_size**2, 4*embed_dim, embed_dim),
            Permute((0,3,1,2)),
            )
