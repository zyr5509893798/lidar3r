#!/usr/bin/env python3
# Modified version to directly process images and save PLY without Gradio
import functools
import sys
import tempfile

import gradio
import os
import torch
from huggingface_hub import hf_hub_download
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
sys.path.append('src/pixelsplat_src')
from dust3r.utils.image import load_images
from mast3r.utils.misc import hash_md5
import main
import utils.export as export


def process_images_and_save_ply(image_paths, output_ply_path, model, device, image_size=512):
    """Process images and save directly to PLY file without Gradio"""
    assert len(image_paths) == 1 or len(image_paths) == 2, "Please provide one or two images"
    if len(image_paths) == 1:
        image_paths = [image_paths[0], image_paths[0]]

    # Load and prepare images
    imgs = load_images(image_paths, size=image_size, verbose=True)
    for img in imgs:
        img['img'] = img['img'].to(device)
        img['original_img'] = img['original_img'].to(device)
        img['true_shape'] = torch.from_numpy(img['true_shape'])

    # Run model inference
    output = model(imgs[0], imgs[1])
    pred1, pred2 = output

    # Save as PLY
    export.save_as_ply(pred1, pred2, output_ply_path)
    print(f"Successfully saved 3D Gaussian Splat to: {output_ply_path}")


if __name__ == '__main__':
    # Configuration
    image_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Input images (modify these paths)
    input_images = [
        "demo_examples/in_the_wild_1_img_1.jpg",
        "demo_examples/in_the_wild_1_img_2.jpg"
    ]

    # Output PLY path
    output_ply = "output_gaussians.ply"

    # Load model
    model_name = "brandonsmart/splatt3r_v1.0"
    filename = "epoch=19-step=1200.ckpt"
    weights_path = "pretrained/epoch3D1200.ckpt"  # or use hf_hub_download
    model = main.MAST3RGaussians.load_from_checkpoint(weights_path, device)

    # Process and save
    process_images_and_save_ply(input_images, output_ply, model, device, image_size)