import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import utils.loss_mask as loss_mask
# from utils import loss_mask as loss_mask
# import loss_mask
import PIL.Image
import src.mast3r_src.dust3r.dust3r.datasets.utils.cropping as cropping # 导入裁剪和缩放模块

import torchvision
from PIL import ImageDraw
import os
import json
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2
from data.data import crop_resize_if_necessary, DUST3RSplattingDataset, DUST3RSplattingTestDataset
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor, as_completed

# Waymo到OpenCV坐标系的转换矩阵
WAYMO2OPENCV = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

#
# @torch.no_grad()
# def calculate_loss_mask(targets, context):
#     '''计算目标视图在上下文视图视锥体内的有效掩码'''
#     # 从目标视图列表中提取深度图
#     target_depth = torch.stack([target_view['depthmap'] for target_view in targets], dim=1)
#     # 提取目标视图的内参矩阵
#     target_intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in targets], dim=1)
#     # 提取目标视图的相机位姿 (c2w)
#     target_c2w = torch.stack([target_view['camera_pose'] for target_view in targets], dim=1)
#
#     # 从上下文视图列表中提取深度图
#     context_depth = torch.stack([context_view['depthmap'] for context_view in context], dim=1)
#     # 提取上下文视图的内参矩阵
#     context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in context], dim=1)
#     # 提取上下文视图的相机位姿
#     context_c2w = torch.stack([context_view['camera_pose'] for context_view in context], dim=1)
#
#     # 确保内参矩阵是3x3形式
#     target_intrinsics = target_intrinsics[..., :3, :3]
#     context_intrinsics = context_intrinsics[..., :3, :3]
#
#     # 计算目标视图在上下文视图视锥体内的有效掩码
#     mask = loss_mask.calculate_in_frustum_mask(
#         target_depth, target_intrinsics, target_c2w,
#         context_depth, context_intrinsics, context_c2w
#     )
#     return mask
#

# 深度图处理函数保持不变...
# 从压缩格式重建深度图 (从第二段代码移植)
def reconstruct_depth_map(depth_data, original_shape):
    """
    从压缩格式重建深度图
    :param depth_data: 从.npy文件加载的字典数据
    :param original_shape: 原始深度图形状 (H, W)
    :return: 重建后的深度图 (H, W)
    """
    depth_map = np.zeros(original_shape, dtype=np.float32)
    mask = depth_data['mask']
    values = depth_data['value']
    depth_map[mask] = values
    return depth_map


# 深度图标准化函数
def normalize_depth_map(depth_map, max_depth=100.0):
    """.astype(np.float32)
    标准化深度图：
    1. 截断到最大深度值
    2. 归一化到[0, 1]范围
    3. 转换为三通道伪RGB
    """
    # 截断深度值
    depth_map = np.clip(depth_map, 0, max_depth)

    # 归一化到[0, 1]
    normalized = depth_map / max_depth
    return np.moveaxis(normalized, -1, 0)
    # # 转换为三通道伪RGB
    # rgb = np.stack([normalized] * 3, axis=-1)  # 形状变为 [H, W, 3]
    #
    # return np.m

'''
def crop_resize_if_necessary(image, depthmap, intrinsics, resolution):
    """Adapted from DUST3R's Co3D dataset implementation"""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # Downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    assert min_margin_x > W / 5
    assert min_margin_y > H / 5
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # High-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    # Actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2
'''


def crop_resize_if_necessary(image, depthmap, intrinsics, resolution):
    """Adapted from DUST3R's Co3D dataset implementation with debug info"""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # Downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)

    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)

    try:
        assert min_margin_x > W / 5, f"min_margin_x ({min_margin_x}) <= W/5 ({W/5})"
        assert min_margin_y > H / 5, f"min_margin_y ({min_margin_y}) <= H/5 ({H/5})"
    except AssertionError as e:
        # 创建错误截图保存到文件
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        # 绘制主点位置
        draw.ellipse([(cx-10, cy-10), (cx+10, cy+10)], outline="red", width=3)
        # 绘制预期裁剪区域
        draw.rectangle([(cx - min_margin_x, cy - min_margin_y), 
                       (cx + min_margin_x, cy + min_margin_y)], outline="yellow", width=2)
        # 保存调试图像
        debug_img.save("crop_debug_error.png")
        print(f"Assertion failed! Debug image saved as crop_debug_error.png")
        raise e
    
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    
    # ==================== DEBUG INFO ADDED ====================
    # print(f"Debug Info - Crop BBOX: {crop_bbox}")
    
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # High-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    # Actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2

def load_waymo_scene_data(scene_dir, resolution=None):
    """
    加载Waymo场景数据
    :param scene_dir: 场景目录路径 (Path对象)
    :param resolution: 目标分辨率 (width, height)
    :return: 视图列表，每个视图包含图像、深度图、内参、位姿等完整信息
    """
    views = []
    # 加载相机内参 (5个相机)
    intrinsics = {}
    for cam_id in range(5):
        intr_file = scene_dir / "intrinsics" / f"{cam_id}.txt"
        params = np.loadtxt(intr_file)
        # Waymo参数格式: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3] (共9个)
        if len(params) >= 4:
            # 提取主要内参参数
            f_u, f_v, c_u, c_v = params[:4]
            # 构建标准3x3内参矩阵
            K = np.array([
                [f_u, 0,   c_u],
                [0,   f_v, c_v],
                [0,   0,   1]
            ])
        else:
            raise ValueError(f"Invalid intrinsic parameters: {params}")
        intrinsics[cam_id] = K

    # 加载相机到ego的外参
    extrinsics = {}
    for cam_id in range(5):
        extr_file = scene_dir / "extrinsics" / f"{cam_id}.txt"
        extrinsics[cam_id] = np.loadtxt(extr_file).reshape(4, 4)

    # 获取所有图像帧ID (基于图像文件名)
    img_dir = scene_dir / "images"
    frame_ids = sorted({int(f.name.split('_')[0]) for f in img_dir.glob("*.png")})

    # 遍历所有帧和相机
    for frame_id in frame_ids:
        # 加载ego位姿 (世界坐标系到ego)
        ego_pose_file = scene_dir / "ego_pose" / f"{frame_id:06d}.txt"
        if not ego_pose_file.exists():
            continue  # 跳过缺失位姿的帧
        ego_pose = np.loadtxt(ego_pose_file).reshape(4, 4)

        for cam_id in range(5):
            # 构建图像路径
            img_path = img_dir / f"{frame_id:06d}_{cam_id}.png"
            if not img_path.exists():
                continue

            # ===== 计算相机到世界的变换 (c2w) =====
            # 相机到ego的变换
            cam_to_ego = extrinsics[cam_id]
            # ego到世界的变换
            ego_to_world = ego_pose
            # 组合变换: 相机->ego->世界
            cam_to_world = ego_to_world @ cam_to_ego
            # 转换为OpenCV坐标系
            cam_to_world_opencv = WAYMO2OPENCV @ cam_to_world

            # ===== 加载原始图像 =====
            rgb_image = PIL.Image.open(img_path)

            # ===== 加载深度图 =====
            depth_path = scene_dir / "lidar_depth" / f"{frame_id:06d}_{cam_id}.npy"
            if not depth_path.exists():
                continue

            # 加载压缩的深度数据
            depth_data = np.load(depth_path, allow_pickle=True).item()
            # 重建深度图
            original_shape = rgb_image.size[::-1]  # (H, W)
            depth_map = reconstruct_depth_map(depth_data, original_shape)

            # 获取当前相机的内参
            K = intrinsics[cam_id].copy()

            # ===== 应用裁剪和缩放 =====
            if resolution:
                rgb_image, depth_map, K = crop_resize_if_necessary(
                    rgb_image, depth_map, K, resolution
                )

            # ===== 创建有效掩码 =====
            valid_mask = depth_map > 1e-6
            sky_mask = depth_map <= 0.0

            # ===== 构建完整的视图字典 =====
            view_data = {
                'original_img': rgb_image,  # 保持为PIL图像，稍后转换
                'depthmap': depth_map,  # numpy数组
                'camera_pose': cam_to_world_opencv,  # 4x4矩阵
                'camera_intrinsics': K,  # 3x3矩阵
                'dataset': 'waymo',
                'label': f"waymo/{scene_dir.name}/{frame_id}_{cam_id}",
                'instance': f'{frame_id}_{cam_id}',
                'is_metric_scale': True,
                'sky_mask': sky_mask,  # 天空掩码
                'valid_mask': valid_mask,  # 有效深度掩码
            }
            views.append(view_data)

    return views


if __name__ == '__main__':
    # 配置参数
    DATA_ROOT = "/home/robot/zyr/waymo/Sprocessed"  # 替换为实际路径
    OUTPUT_DIR = "/home/robot/zyr/waymo/coverage"
    BATCH_SIZE = 5  # 根据GPU内存调整/home/robot/mfx
    RESOLUTION = (1920, 1280)  # 处理分辨率 (width, height)

    # 获取GPU编号
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    device = torch.device(f'cuda:{gpu_id}')

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有场景
    base_path = Path(DATA_ROOT)
    scenes = sorted([d.name for d in base_path.iterdir() if d.is_dir()])

    # 多GPU分配场景 (示例: 4个GPU)
    scenes = [scenes[i] for i in range(int(gpu_id), len(scenes), 4)]

    # 图像转换器 (稍后用于转换为张量)
    org_transform = torchvision.transforms.ToTensor()

    for scene_id in tqdm(scenes, desc=f"GPU {gpu_id} Processing Scenes"):
        scene_dir = base_path / scene_id
        output_path = Path(OUTPUT_DIR) / f"{scene_id}.json"

        # 如果已处理则跳过
        if output_path.exists():
            continue

        print(f"\nProcessing scene {scene_id} on GPU {gpu_id}")

        # 加载场景数据
        views = load_waymo_scene_data(scene_dir, resolution=RESOLUTION)
        if not views:
            print(f"警告: 场景 {scene_id} 无有效数据")
            continue

        # 转换视图数据为张量并移至GPU
        for view in views:
            # 转换图像为张量
            view['original_img'] = org_transform(view['original_img']).to(device)

            # 转换深度图为张量
            depth_tensor = torch.tensor(view['depthmap']).unsqueeze(0).float()
            view['depthmap'] = depth_tensor.to(device)

            # 转换掩码为张量
            view['valid_mask'] = torch.tensor(view['valid_mask']).unsqueeze(0).to(device)
            view['sky_mask'] = torch.tensor(view['sky_mask']).unsqueeze(0).to(device)

            # 转换内参和位姿为张量
            view['camera_intrinsics'] = torch.tensor(view['camera_intrinsics']).float().unsqueeze(0).to(device)
            view['camera_pose'] = torch.tensor(view['camera_pose']).float().unsqueeze(0).to(device)

        # 存储覆盖率结果
        coverage_matrix = []
        num_views = len(views)

        # 遍历每个视图作为上下文 (context)
        for i in tqdm(range(num_views), desc="Processing context views"):
            context_view = [views[i]]  # 包装为列表

            # 分批计算覆盖率
            coverage_vals = []
            for batch_start in range(0, num_views, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_views)
                batch_targets = views[batch_start:batch_end]

                # 计算当前批次的掩码
                masks = calculate_loss_mask(batch_targets, context_view)

                # 计算覆盖率 (有效像素比例)
                # masks形状: [batch_size, 1, H, W] -> 在空间维度求平均
                batch_coverage = masks.float().mean(dim=[2, 3]).squeeze(1)
                coverage_vals.append(batch_coverage)

            # 合并批次结果
            coverage_vals = torch.cat(coverage_vals).cpu().numpy()
            coverage_matrix.append(coverage_vals.tolist())

        # 保存结果 (使用序列号作为键)
        result = {scene_id: coverage_matrix}
        with open(output_path, 'w') as f:
            json.dump(result, f)

        print(f"Saved coverage for {scene_id} with {num_views} views")