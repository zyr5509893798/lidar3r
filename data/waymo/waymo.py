import os
import json
import logging
import sys

import cv2
import numpy as np
import logging
import numpy as np
from pathlib import Path
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2
from data.data import crop_resize_if_necessary

logger = logging.getLogger(__name__)

from data.data import crop_resize_if_necessary, DUST3RSplattingDataset, DUST3RSplattingTestDataset

# Waymo到OpenCV坐标系的转换矩阵
WAYMO2OPENCV = np.array([
    [0, -1, 0, 0],  # Waymo Y(左) -> OpenCV X(右)
    [0, 0, -1, 0],  # Waymo Z(上) -> OpenCV Y(下)
    [1, 0, 0, 0],  # Waymo X(前) -> OpenCV Z(前)
    [0, 0, 0, 1]
], dtype=np.float32)


class WaymoData:
    def __init__(self, root, stage):
        self.root = root
        self.stage = stage

        # 数据结构
        self.color_paths = {}
        self.depth_paths = {}  # 新增：存储深度图路径
        self.intrinsics = {}
        self.c2ws = {}
        self.sequences = []

        # 获取场景列表
        split_file = Path(root) / 'splits' / f'{stage}.txt'
        if not split_file.exists():
            logger.error(f"划分文件不存在: {split_file}")
            return

        with open(split_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]

        # 处理每个场景
        scenes_with_no_frames = []
        for seq in self.sequences:
            scene_dir = Path(root) / 'Sprocessed' / seq

            # 检查基本目录结构
            required_dirs = ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar_depth']  # 增加lidar_depth检查
            if not all((scene_dir / d).exists() for d in required_dirs):
                logger.warning(f"场景 {seq} 缺少必要目录，跳过")
                continue

            # 加载相机内参和外参
            cam_intrinsics = {}
            cam_extrinsics = {}
            valid_cameras = True

            for cam_id in range(5):  # 5个相机
                # 内参
                intr_file = scene_dir / "intrinsics" / f"{cam_id}.txt"
                params = np.loadtxt(intr_file).astype(np.float32)
                # Waymo参数格式: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3] (共9个)
                if len(params) >= 4:
                    # 提取主要内参参数
                    f_u, f_v, c_u, c_v = params[:4]
                    # 构建标准3x3内参矩阵
                    K = np.array([
                        [f_u, 0, c_u],
                        [0, f_v, c_v],
                        [0, 0, 1]
                    ], dtype=np.float32)
                else:
                    raise ValueError(f"Invalid intrinsic parameters: {params}")
                cam_intrinsics[cam_id] = K

                # 外参 (相机到ego的变换)
                extrinsics_file = scene_dir / 'extrinsics' / f'{cam_id}.txt'
                if not extrinsics_file.exists():
                    logger.warning(f"场景 {seq} 相机 {cam_id} 缺少外参文件")
                    valid_cameras = False
                    break

                T_cam_ego = np.loadtxt(extrinsics_file).astype(np.float32)
                if T_cam_ego.size != 16:
                    logger.warning(f"场景 {seq} 相机 {cam_id} 外参格式错误")
                    valid_cameras = False
                    break

                cam_extrinsics[cam_id] = T_cam_ego.reshape(4, 4)

            if not valid_cameras:
                logger.warning(f"场景 {seq} 相机参数不完整，跳过")
                continue

            # 获取所有帧的ID (通过ego_pose文件)
            pose_files = sorted((scene_dir / 'ego_pose').glob('*.txt'))
            # 关键修改：过滤掉文件名带下划线的文件（如000000_0.txt）
            pose_files = [f for f in pose_files if '_' not in f.stem]  # 新增过滤行

            if not pose_files:
                logger.warning(f"场景 {seq} 没有ego位姿文件")
                scenes_with_no_frames.append(seq)
                continue

            frame_ids = [int(f.stem) for f in pose_files]


            # 为场景存储数据
            scene_color_paths = []
            scene_depth_paths = []  # 新增：存储深度图路径
            scene_intrinsics = []
            scene_c2ws = []

            for frame_id in frame_ids:

                # 初始化当前帧的数据列表，5个相机，二维
                frame_color_paths = [None] * 5
                frame_depth_paths = [None] * 5
                frame_intrinsics = [None] * 5
                frame_c2ws = [None] * 5

                # 加载ego位姿
                ego_pose_file = scene_dir / 'ego_pose' / f'{frame_id:06d}.txt'
                if not ego_pose_file.exists():
                    logger.warning(f"场景 {seq} 帧 {frame_id} 缺少ego位姿，跳过")
                    continue

                ego_pose = np.loadtxt(ego_pose_file).reshape(4, 4).astype(np.float32)

                # 处理每个相机
                for cam_id in range(5):
                    img_file = scene_dir / 'images' / f'{frame_id:06d}_{cam_id}.png'
                    depth_file = scene_dir / 'lidar_depth' / f'{frame_id:06d}_{cam_id}.npy'  # 深度图路径

                    if not img_file.exists():
                        logger.warning(f"场景 {seq} 帧 {frame_id} 相机 {cam_id} 缺少图像，跳过")
                        continue

                    # 新增：检查深度图是否存在
                    if not depth_file.exists():
                        logger.warning(f"场景 {seq} 帧 {frame_id} 相机 {cam_id} 缺少深度图，跳过")
                        continue

                    # 计算相机到世界的变换
                    T_cam_ego = cam_extrinsics[cam_id]
                    T_ego_world = ego_pose
                    T_cam_world = T_ego_world @ T_cam_ego

                    # 转换到OpenCV坐标系
                    T_cam_world_opencv = WAYMO2OPENCV @ T_cam_world

                    # 将数据存储到当前帧的对应相机位置，二维
                    frame_color_paths[cam_id] = str(img_file)
                    frame_depth_paths[cam_id] = str(depth_file)
                    frame_intrinsics[cam_id] = cam_intrinsics[cam_id].copy()
                    frame_c2ws[cam_id] = T_cam_world_opencv

                    # # 存储数据，一维
                    # scene_color_paths.append(str(img_file))
                    # scene_depth_paths.append(str(depth_file))  # 存储深度图路径
                    # scene_intrinsics.append(cam_intrinsics[cam_id].copy())
                    # scene_c2ws.append(T_cam_world_opencv)

                # 将当前帧的数据添加到场景数据中，二维
                scene_color_paths.append(frame_color_paths)
                scene_depth_paths.append(frame_depth_paths)
                scene_intrinsics.append(frame_intrinsics)
                scene_c2ws.append(frame_c2ws)

            if not scene_color_paths:
                logger.warning(f"场景 {seq} 没有有效帧，跳过")
                scenes_with_no_frames.append(seq)
                continue

            self.color_paths[seq] = scene_color_paths
            self.depth_paths[seq] = scene_depth_paths  # 存储深度图路径
            self.intrinsics[seq] = scene_intrinsics
            self.c2ws[seq] = scene_c2ws

        # 更新有效场景列表
        self.sequences = [seq for seq in self.sequences if seq not in scenes_with_no_frames]
        logger.info(f"成功加载 {len(self.sequences)} 个场景的数据")

    # def get_view(self, sequence, view_idx, resolution):
    def get_view(self, sequence, frame_idx, cam_idx, resolution):
        if sequence not in self.color_paths:
            raise ValueError(f"无效场景: {sequence}")

        if frame_idx >= len(self.color_paths[sequence]):
            raise ValueError(f"无效视图索引: {frame_idx} (最大 {len(self.color_paths[sequence]) - 1})")

        # 读取图像
        img_path = self.color_paths[sequence][frame_idx][cam_idx]
        rgb_image = imread_cv2(img_path)

        # 读取深度图
        depth_path = self.depth_paths[sequence][frame_idx][cam_idx]
        depth_data = np.load(depth_path, allow_pickle=True).item()
        depth_map = reconstruct_depth_map(depth_data, rgb_image.shape[:2])  # (H, W)

        # 获取内参和位姿
        intrinsics = self.intrinsics[sequence][frame_idx][cam_idx]
        c2w = self.c2ws[sequence][frame_idx][cam_idx]

        # 调整大小 (同时处理图像和深度图)
        rgb_image, depth_map, intrinsics = crop_resize_if_necessary(
            rgb_image, depth_map, intrinsics, resolution
        )

        # 创建有效掩码和天空掩码
        valid_mask = (depth_map > 1e-6).astype(np.float32)  # 转换为float32
        sky_mask = depth_map <= 0.0

        # 标准化深度图
        normalized_depth = normalize_depth_map(depth_map)  # (H, W)

        # 创建两通道深度图 [2, H, W],这里调整到model内部进行堆叠，因为loss_mask要使用一维深度图。必须是{H W]
        # depth_two_channel = np.stack([normalized_depth, valid_mask], axis=0)  # 直接堆叠为 [2, H, W]

        return {
            'original_img': rgb_image,
            'depthmap': normalized_depth,  # 形状 [H, W]
            'camera_pose': c2w,
            'camera_intrinsics': intrinsics,
            'dataset': 'waymo',
            'label': f"waymo/{sequence}",
            'instance': f'{frame_idx}',
            'camera_id': f"{cam_idx}",
            'is_metric_scale': True,
            'sky_mask': sky_mask,
            'valid_mask': valid_mask  # 仍然是二维数组 (H, W)
        }

# 下面是原始get_view，没有相机id
    # def get_view(self, sequence, view_idx, resolution):
    #     if sequence not in self.color_paths:
    #         raise ValueError(f"无效场景: {sequence}")
    #
    #     if view_idx >= len(self.color_paths[sequence]):
    #         raise ValueError(f"无效视图索引: {view_idx} (最大 {len(self.color_paths[sequence]) - 1})")
    #
    #     # 读取图像
    #     img_path = self.color_paths[sequence][view_idx]
    #     rgb_image = imread_cv2(img_path)
    #
    #     # 新增：读取深度图
    #     depth_path = self.depth_paths[sequence][view_idx]
    #     depth_data = np.load(depth_path, allow_pickle=True).item()
    #     depth_map = reconstruct_depth_map(depth_data, rgb_image.shape[:2])  # 原始图像形状 (H, W)
    #
    #     # 获取内参和位姿
    #     intrinsics = self.intrinsics[sequence][view_idx]
    #     c2w = self.c2ws[sequence][view_idx]
    #
    #     # 调整大小 (同时处理图像和深度图)
    #     rgb_image, depth_map, intrinsics = crop_resize_if_necessary(
    #         rgb_image, depth_map, intrinsics, resolution
    #     )
    #
    #     # 创建有效掩码和天空掩码
    #     valid_mask = depth_map > 1e-6
    #     sky_mask = depth_map <= 0.0
    #
    #     # 标准化并转换为伪RGB
    #     depth_rgb = normalize_depth_map(depth_map)
    #
    #     return {
    #         'original_img': rgb_image,
    #         'depthmap': depth_rgb,  # 现在返回真实的深度图(标准化之后）
    #         'camera_pose': c2w,
    #         'camera_intrinsics': intrinsics,
    #         'dataset': 'waymo',
    #         'label': f"waymo/{sequence}",
    #         'instance': f'{view_idx}',
    #         'is_metric_scale': True,
    #         'sky_mask': sky_mask,  # 新增天空掩码
    #         'valid_mask': valid_mask  # 新增有效深度掩码
    #     }

def get_waymo_dataset(root, stage, resolution, num_epochs_per_epoch=1):

    data = WaymoData(root, stage)

    # coverage = {}
    # for sequence in data.sequences:
    #     with open(f'/home/robot/zyr/waymo/coverage/{sequence}.json', 'r') as f:
    #         sequence_coverage = json.load(f)
    #     coverage[sequence] = {}
    #     # 提取所有5个相机的矩阵 (ID 0 到 4)
    #     for camera_id in range(5):  # 生成 0,1,2,3,4
    #         # JSON键是字符串类型
    #         camera_key = str(camera_id)
    #
    #         # 检查相机是否存在
    #         if camera_key in sequence_coverage['cameras']:
    #             # 提取并存储覆盖矩阵
    #             coverage[sequence][camera_id] = sequence_coverage['cameras'][camera_key]['coverage_matrix']
    #         else:
    #             print(f"警告: 场景{sequence} 相机 {camera_id} 矩阵未在数据中找到")
    #
    #     print(f"场景{sequence}矩阵读取完毕")

    dataset = DUST3RSplattingDataset(
        data,
        # coverage,
        resolution,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )

    return dataset

def get_waymo_test_dataset(root, resolution, use_every_n_sample=100):

    data = WaymoData(root, 'test')

    samples_file = f'data/waymo/test_set.json'
    print(f"Loading samples from: {samples_file}")
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    samples = samples[::use_every_n_sample]

    dataset = DUST3RSplattingTestDataset(data, samples, resolution)

    return dataset

# 深度处理旧代码的问题：H和W被转置了！图都是躺着的。另外我们现在需要2维，带上mask
def normalize_depth_map(depth_map, max_depth=100.0):
    """
    标准化深度图：
    1. 截断到最大深度值
    2. 归一化到[0, 1]范围
    返回形状为 (H, W) 的二维数组
    """
    depth_map = np.clip(depth_map, 0, max_depth)
    normalized = depth_map / max_depth
    return normalized  # 返回二维数组 (H, W)

def reconstruct_depth_map(depth_data, original_shape):
    """从压缩格式重建深度图"""
    depth_map = np.zeros(original_shape, dtype=np.float32)
    mask = depth_data['mask']
    values = depth_data['value']
    depth_map[mask] = values
    return depth_map

