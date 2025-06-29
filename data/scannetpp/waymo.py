import os
import logging
import numpy as np
from pathlib import Path
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2
from data.data import crop_resize_if_necessary

logger = logging.getLogger(__name__)

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
            required_dirs = ['images', 'intrinsics', 'extrinsics', 'ego_pose']
            if not all((scene_dir / d).exists() for d in required_dirs):
                logger.warning(f"场景 {seq} 缺少必要目录，跳过")
                continue

            # 加载相机内参和外参
            cam_intrinsics = {}
            cam_extrinsics = {}
            valid_cameras = True

            for cam_id in range(5):  # 5个相机
                # 内参
                intrinsics_file = scene_dir / 'intrinsics' / f'{cam_id}.txt'
                if not intrinsics_file.exists():
                    logger.warning(f"场景 {seq} 相机 {cam_id} 缺少内参文件")
                    valid_cameras = False
                    break

                K = np.loadtxt(intrinsics_file)
                if K.size != 9:
                    logger.warning(f"场景 {seq} 相机 {cam_id} 内参格式错误")
                    valid_cameras = False
                    break

                K = K.reshape(3, 3)
                # 转换为4x4齐次矩阵
                K_hom = np.eye(4)
                K_hom[:3, :3] = K
                cam_intrinsics[cam_id] = K_hom

                # 外参 (相机到ego的变换)
                extrinsics_file = scene_dir / 'extrinsics' / f'{cam_id}.txt'
                if not extrinsics_file.exists():
                    logger.warning(f"场景 {seq} 相机 {cam_id} 缺少外参文件")
                    valid_cameras = False
                    break

                T_cam_ego = np.loadtxt(extrinsics_file)
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
            if not pose_files:
                logger.warning(f"场景 {seq} 没有ego位姿文件")
                scenes_with_no_frames.append(seq)
                continue

            frame_ids = [int(f.stem) for f in pose_files]

            # 为场景存储数据
            scene_color_paths = []
            scene_intrinsics = []
            scene_c2ws = []

            for frame_id in frame_ids:
                # 加载ego位姿
                ego_pose_file = scene_dir / 'ego_pose' / f'{frame_id:06d}.txt'
                if not ego_pose_file.exists():
                    logger.warning(f"场景 {seq} 帧 {frame_id} 缺少ego位姿，跳过")
                    continue

                ego_pose = np.loadtxt(ego_pose_file).reshape(4, 4)

                # 处理每个相机
                for cam_id in range(5):
                    img_file = scene_dir / 'images' / f'{frame_id:06d}_{cam_id}.png'
                    if not img_file.exists():
                        logger.warning(f"场景 {seq} 帧 {frame_id} 相机 {cam_id} 缺少图像，跳过")
                        continue

                    # 计算相机到世界的变换
                    T_cam_ego = cam_extrinsics[cam_id]
                    T_ego_world = ego_pose
                    T_cam_world = T_ego_world @ T_cam_ego

                    # 转换到OpenCV坐标系
                    T_cam_world_opencv = WAYMO2OPENCV @ T_cam_world

                    # 存储数据
                    scene_color_paths.append(str(img_file))
                    scene_intrinsics.append(cam_intrinsics[cam_id].copy())
                    scene_c2ws.append(T_cam_world_opencv)

            if not scene_color_paths:
                logger.warning(f"场景 {seq} 没有有效帧，跳过")
                scenes_with_no_frames.append(seq)
                continue

            self.color_paths[seq] = scene_color_paths
            self.intrinsics[seq] = scene_intrinsics
            self.c2ws[seq] = scene_c2ws

        # 更新有效场景列表
        self.sequences = [seq for seq in self.sequences if seq not in scenes_with_no_frames]
        logger.info(f"成功加载 {len(self.sequences)} 个场景的数据")

    def get_view(self, sequence, view_idx, resolution):
        if sequence not in self.color_paths:
            raise ValueError(f"无效场景: {sequence}")

        if view_idx >= len(self.color_paths[sequence]):
            raise ValueError(f"无效视图索引: {view_idx} (最大 {len(self.color_paths[sequence]) - 1})")

        # 读取图像
        img_path = self.color_paths[sequence][view_idx]
        rgb_image = imread_cv2(img_path)

        # 获取内参和位姿
        intrinsics = self.intrinsics[sequence][view_idx]
        c2w = self.c2ws[sequence][view_idx]

        # 调整大小 (使用占位深度图)
        fake_depth = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
        rgb_image, _, intrinsics = crop_resize_if_necessary(
            rgb_image, fake_depth, intrinsics, resolution
        )

        return {
            'original_img': rgb_image,
            'depthmap': None,  # Waymo不提供深度图
            'camera_pose': c2w,
            'camera_intrinsics': intrinsics,
            'dataset': 'waymo',
            'label': f"waymo/{sequence}",
            'instance': f'{view_idx}',
            'is_metric_scale': True,
            'sky_mask': None  # 没有深度图无法计算天空掩膜
        }