import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('precompute_coverage')

# Waymo到OpenCV坐标系的转换矩阵
WAYMO2OPENCV = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


def compute_view_overlap(c2w1, K1, c2w2, K2, img_size, depth_estimate=20.0):
    """
    计算两个视图之间的重叠度
    """
    # 获取相对变换
    w2c1 = np.linalg.inv(c2w1)
    T_1_to_2 = K2[:3, :3] @ w2c1[:3, :3] @ c2w2[:3, :3] @ np.linalg.inv(K1[:3, :3])

    # 创建图像网格
    W, H = img_size
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).T

    # 变换坐标
    transformed = T_1_to_2 @ coords
    transformed = transformed[:2] / transformed[2]  # 齐次坐标归一化

    # 计算有效点
    valid_x = (transformed[0] >= 0) & (transformed[0] < W)
    valid_y = (transformed[1] >= 0) & (transformed[1] < H)
    valid = valid_x & valid_y

    # 计算重叠比例
    coverage_ratio = np.sum(valid) / (W * H)

    return coverage_ratio


def precompute_sequence_coverage(sequence_dir, output_dir, img_size=(1920, 1280)):
    """
    预计算单个场景的重叠度矩阵
    """
    sequence = sequence_dir.name
    logger.info(f"开始处理场景: {sequence}")

    # 加载相机内参
    intrinsics = {}
    for cam_id in range(5):  # 5个相机
        intrinsics_file = sequence_dir / 'intrinsics' / f'{cam_id}.txt'
        if not intrinsics_file.exists():
            logger.warning(f"场景 {sequence} 相机 {cam_id} 缺少内参文件")
            return None

        K = np.loadtxt(intrinsics_file).reshape(3, 3)
        # 转换为4x4齐次矩阵
        K_hom = np.eye(4)
        K_hom[:3, :3] = K
        intrinsics[cam_id] = K_hom

    # 加载相机外参
    extrinsics = {}
    for cam_id in range(5):
        extrinsics_file = sequence_dir / 'extrinsics' / f'{cam_id}.txt'
        if not extrinsics_file.exists():
            logger.warning(f"场景 {sequence} 相机 {cam_id} 缺少外参文件")
            return None

        T_cam_ego = np.loadtxt(extrinsics_file).reshape(4, 4)
        extrinsics[cam_id] = T_cam_ego

    # 获取所有帧的ID
    pose_files = sorted((sequence_dir / 'ego_pose').glob('*.txt'))
    if not pose_files:
        logger.warning(f"场景 {sequence} 没有ego位姿文件")
        return None

    frame_ids = [int(f.stem) for f in pose_files]

    # 收集所有视图的位姿和内参
    all_poses = []
    all_intrinsics = []

    for frame_id in frame_ids:
        ego_pose_file = sequence_dir / 'ego_pose' / f'{frame_id:06d}.txt'
        if not ego_pose_file.exists():
            continue

        ego_pose = np.loadtxt(ego_pose_file).reshape(4, 4)

        for cam_id in range(5):
            img_file = sequence_dir / 'images' / f'{frame_id:06d}_{cam_id}.png'
            if not img_file.exists():
                continue

            # 计算相机到世界的变换
            T_cam_ego = extrinsics[cam_id]
            T_ego_world = ego_pose
            T_cam_world = T_ego_world @ T_cam_ego

            # 转换到OpenCV坐标系
            T_cam_world_opencv = WAYMO2OPENCV @ T_cam_world

            all_poses.append(T_cam_world_opencv)
            all_intrinsics.append(intrinsics[cam_id])

    num_views = len(all_poses)
    coverage_matrix = np.zeros((num_views, num_views))

    # 使用多线程计算所有视图对
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_views):
            for j in range(i, num_views):  # 利用对称性
                futures.append(executor.submit(
                    compute_view_overlap,
                    all_poses[i], all_intrinsics[i],
                    all_poses[j], all_intrinsics[j],
                    img_size
                ))

        # 收集结果
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc=f"计算 {sequence}"):
            results.append(future.result())

        # 填充矩阵
        idx = 0
        for i in range(num_views):
            for j in range(i, num_views):
                coverage_matrix[i, j] = results[idx]
                coverage_matrix[j, i] = results[idx]  # 对称性
                idx += 1

    # 保存结果
    output_file = output_dir / f"{sequence}.json"
    coverage_data = {
        'sequence': sequence,
        'num_views': num_views,
        'matrix': coverage_matrix.tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(coverage_data, f, indent=2)

    logger.info(f"场景 {sequence} 的重叠度矩阵已保存到 {output_file}")
    return coverage_data


def precompute_all_coverage(data_root, output_root, stage='train'):
    """
    预计算所有场景的重叠度矩阵
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 获取场景列表
    split_file = data_root / 'splits' / f'{stage}.txt'
    if not split_file.exists():
        logger.error(f"划分文件不存在: {split_file}")
        return

    with open(split_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]

    # 处理每个场景
    for seq in sequences:
        scene_dir = data_root / 'Sprocessed' / seq
        if not scene_dir.exists():
            logger.warning(f"场景目录不存在: {scene_dir}")
            continue

        precompute_sequence_coverage(scene_dir, output_root)


if __name__ == "__main__":
    # 配置参数
    DATA_ROOT = "/home/robot/zyr/waymo"
    OUTPUT_ROOT = "/home/robot/zyr/waymo/coverage"
    STAGE = "train"  # 可以是 "train" 或 "val"

    # 执行预计算
    precompute_all_coverage(DATA_ROOT, OUTPUT_ROOT, STAGE)