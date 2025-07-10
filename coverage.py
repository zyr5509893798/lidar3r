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

logger = logging.getLogger(__name__)

# Waymo到OpenCV坐标系的转换矩阵
WAYMO2OPENCV = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


class WaymoData:
    def __init__(self, root, stage):
        self.root = root
        self.stage = stage
        self.coverage = {}  # 新增：存储覆盖度矩阵

        # 数据结构
        self.color_paths = {}
        self.depth_paths = {}
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
            required_dirs = ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar_depth']
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
                if len(params) >= 4:
                    f_u, f_v, c_u, c_v = params[:4]
                    K = np.array([
                        [f_u, 0, c_u],
                        [0, f_v, c_v],
                        [0, 0, 1]
                    ], dtype=np.float32)
                else:
                    raise ValueError(f"Invalid intrinsic parameters: {params}")
                cam_intrinsics[cam_id] = K

                # 外参
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

            # 获取所有帧的ID
            pose_files = sorted((scene_dir / 'ego_pose').glob('*.txt'))
            pose_files = [f for f in pose_files if '_' not in f.stem]

            if not pose_files:
                logger.warning(f"场景 {seq} 没有ego位姿文件")
                scenes_with_no_frames.append(seq)
                continue

            frame_ids = [int(f.stem) for f in pose_files]

            # 为场景存储数据
            scene_color_paths = []
            scene_depth_paths = []
            scene_intrinsics = []
            scene_c2ws = []

            for frame_id in frame_ids:
                # 加载ego位姿
                ego_pose_file = scene_dir / 'ego_pose' / f'{frame_id:06d}.txt'
                if not ego_pose_file.exists():
                    logger.warning(f"场景 {seq} 帧 {frame_id} 缺少ego位姿，跳过")
                    continue

                ego_pose = np.loadtxt(ego_pose_file).reshape(4, 4).astype(np.float32)

                # 处理每个相机
                for cam_id in range(5):
                    img_file = scene_dir / 'images' / f'{frame_id:06d}_{cam_id}.png'
                    depth_file = scene_dir / 'lidar_depth' / f'{frame_id:06d}_{cam_id}.npy'

                    if not img_file.exists():
                        logger.warning(f"场景 {seq} 帧 {frame_id} 相机 {cam_id} 缺少图像，跳过")
                        continue

                    if not depth_file.exists():
                        logger.warning(f"场景 {seq} 帧 {frame_id} 相机 {cam_id} 缺少深度图，跳过")
                        continue

                    # 计算相机到世界的变换
                    T_cam_ego = cam_extrinsics[cam_id]
                    T_ego_world = ego_pose
                    T_cam_world = T_ego_world @ T_cam_ego

                    # 转换到OpenCV坐标系
                    T_cam_world_opencv = WAYMO2OPENCV @ T_cam_world

                    # 存储数据
                    scene_color_paths.append(str(img_file))
                    scene_depth_paths.append(str(depth_file))
                    scene_intrinsics.append(cam_intrinsics[cam_id].copy())
                    scene_c2ws.append(T_cam_world_opencv)

            if not scene_color_paths:
                logger.warning(f"场景 {seq} 没有有效帧，跳过")
                scenes_with_no_frames.append(seq)
                continue

            self.color_paths[seq] = scene_color_paths
            self.depth_paths[seq] = scene_depth_paths
            self.intrinsics[seq] = scene_intrinsics
            self.c2ws[seq] = scene_c2ws

        # 更新有效场景列表
        self.sequences = [seq for seq in self.sequences if seq not in scenes_with_no_frames]
        logger.info(f"成功加载 {len(self.sequences)} 个场景的数据")

        # 为每个场景计算覆盖度矩阵
        if stage in ['train', 'val']:
            logger.info("开始计算场景覆盖度矩阵...")
            for seq in self.sequences:
                self._compute_coverage_matrix(seq)
            logger.info("覆盖度矩阵计算完成")

    def _compute_coverage_matrix(self, sequence):
        """计算指定序列的覆盖度矩阵"""
        n_frames = len(self.color_paths[sequence])
        coverage_matrix = np.zeros((n_frames, n_frames), dtype=np.float32)

        # 获取图像尺寸（假设所有帧尺寸相同）
        img0 = imread_cv2(self.color_paths[sequence][0])
        img_size = (img0.shape[1], img0.shape[0])  # (width, height)

        # 使用多线程加速计算
        with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = []

            # 为每个帧对提交任务（只计算相邻帧）
            for i in range(n_frames):
                # 计算相邻帧范围（前后30帧）
                start_j = max(0, i - 30)
                end_j = min(n_frames, i + 31)

                for j in range(start_j, end_j):
                    if i == j:
                        coverage_matrix[i, j] = 1.0  # 自身重叠度为1
                        continue

                    # 提交计算任务
                    futures.append(executor.submit(
                        self._calculate_frame_overlap,
                        sequence, i, j, img_size
                    ))

            # 收集结果
            for future in as_completed(futures):
                i, j, overlap = future.result()
                coverage_matrix[i, j] = overlap
                coverage_matrix[j, i] = overlap  # 对称赋值

        self.coverage[sequence] = coverage_matrix

    def _calculate_frame_overlap(self, sequence, i, j, img_size):
        """计算两帧之间的重叠度"""
        # 快速检查：如果帧索引相同则返回1.0
        if i == j:
            return i, j, 1.0

        # 获取相机参数
        K1 = self.intrinsics[sequence][i]
        K2 = self.intrinsics[sequence][j]
        c2w1 = self.c2ws[sequence][i]
        c2w2 = self.c2ws[sequence][j]

        # 提取光心位置
        c1 = c2w1[:3, 3]
        c2 = c2w2[:3, 3]

        # 快速排除：距离过远
        dist = np.linalg.norm(c1 - c2)
        if dist > 50.0:  # 50米阈值
            return i, j, 0.0

        # 快速排除：视锥方向相反
        view_dir1 = c2w1[:3, 2] / np.linalg.norm(c2w1[:3, 2])
        view_dir2 = c2w2[:3, 2] / np.linalg.norm(c2w2[:3, 2])
        if np.dot(view_dir1, view_dir2) < 0.0:  # 夹角大于90度
            return i, j, 0.0

        # 计算相对位姿
        w2c1 = np.linalg.inv(c2w1)
        R_rel = w2c1[:3, :3] @ c2w2[:3, :3]
        t_rel = w2c1[:3, :3] @ c2w2[:3, 3] + w2c1[:3, 3]

        # 计算基线长度
        baseline = np.linalg.norm(t_rel)
        if baseline < 1e-6:  # 避免除零
            return i, j, 0.0

        # 设置深度采样平面（基于基线动态调整）
        depth_planes = [
            0.1 * baseline,
            1.0 * baseline,
            5.0 * baseline,
            50.0 * baseline
        ]

        # 计算每个深度平面上的重叠区域
        overlap_areas = []
        for depth in depth_planes:
            # 计算帧1的视锥投影
            poly1 = self._project_frustum(K1, depth, img_size)

            # 计算帧2的视锥投影（转换到帧1的坐标系）
            poly2 = self._project_frustum(K2, depth, img_size)
            poly2 = self._transform_points(poly2, R_rel, t_rel)

            # 计算两个多边形的交集面积
            inter_area = self._polygon_intersection_area(poly1, poly2)
            if inter_area > 0:
                overlap_areas.append(inter_area)

        # 如果没有有效的重叠区域，返回0
        if not overlap_areas:
            return i, j, 0.0

        # 计算平均重叠面积
        avg_overlap_area = np.mean(overlap_areas)
        img_area = img_size[0] * img_size[1]

        # 计算重叠度（限制在0-1之间）
        overlap = min(avg_overlap_area / img_area, 1.0)
        return i, j, overlap

    def _project_frustum(self, K, depth, img_size):
        """计算在指定深度平面上的视锥投影"""
        w, h = img_size
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 计算四个角点（在相机坐标系）
        points = np.array([
            [0, 0],  # 左上
            [w, 0],  # 右上
            [w, h],  # 右下
            [0, h]  # 左下
        ], dtype=np.float32)

        # 转换到3D空间（在指定深度平面）
        points_3d = np.zeros((4, 3), dtype=np.float32)
        points_3d[:, 0] = (points[:, 0] - cx) * depth / fx  # X
        points_3d[:, 1] = (points[:, 1] - cy) * depth / fy  # Y
        points_3d[:, 2] = depth  # Z

        return points_3d

    def _transform_points(self, points, R, t):
        """使用相对位姿变换点集"""
        # 应用旋转
        rotated = points @ R.T

        # 应用平移
        transformed = rotated + t[np.newaxis, :]

        return transformed

    def _polygon_intersection_area(self, poly1, poly2):
        """计算两个多边形的交集面积（凸包近似）"""
        try:
            # 将两个多边形的点合并
            all_points = np.vstack([poly1, poly2])

            # 计算凸包
            hull = ConvexHull(all_points)

            # 提取凸包点
            hull_points = all_points[hull.vertices]

            # 计算凸包面积作为交集近似
            return hull.volume
        except:
            # 凸包计算失败时返回0
            return 0.0

    def get_view(self, sequence, view_idx, resolution):
        if sequence not in self.color_paths:
            raise ValueError(f"无效场景: {sequence}")

        if view_idx >= len(self.color_paths[sequence]):
            raise ValueError(f"无效视图索引: {view_idx} (最大 {len(self.color_paths[sequence]) - 1})")

        # 读取图像
        img_path = self.color_paths[sequence][view_idx]
        rgb_image = imread_cv2(img_path)

        # 新增：读取深度图
        depth_path = self.depth_paths[sequence][view_idx]
        depth_data = np.load(depth_path, allow_pickle=True).item()
        depth_map = reconstruct_depth_map(depth_data, rgb_image.shape[:2])  # 原始图像形状 (H, W)

        # 获取内参和位姿
        intrinsics = self.intrinsics[sequence][view_idx]
        c2w = self.c2ws[sequence][view_idx]

        # 调整大小 (同时处理图像和深度图)
        rgb_image, depth_map, intrinsics = crop_resize_if_necessary(
            rgb_image, depth_map, intrinsics, resolution
        )

        # 创建有效掩码和天空掩码
        valid_mask = depth_map > 1e-6
        sky_mask = depth_map <= 0.0

        # 标准化并转换为伪RGB
        depth_rgb = normalize_depth_map(depth_map)

        return {
            'original_img': rgb_image,
            'depthmap': depth_rgb,  # 现在返回真实的深度图(标准化之后）
            'camera_pose': c2w,
            'camera_intrinsics': intrinsics,
            'dataset': 'waymo',
            'label': f"waymo/{sequence}",
            'instance': f'{view_idx}',
            'is_metric_scale': True,
            'sky_mask': sky_mask,  # 新增天空掩码
            'valid_mask': valid_mask  # 新增有效深度掩码
        }

def get_waymo_dataset(root, stage, resolution, num_epochs_per_epoch=1):
    data = WaymoData(root, stage)

    # 确保覆盖度矩阵已计算
    if stage in ['train', 'val'] and not data.coverage:
        logger.warning("覆盖度矩阵未计算，将重新计算...")
        for seq in data.sequences:
            data._compute_coverage_matrix(seq)

    dataset = DUST3RSplattingDataset(
        data,
        resolution,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )

    return dataset

if __name__ == '__main__':
    # 配置参数
    DATA_ROOT = "/home/robot/zyr/waymo"  # 替换为实际路径
    OUTPUT_DIR = "/home/robot/zyr/waymo/coverage"
    BATCH_SIZE = 5  # 根据GPU内存调整/home/robot/mfx
    RESOLUTION = (1920, 1280)  # 处理分辨率 (width, height)

    # 创建覆盖度矩阵保存目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"覆盖度矩阵将保存到: {OUTPUT_DIR}")

    # 处理训练集和验证集
    for stage in ['train']:
        logger.info(f"处理 {stage} 数据集...")
        data = WaymoData(DATA_ROOT, stage)

        # 确保覆盖度矩阵已计算
        if not data.coverage:
            logger.warning(f"{stage} 数据集覆盖度矩阵未计算，将重新计算...")
            for seq in data.sequences:
                data._compute_coverage_matrix(seq)

                # 保存覆盖度矩阵到JSON文件
                if seq in data.coverage:
                    coverage_matrix = data.coverage[seq]

                    # 转换为列表（JSON可序COVERAGE_DIR列化）
                    coverage_list = coverage_matrix.tolist()

                    # 构建保存数据结构
                    save_data = {seq: coverage_list}

                    # 保存到文件
                    save_path = os.path.join(OUTPUT_DIR, f"{seq}.json")
                    with open(save_path, 'w') as f:
                        json.dump(save_data, f, indent=2)

                    logger.info(f"已保存 {seq} 的覆盖度矩阵: {save_path}")
                else:
                    logger.warning(f"场景 {seq} 没有覆盖度矩阵，跳过保存")

    logger.info("覆盖度矩阵保存完成")

    # data = WaymoData(DATA_ROOT, 'test')
    #
    # # 确保覆盖度矩阵已计算
    # if stage in ['train', 'val'] and not data.coverage:
    #     logger.warning("覆盖度矩阵未计算，将重新计算...")
    #     for seq in data.sequences:
    #         data._compute_coverage_matrix(seq)




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
    # return np.moveaxis(rgb, -1, 0) # 3 H W，我们最终需要的格式。