import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import utils.loss_mask as loss_mask  # 确保包含视锥体掩码计算函数

# Waymo坐标系到OpenCV坐标系的转换矩阵
WAYMO2OPENCV = np.array([
    [0, -1, 0, 0],  # Waymo Y(左) -> OpenCV X(右)
    [0, 0, -1, 0],  # Waymo Z(上) -> OpenCV Y(下)
    [1, 0, 0, 0],  # Waymo X(前) -> OpenCV Z(前)
    [0, 0, 0, 1]
])


@torch.no_grad()
def calculate_loss_mask(targets, context):
    '''计算目标视图在上下文视图视锥体内的有效掩码'''
    # 从目标视图列表中提取深度图
    target_depth = torch.stack([target_view['depthmap'] for target_view in targets], dim=1)
    # 提取目标视图的内参矩阵
    target_intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in targets], dim=1)
    # 提取目标视图的相机位姿 (c2w)
    target_c2w = torch.stack([target_view['camera_pose'] for target_view in targets], dim=1)

    # 从上下文视图列表中提取深度图
    context_depth = torch.stack([context_view['depthmap'] for context_view in context], dim=1)
    # 提取上下文视图的内参矩阵
    context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in context], dim=1)
    # 提取上下文视图的相机位姿
    context_c2w = torch.stack([context_view['camera_pose'] for context_view in context], dim=1)

    # 确保内参矩阵是3x3形式
    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    # 计算目标视图在上下文视图视锥体内的有效掩码
    mask = loss_mask.calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask


def load_waymo_scene_data(scene_dir, resolution=None):
    """
    加载Waymo场景数据
    :param scene_dir: 场景目录路径 (Path对象)
    :param resolution: 可选的目标分辨率 (height, width)
    :return: 视图列表，每个视图包含深度图、内参、位姿和有效掩码
    """
    views = []
    # 加载相机内参 (5个相机)
    intrinsics = {}
    for cam_id in range(5):
        intr_file = scene_dir / "intrinsics" / f"{cam_id}.txt"
        intrinsics[cam_id] = np.loadtxt(intr_file).reshape(3, 3)[:3, :3]

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

            # ===== 加载深度图 =====
            # 假设深度图存储在scene_dir/"depth"目录下，与图像同名
            depth_path = scene_dir / "depth" / f"{frame_id:06d}_{cam_id}.npy"
            if not depth_path.exists():
                continue

            depth_map = np.load(depth_path)
            # 转换为PyTorch张量并添加批次维度
            depth_tensor = torch.tensor(depth_map).unsqueeze(0).float()

            # ===== 创建有效掩码 =====
            valid_mask = depth_tensor > 1e-6

            # ===== 处理内参 =====
            K = intrinsics[cam_id].copy()
            # 如果指定分辨率，调整内参 (假设深度图需要同样缩放)
            if resolution:
                orig_h, orig_w = depth_map.shape
                new_h, new_w = resolution
                # 计算缩放因子
                scale_x = new_w / orig_w
                scale_y = new_h / orig_h
                # 调整内参
                K[0, :] *= scale_x  # fx, cx
                K[1, :] *= scale_y  # fy, cy
                # 缩放深度图
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor.unsqueeze(0),
                    size=resolution,
                    mode='nearest'
                ).squeeze(0)
                valid_mask = torch.nn.functional.interpolate(
                    valid_mask.float().unsqueeze(0),
                    size=resolution,
                    mode='nearest'
                ).squeeze(0) > 0.5

            # ===== 构建视图字典 =====
            view_data = {
                'depthmap': depth_tensor,
                'valid_mask': valid_mask,
                'camera_intrinsics': torch.tensor(K).float(),
                'camera_pose': torch.tensor(cam_to_world_opencv).float(),
                'frame_id': frame_id,
                'camera_id': cam_id
            }
            views.append(view_data)

    return views


if __name__ == '__main__':
    # 配置参数
    DATA_ROOT = "/path/to/Sprocessed"  # 替换为实际路径
    OUTPUT_DIR = "coverage"
    BATCH_SIZE = 50  # 根据GPU内存调整
    RESOLUTION = (256, 256)  # 处理分辨率 (H,W)，根据需求调整

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

        # 将数据移至GPU
        for view in views:
            for key in ['depthmap', 'valid_mask', 'camera_intrinsics', 'camera_pose']:
                view[key] = view[key].to(device)

        # 存储覆盖率结果
        coverage_matrix = []
        num_views = len(views)

        # 遍历每个视图作为上下文 (context)
        for i in tqdm(range(num_views), desc="Processing context views"):
            context_view = views[i]

            # 分批计算覆盖率
            coverage_vals = []
            for batch_start in range(0, num_views, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_views)
                batch_targets = views[batch_start:batch_end]

                # 计算当前批次的掩码
                masks = calculate_loss_mask(batch_targets, [context_view])

                # 计算覆盖率 (有效像素比例)
                # masks形状: [batch_size, 1, H, W] -> 在空间维度求平均
                batch_coverage = masks.mean(dim=[2, 3]).squeeze(1)
                coverage_vals.append(batch_coverage)

            # 合并批次结果
            coverage_vals = torch.cat(coverage_vals).cpu().numpy()
            coverage_matrix.append(coverage_vals.tolist())

        # 保存结果 (使用序列号作为键)
        result = {scene_id: coverage_matrix}
        with open(output_path, 'w') as f:
            json.dump(result, f)

        print(f"Saved coverage for {scene_id} with {num_views} views")