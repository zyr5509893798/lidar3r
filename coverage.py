import os
import json
import torch
import numpy as np
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = "/home/robot/zyr/waymo/Sprocessed"
output_dir = "/home/robot/zyr/waymo/coverage"
os.makedirs(output_dir, exist_ok=True)

# 初始化模型
extractor = SuperPoint(max_num_keypoints=256).eval().to(device)
matcher = LightGlue(
    features="superpoint",
    depth_confidence=0.95,
    width_confidence=0.98,
    flash=True,
    filter_threshold=0.2,
    mp=True
).eval().to(device)

# 启用CUDA优化
torch.backends.cudnn.benchmark = True


def extract_features_batch(image_paths, batch_size=16):
    """批量特征提取函数"""
    features = []
    for img_path in image_paths:
        img = load_image(img_path, resize=512).to(device)
        img = img.unsqueeze(0)  # 添加批次维度 [1, 1, H, W]

        with torch.no_grad(), torch.inference_mode():
            feats = extractor.extract(img)
            features.append(rbd(feats))

    return features


def optimized_matching(ref_feat, target_feats, batch_size=32):
    """更安全的匹配函数 - 解决索引越界问题"""
    if ref_feat["keypoints"].numel() == 0:
        return [0.0] * len(target_feats)

    # 确保参考帧特征有正确的形状
    ref_keypoints = ref_feat["keypoints"].unsqueeze(0)  # [1, N, 2]
    ref_descriptors = ref_feat["descriptors"].unsqueeze(0)  # [1, D, N]

    similarities = []

    # 逐个处理目标帧 - 更安全的方法
    for target_feat in target_feats:
        if target_feat["keypoints"].numel() == 0:
            similarities.append(0.0)
            continue

        # 准备目标特征
        target_keypoints = target_feat["keypoints"].unsqueeze(0)  # [1, M, 2]
        target_descriptors = target_feat["descriptors"].unsqueeze(0)  # [1, D, M]

        # 确保描述子维度一致
        if ref_descriptors.size(1) != target_descriptors.size(1):
            # 如果维度不匹配，调整参考帧描述子
            ref_descriptors = ref_descriptors[:, :target_descriptors.size(1), :]

        # 单个匹配
        with torch.no_grad(), torch.inference_mode():
            try:
                matches = matcher({
                    "image0": {
                        "keypoints": ref_keypoints,
                        "descriptors": ref_descriptors
                    },
                    "image1": {
                        "keypoints": target_keypoints,
                        "descriptors": target_descriptors
                    }
                })

                matches = rbd(matches) if matches is not None else None
            except Exception as e:
                print(f"Matching error: {str(e)}")
                matches = None

        # 计算相似度
        num_kpts_ref = ref_feat["keypoints"].size(0)
        if matches is None or "matches" not in matches:
            similarities.append(0.0)
        else:
            valid_matches = matches["matches"][..., 0] > -1
            num_matches = valid_matches.sum().item()
            sim = min(num_matches / max(1, num_kpts_ref) * 100.0, 100.0)
            similarities.append(sim)

    return similarities


def process_camera(camera_images):
    """处理单个相机"""
    camera_images.sort(key=lambda x: x[0])
    n = len(camera_images)
    coverage_matrix = np.zeros((n, n), dtype=np.float16)
    np.fill_diagonal(coverage_matrix, 100.0)

    # 提取所有特征
    image_paths = [img_path for _, img_path in camera_images]
    features = extract_features_batch(image_paths)
    features_dict = {idx: feat for idx, feat in enumerate(features)}

    # 使用窗口处理
    WINDOW_SIZE = 30
    for i in tqdm(range(n), desc="Processing frames", leave=False):
        start_idx = max(0, i - WINDOW_SIZE)
        end_idx = min(n, i + WINDOW_SIZE + 1)

        # 跳过自身
        window_indices = [j for j in range(start_idx, end_idx) if j != i]
        if not window_indices:
            continue

        # 准备目标特征
        target_feats = [features_dict[j] for j in window_indices]

        # 批量匹配
        similarities = optimized_matching(features_dict[i], target_feats)

        # 填充结果矩阵
        for idx, j in enumerate(window_indices):
            coverage_matrix[i, j] = similarities[idx]

    return coverage_matrix.tolist()


def process_scene(scene_path, scene_name):
    """处理场景"""
    image_dir = os.path.join(scene_path, "images")
    if not os.path.exists(image_dir):
        return

    # 收集相机图像
    camera_images = {}
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".png"):
            parts = img_file.split("_")
            frame_id = int(parts[0])
            cam_id = int(parts[1].split(".")[0])
            img_path = os.path.join(image_dir, img_file)
            camera_images.setdefault(cam_id, []).append((frame_id, img_path))

    output_file = os.path.join(output_dir, f"{scene_name}.json")
    result = {"scene": scene_name, "cameras": {}}

    # 使用线程池处理相机
    with ThreadPoolExecutor(max_workers=min(5, os.cpu_count())) as executor:
        futures = {}
        for cam_id, images in camera_images.items():
            future = executor.submit(process_camera, images)
            futures[future] = cam_id

        for future in tqdm(futures, desc=f"Processing cameras in {scene_name}"):
            cam_id = futures[future]
            try:
                coverage_matrix = future.result()
                result["cameras"][str(cam_id)] = {
                    "num_frames": len(camera_images[cam_id]),
                    "coverage_matrix": coverage_matrix
                }
            except Exception as e:
                print(f"Error processing camera {cam_id} in scene {scene_name}: {str(e)}")

    # 保存结果
    with open(output_file, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"Saved coverage matrices for scene {scene_name}")


if __name__ == "__main__":
    scene_dirs = [d for d in os.listdir(base_path)
                  if os.path.isdir(os.path.join(base_path, d))]

    # 设置线程数
    torch.set_num_threads(2)

    # 添加性能优化
    torch.set_float32_matmul_precision('high')  # 对于Ampere架构GPU

    for scene_name in tqdm(scene_dirs, desc="Total Scenes"):
        scene_path = os.path.join(base_path, scene_name)
        process_scene(scene_path, scene_name)