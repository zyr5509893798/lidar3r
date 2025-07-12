import os
import json
import torch
import numpy as np
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime

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

# 创建错误日志文件
error_log_path = os.path.join(output_dir, "problem_images.csv")
error_log_header = ["scene", "camera", "image_path", "error_type", "first_occurrence"]

# 初始化错误日志
if not os.path.exists(error_log_path):
    with open(error_log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(error_log_header)

# 全局集合记录已报告的问题图像
reported_problem_images = set()


def log_problem_image(scene, camera, img_path, error_type):
    """记录问题图像到CSV文件（每个图像只记录一次）"""
    global reported_problem_images

    # 创建唯一标识符
    identifier = f"{scene}_{camera}_{img_path}"

    # 如果已经报告过，直接返回
    if identifier in reported_problem_images:
        return

    # 添加到已报告集合
    reported_problem_images.add(identifier)

    # 写入文件
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([scene, camera, img_path, error_type, timestamp])

    # 在控制台输出简洁信息
    print(f"标记问题图像: {os.path.basename(img_path)} | {error_type}")


def extract_features_batch(image_paths, scene, camera):
    """批量特征提取函数"""
    features = []
    for img_path in image_paths:
        try:
            img = load_image(img_path, resize=512).to(device)
            img = img.unsqueeze(0)  # 添加批次维度 [1, 1, H, W]

            with torch.no_grad(), torch.inference_mode():
                feats = extractor.extract(img)

                # 关键点数量检查
                if feats["keypoints"].numel() == 0:
                    # 记录问题图像
                    log_problem_image(scene, camera, img_path, "no_keypoints")
                    # 创建空特征
                    feats = {
                        "keypoints": torch.empty((0, 2), device=device),
                        "descriptors": torch.empty((1, 256, 0), device=device),
                        "scores": torch.empty(0, device=device)
                    }
                # 维度标准化
                elif feats["descriptors"].size(1) != 256:
                    # 记录问题图像
                    log_problem_image(scene, camera, img_path, f"dimension_mismatch_{feats['descriptors'].size(1)}")
                    feats = align_features(feats)

                features.append(rbd(feats))
        except Exception as e:
            # 记录问题图像
            log_problem_image(scene, camera, img_path, f"extraction_error_{str(e)}")
            # 创建空特征作为回退
            empty_feats = {
                "keypoints": torch.empty((0, 2), device=device),
                "descriptors": torch.empty((1, 256, 0), device=device),
                "scores": torch.empty(0, device=device)
            }
            features.append(empty_feats)

    return features


def align_features(feats, target_dim=256):
    """对齐特征到目标维度"""
    current_dim = feats["descriptors"].size(1)
    descriptors = feats["descriptors"]

    # 维度不足时填充零
    if current_dim < target_dim:
        pad = torch.zeros(descriptors.size(0), target_dim - current_dim,
                          descriptors.size(2), device=device, dtype=descriptors.dtype)
        feats["descriptors"] = torch.cat([descriptors, pad], dim=1)
    # 维度过多时截断
    elif current_dim > target_dim:
        feats["descriptors"] = descriptors[:, :target_dim, :]

    return feats


def optimized_matching(ref_feat, target_feats, ref_img_path, target_img_paths):
    """安全的匹配函数 - 跳过问题图像对"""
    # 处理空参考特征
    if ref_feat["keypoints"].numel() == 0:
        return [0.0] * len(target_feats)

    # 准备参考帧特征
    ref_keypoints = ref_feat["keypoints"].unsqueeze(0)  # [1, N, 2]
    ref_descriptors = ref_feat["descriptors"].unsqueeze(0)  # [1, D, N]

    # 确保描述符维度为256
    if ref_descriptors.size(1) != 256:
        ref_descriptors = ref_descriptors[:, :256, :]

    similarities = []

    for idx, target_feat in enumerate(target_feats):
        target_img_path = target_img_paths[idx]

        # 处理空目标特征
        if target_feat["keypoints"].numel() == 0:
            similarities.append(0.0)
            continue

        # 准备目标帧特征
        target_keypoints = target_feat["keypoints"].unsqueeze(0)  # [1, M, 2]
        target_descriptors = target_feat["descriptors"].unsqueeze(0)  # [1, D, M]

        # 确保描述符维度为256
        if target_descriptors.size(1) != 256:
            target_descriptors = target_descriptors[:, :256, :]

        # 尝试安全匹配
        try:
            with torch.no_grad(), torch.inference_mode():
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

                # 处理匹配结果
                if matches is None:
                    num_matches = 0
                else:
                    matches = rbd(matches)
                    if "matches" not in matches or matches["matches"].size(0) == 0:
                        num_matches = 0
                    else:
                        num_matches = matches["matches"].size(0)

                # 计算相似度
                num_kpts_ref = ref_feat["keypoints"].size(0)
                sim = min(num_matches / max(1, num_kpts_ref) * 100.0, 100.0) if num_kpts_ref > 0 else 0.0
                similarities.append(sim)

        except Exception as e:
            # 跳过问题图像对，直接返回0
            similarities.append(0.0)

    return similarities


def process_camera(camera_images, scene, camera_id):
    """处理单个相机"""
    camera_images.sort(key=lambda x: x[0])
    n = len(camera_images)
    coverage_matrix = np.zeros((n, n), dtype=np.float16)
    np.fill_diagonal(coverage_matrix, 100.0)

    # 提取图像路径列表
    image_paths = [img_path for _, img_path in camera_images]

    # 提取所有特征（并记录问题图像）
    features = extract_features_batch(image_paths, scene, camera_id)
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

        # 准备目标特征和路径
        target_feats = [features_dict[j] for j in window_indices]
        target_paths = [image_paths[j] for j in window_indices]

        # 批量匹配
        similarities = optimized_matching(
            features_dict[i], target_feats,
            image_paths[i], target_paths
        )

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
            future = executor.submit(process_camera, images, scene_name, cam_id)
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

    # 获取已经处理完成的场景（通过输出文件判断）
    processed_scenes = set()
    for fname in os.listdir(output_dir):
        if fname.endswith(".json"):
            processed_scenes.add(fname.rsplit('.', 1)[0])  # 移除后缀

    # 设置线程数
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')

    for scene_name in tqdm(scene_dirs, desc="Total Scenes"):
        # 跳过已处理的场景
        if scene_name in processed_scenes:
            print(f"跳过已处理场景: {scene_name}")
            continue

        scene_path = os.path.join(base_path, scene_name)
        process_scene(scene_path, scene_name)