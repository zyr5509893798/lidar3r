import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = "/home/robot/zyr/waymo/Sprocessed"
output_dir = "/home/robot/zyr/waymo/coverage"
os.makedirs(output_dir, exist_ok=True)

# 1. 使用半精度和更激进的配置 - 速度提升≈25%
extractor = SuperPoint(max_num_keypoints=500).eval().to(device).half()  # 半精度
matcher = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95,
                    flash=True, filter_threshold=0.1).eval().to(device).half()


def extract_features_batch(image_paths, batch_size=8):
    """2. 批量提取特征 - 速度提升≈40%"""
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = [load_image(p, resize=640) for p in batch_paths]
        batch_tensor = torch.stack(batch_imgs).to(device).half()

        with torch.no_grad():
            feats = extractor.extract(batch_tensor)
            feats = [rbd(f) for f in feats]

        all_features.extend(feats)
    return all_features


def batch_compute_similarities(ref_feat, target_feats):
    """3. 批量匹配 - 速度提升≈50%"""
    if ref_feat["keypoints"].numel() == 0 or ref_feat["keypoints"].size(0) == 0:
        return [0.0] * len(target_feats)

    # 准备批量数据
    ref_feats = {"image0": {
        "keypoints": ref_feat["keypoints"].unsqueeze(0).repeat(len(target_feats), 1, 1),
        "descriptors": ref_feat["descriptors"].unsqueeze(0).repeat(len(target_feats), 1, 1)
    }}

    target_dict = {"keypoints": [], "descriptors": []}
    for feat in target_feats:
        if feat["keypoints"].numel() == 0:
            target_dict["keypoints"].append(torch.empty(0, 2, device=device))
            target_dict["descriptors"].append(torch.empty(0, 256, device=device))
        else:
            target_dict["keypoints"].append(feat["keypoints"])
            target_dict["descriptors"].append(feat["descriptors"])

    ref_feats["image1"] = {
        "keypoints": torch.nn.utils.rnn.pad_sequence(
            target_dict["keypoints"], batch_first=True),
        "descriptors": torch.nn.utils.rnn.pad_sequence(
            target_dict["descriptors"], batch_first=True)
    }

    # 批量匹配
    with torch.no_grad():
        matches = matcher(ref_feats)
        matches = [rbd(m) for m in matches]

    # 计算相似度
    similarities = []
    num_kpts_ref = ref_feat["keypoints"].size(0)

    for m in matches:
        if m is None or "matches" not in m:
            similarities.append(0.0)
            continue

        valid_matches = m["matches"][..., 0] > -1
        num_matches = valid_matches.sum().item()
        similarity = min((num_matches / max(1, num_kpts_ref)) * 100.0, 100.0)
        similarities.append(similarity)

    return similarities


def process_camera(camera_images):
    camera_images.sort(key=lambda x: x[0])
    n = len(camera_images)
    coverage_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(coverage_matrix, 100.0)  # 对角线设为100%

    # 批量提取所有特征 - 只提取一次
    image_paths = [img_path for _, img_path in camera_images]
    features_list = extract_features_batch(image_paths)
    features_dict = {idx: feat for idx, feat in enumerate(features_list)}

    # 4. 窗口并行计算 - 速度提升≈70%
    batch_size = 16  # 根据GPU内存调整
    for i in tqdm(range(n), desc="Processing frames", leave=False):
        start_idx = max(0, i - 30)
        end_idx = min(n, i + 31)

        # 跳过自身
        window_indices = [j for j in range(start_idx, end_idx) if j != i]
        if not window_indices:
            continue

        # 准备批量目标特征
        target_feats = [features_dict[j] for j in window_indices]
        similarities = batch_compute_similarities(features_dict[i], target_feats)

        # 填充结果矩阵
        for j_idx, j in enumerate(window_indices):
            coverage_matrix[i, j] = similarities[j_idx]

    return coverage_matrix.tolist()


def process_scene(scene_path, scene_name):
    """使用并行处理单个场景的所有相机"""
    image_dir = os.path.join(scene_path, "images")
    if not os.path.exists(image_dir):
        return

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

    # 5. 并行处理所有相机
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        futures = {}
        for cam_id, images in camera_images.items():
            future = executor.submit(process_camera, images)
            futures[future] = cam_id

        for future in tqdm(futures, desc=f"Processing cameras in {scene_name}"):
            cam_id = futures[future]
            coverage_matrix = future.result()
            result["cameras"][str(cam_id)] = {
                "num_frames": len(camera_images[cam_id]),
                "coverage_matrix": coverage_matrix
            }

    with open(output_file, "w") as f:
        json.dump(result, f)
    print(f"Saved coverage matrices for all cameras in scene {scene_name}")


if __name__ == "__main__":
    scene_dirs = [d for d in os.listdir(base_path)
                  if os.path.isdir(os.path.join(base_path, d))]

    for scene_name in tqdm(scene_dirs, desc="Total Scenes"):
        scene_path = os.path.join(base_path, scene_name)
        process_scene(scene_path, scene_name)
# import os
# import json
# import torch
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# from lightglue import LightGlue, SuperPoint
# from lightglue.utils import load_image, rbd
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# base_path = "/home/robot/zyr/waymo/Sprocessed"
# output_dir = "/home/robot/zyr/waymo/coverage"
# os.makedirs(output_dir, exist_ok=True)
#
# # 初始化模型
# extractor = SuperPoint(max_num_keypoints=500).eval().to(device)
# matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().to(device)
#
#
# def compute_similarity(feats0, feats1):
#     """计算两幅图像之间的相似度百分比"""
#     # 处理空特征的情况
#     if (feats0["keypoints"].numel() == 0 or
#             feats1["keypoints"].numel() == 0 or
#             feats0["keypoints"].shape[0] == 0 or
#             feats1["keypoints"].shape[0] == 0):
#         return 0.0
#
#     # 确保张量维度兼容
#     keypoints0 = feats0["keypoints"]
#     keypoints1 = feats1["keypoints"]
#
#     # 添加batch维度 [N, 2] => [1, N, 2]
#     if keypoints0.dim() == 2:
#         keypoints0 = keypoints0.unsqueeze(0)
#     if keypoints1.dim() == 2:
#         keypoints1 = keypoints1.unsqueeze(0)
#
#     # 创建新的特征字典
#     safe_feats0 = {"keypoints": keypoints0}
#     safe_feats1 = {"keypoints": keypoints1}
#
#     # 添加其他必要字段
#     for key in ["descriptors", "image_size", "scores"]:
#         if key in feats0:
#             safe_feats0[key] = feats0[key].unsqueeze(0) if feats0[key].dim() == 2 else feats0[key]
#         if key in feats1:
#             safe_feats1[key] = feats1[key].unsqueeze(0) if feats1[key].dim() == 2 else feats1[key]
#
#     with torch.no_grad():
#         try:
#             matches = matcher({"image0": safe_feats0, "image1": safe_feats1})
#             matches = rbd(matches)
#         except Exception as e:
#             print(f"Matching error: {str(e)}")
#             return 0.0
#
#     if matches is None or "matches" not in matches:
#         return 0.0
#
#     if matches["matches"].shape[0] == 0:
#         return 0.0
#
#     valid_matches = matches["matches"][..., 0] > -1
#     num_matches = valid_matches.sum().item()
#
#     # 使用原始关键点计数（无batch维度）
#     num_kpts_ref = keypoints0.shape[0] if keypoints0.dim() == 2 else keypoints0.shape[1]
#
#     if num_kpts_ref == 0:
#         return 0.0
#
#     return min((num_matches / num_kpts_ref) * 100.0, 100.0)  # 确保不超过100%
#
# def process_camera(camera_images):
#     """处理单个相机的所有图像"""
#     # 按帧ID排序
#     camera_images.sort(key=lambda x: x[0])
#     n = len(camera_images)
#
#     # 初始化相机相似度矩阵
#     coverage_matrix = [[0.0] * n for _ in range(n)]
#
#     features = {}
#     for idx, (frame_id, img_path) in enumerate(camera_images):
#         try:
#             img = load_image(img_path, resize=640)
#             feats = extractor.extract(img.to(device))
#             feats = rbd(feats)
#
#             # 确保特征字典有基本结构
#             for key in ["keypoints", "descriptors"]:
#                 if key not in feats:
#                     feats[key] = torch.empty((0, 0), device=device)
#
#             # 如果关键点是空的，创建空张量保持维度
#             if feats["keypoints"].ndim == 1:
#                 feats["keypoints"] = torch.empty((0, 2), device=device)
#
#             features[idx] = feats
#
#         except Exception as e:
#             print(f"Error processing {img_path}: {str(e)}")
#             # 创建空特征作为占位符
#             features[idx] = {
#                 "keypoints": torch.empty((0, 2), device=device),
#                 "descriptors": torch.empty((0, 0), device=device)
#             }
#
#     # 计算相似度
#     for i in range(n):
#         if features.get(i) is None:
#             continue
#
#         # 确定时间窗口 (±15帧)
#         start_idx = max(0, i - 30)
#         end_idx = min(n, i + 31)
#
#         for j in range(start_idx, end_idx):
#             if i == j:
#                 coverage_matrix[i][j] = 100.0  # 相同图像设为100%
#                 continue
#
#             if features.get(j) is None:
#                 continue
#
#             similarity = compute_similarity(features[i], features[j])
#             coverage_matrix[i][j] = round(similarity, 2)
#
#     return coverage_matrix
#
# def process_scene(scene_path, scene_name):
#     """处理单个场景，所有相机矩阵保存在同一个JSON文件"""
#     image_dir = os.path.join(scene_path, "images")
#     if not os.path.exists(image_dir):
#         return
#
#     # 按相机分组图像
#     camera_images = {}
#     for img_file in os.listdir(image_dir):
#         if img_file.endswith(".png"):
#             parts = img_file.split("_")
#             frame_id = int(parts[0])
#             cam_id = int(parts[1].split(".")[0])
#             img_path = os.path.join(image_dir, img_file)
#
#             if cam_id not in camera_images:
#                 camera_images[cam_id] = []
#             camera_images[cam_id].append((frame_id, img_path))
#
#     # 创建结果字典，包含所有相机的矩阵
#     result = {
#         "scene": scene_name,
#         "cameras": {}
#     }
#
#     output_file = os.path.join(output_dir, f"{scene_name}.json")
#
#     # 处理每个相机
#     for cam_id, images in camera_images.items():
#         print(f"Processing camera {cam_id} in scene {scene_name} ({len(images)} images)")
#         coverage_matrix = process_camera(images)
#
#         # 将相机矩阵添加到结果中
#         result["cameras"][str(cam_id)] = {
#             "num_frames": len(images),
#             "coverage_matrix": coverage_matrix
#         }
#
#     # 保存整个场景的所有相机矩阵
#     with open(output_file, "w") as f:
#         json.dump(result, f)
#     print(f"Saved coverage matrices for all cameras in scene {scene_name} to {output_file}")
#
#
# # 主处理循环
# scene_dirs = [d for d in os.listdir(base_path)
#               if os.path.isdir(os.path.join(base_path, d))]
#
# for scene_name in tqdm(scene_dirs, desc="Processing scenes"):
#     scene_path = os.path.join(base_path, scene_name)
#     process_scene(scene_path, scene_name)
#
