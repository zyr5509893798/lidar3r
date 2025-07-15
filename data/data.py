import random

import numpy as np
import PIL
import torch
import torchvision

from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
from src.mast3r_src.dust3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf
from src.mast3r_src.dust3r.dust3r.utils.misc import invalid_to_zeros
import src.mast3r_src.dust3r.dust3r.datasets.utils.cropping as cropping


def crop_resize_if_necessary(image, depthmap, intrinsics, resolution):
    """Adapted from DUST3R's Co3D dataset implementation"""

    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # Downscale with lanczos interpolation so that image.size == resolution cropping centered on the principal point
    # The new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
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


class DUST3RSplattingDataset(torch.utils.data.Dataset):

    def __init__(self, data, coverage, resolution, num_epochs_per_epoch=1, alpha=0.3, beta=0.3):

        super(DUST3RSplattingDataset, self).__init__()
        self.data = data

        self.num_context_views = 2
        self.num_target_views = 3
        self.coverage = coverage
        self.resolution = resolution
        self.transform = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()
        self.num_epochs_per_epoch = num_epochs_per_epoch

        self.alpha = alpha
        self.beta = beta

    def __getitem__(self, idx):
        # 返回views，这里面都是真数据而不是路径和id，结构：
        # views = {
        #   "context": [
        #       {
        #             'original_img': rgb_image,
        #             'img': 处理后的图像,
        #             'pts3d': 每个像素位置在世界坐标系下的三维点云,
        #             'valid_mask': 点云有效性掩码,
        #             'depthmap': depthmap,
        #             'camera_pose': c2w,
        #             'camera_intrinsics': intrinsics,
        #             'dataset': 'scannet++',
        #             'label': f"scannet++/{sequence}",
        #             'instance': f'{view_idx}',
        #             'is_metric_scale': True,
        #             'sky_mask': depthmap <= 0.0,
        #         }, {……} ], 一共两个，图一图二
        #   "target": [{
        #             'original_img': rgb_image,
        #             'depthmap': depthmap,
        #             'camera_pose': c2w,
        #             'camera_intrinsics': intrinsics,
        #             'dataset': 'scannet++',
        #             'label': f"scannet++/{sequence}",
        #             'instance': f'{view_idx}',
        #             'is_metric_scale': True,
        #             'sky_mask': depthmap <= 0.0,
        #         }, ……],
        #   "scene": sequence}
        sequence = self.data.sequences[idx // self.num_epochs_per_epoch]
        sequence_length = len(self.data.color_paths[sequence])

        # 确定序列id之后，选择图1，图2，对照测试图
        context_views, target_views, camera_id = self.sample(sequence, self.num_target_views, self.alpha, self.beta)

        views = {"context": [], "target": [], "scene": sequence}

        # Fetch the context views
        for c_view in context_views:
            # 图1和图2需要取出图像处理，取出深度图得到对应的世界坐标系下的三维点云和有效性掩码，把这些放入context
            assert c_view < sequence_length, f"Invalid view index: {c_view}, sequence length: {sequence_length}, c_views: {context_views}"
            # 这里的c_view改为代表随机选取的那一帧，camera_id代表我们选择这一帧的哪个相机视角
            # 对每次选择context和target对，我们只会选择同一个相机视角（不同视角重叠度太低）
            view = self.data.get_view(sequence, c_view, camera_id, self.resolution) # 从各种路径和id中取真实的这一帧的图像，深度图，ctw等

            # Transform the input
            view['img'] = self.transform(view['original_img']) # 图像转换
            view['original_img'] = self.org_transform(view['original_img'])

            # Create the point cloud and validity mask
            # pts3d：世界坐标系下的点云图，每个像素对应一个xyz，HxWx3
            # valid_mask：每个像素位置的点是否有效
            # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            # view['pts3d'] = pts3d
            # view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)  # 确保一个点的所有坐标均有限，得到 HxW 掩码。
            assert view['valid_mask'].any(), f"Invalid mask for sequence: {sequence}, view: {c_view}"

            views['context'].append(view)

        # Fetch the target views
        for t_view in target_views:
            # 参照图像只需要对原图进行简单处理。
            view = self.data.get_view(sequence, t_view, camera_id, self.resolution)
            view['original_img'] = self.org_transform(view['original_img'])
            views['target'].append(view)

        return views

    def __len__(self):

        return len(self.data.sequences) * self.num_epochs_per_epoch

    def sample(self, sequence, num_target_views, context_overlap_threshold=0.5, target_overlap_threshold=0.6):
# 魔改sample，暂时失去了作用，只随便挑一张图。     测试。
# 魔改不行，重新搞

        # 注意这里改了之后，view的id选择只是选择了随机一帧，另外要选择随机一个相机视角
        first_context_view = random.randint(0, len(self.data.color_paths[sequence]) - 1) # 随便选图1
        camera_id = random.randint(0, 4) # 随机选相机视角

        # Pick a second context view that has sufficient overlap with the first context view
        valid_second_context_views = []
        for frame in range(len(self.data.color_paths[sequence])):
            if frame == first_context_view:
                continue
            overlap = self.coverage[sequence][camera_id][first_context_view][frame] # 重叠度矩阵多了一维[camera_id]
            # coverage[sequence]帧间重叠度矩阵，通过 coverage[sequence][i][j] 可直接访问帧 i 和帧 j 的重叠度。
            if overlap > context_overlap_threshold:
                valid_second_context_views.append(frame)
                # 将所有重叠度满足要求的帧加入候选序列
        if len(valid_second_context_views) > 0: # 从所有满足要求的帧中随机选择一个作为图2
            second_context_view = random.choice(valid_second_context_views)

        # If there are no valid second context views, pick the best one
        else: # 没满足要求的，选最好的一个
            best_view = None
            best_overlap = None
            for frame in range(len(self.data.color_paths[sequence])):
                if frame == first_context_view:
                    continue
                overlap = self.coverage[sequence][camera_id][first_context_view][frame] # 加上[camera_id]
                if best_view is None or overlap > best_overlap:
                    best_view = frame
                    best_overlap = overlap
            second_context_view = best_view

        # Pick the target views
        valid_target_views = []  # 在同一个序列中选择测试帧，用于最终的对照
        for frame in range(len(self.data.color_paths[sequence])):
            if frame == first_context_view:
                continue
            overlap_max = max(   # 测试帧要与至少一个输入图有一定的重合度
                self.coverage[sequence][camera_id][first_context_view][frame],
                self.coverage[sequence][camera_id][second_context_view][frame]
            )
            if overlap_max > target_overlap_threshold:
                valid_target_views.append(frame)
        if len(valid_target_views) >= num_target_views:
            target_views = random.sample(valid_target_views, num_target_views)

        # If there are not enough valid target views, pick the best ones
        else:   # 没有符合要求的就选最合适的。
            overlaps = []
            for frame in range(len(self.data.color_paths[sequence])):
                if frame == first_context_view or frame == second_context_view:
                    continue
                overlap = max(
                    self.coverage[sequence][camera_id][first_context_view][frame], # 都加上[camera_id]
                    self.coverage[sequence][camera_id][second_context_view][frame]
                )
                overlaps.append((frame, overlap))
            overlaps.sort(key=lambda x: x[1], reverse=True)
            target_views = [frame for frame, _ in overlaps[:num_target_views]]

        return [first_context_view, second_context_view], target_views, camera_id

        # return [first_context_view], [first_context_view]  # 先全用一张图和它自己测试模型


class DUST3RSplattingTestDataset(torch.utils.data.Dataset):

    def __init__(self, data, samples, resolution):

        self.data = data
        self.samples = samples

        self.resolution = resolution
        self.transform = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()

    def get_view(self, sequence, c_view, camera_id):

        view = self.data.get_view(sequence, c_view, camera_id, self.resolution)

        # Transform the input
        view['img'] = self.transform(view['original_img'])
        view['original_img'] = self.org_transform(view['original_img'])

        # Create the point cloud and validity mask
        # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        # view['pts3d'] = pts3d
        # view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
        assert view['valid_mask'].any(), f"Invalid mask for sequence: {sequence}, view: {c_view}"

        return view

    def __getitem__(self, idx):

        sequence, c_view_1, c_view_2, target_view, camera_id = self.samples[idx]
        # sequence, c_view_1 = self.samples[idx]
        # 这里修改成合适的结构，包括view2和target view和camera_id
        c_view_1, c_view_2, target_view, camera_id = int(c_view_1), int(c_view_2), int(target_view), int(camera_id)
        # c_view_1 = int(c_view_1)

        fetched_c_view_1 = self.get_view(sequence, c_view_1, camera_id)
        fetched_c_view_2 = self.get_view(sequence, c_view_2, camera_id)
        fetched_target_view = self.get_view(sequence, target_view, camera_id)

        views = {"context": [fetched_c_view_1, fetched_c_view_2], "target": [fetched_target_view], "scene": sequence}

        return views

    def __len__(self):

        return len(self.samples)
