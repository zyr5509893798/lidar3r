# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
from src.mast3r_src.dust3r.dust3r.utils.geometry import colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics  # noqa
from scipy.ndimage import minimum_filter
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


# 更新深度图压缩方法
def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(output_resolution, resample=lanczos if scale_final < 1 else bicubic)
    # 深度图压缩方法改进：
    if depthmap is not None:
        scale_factor = max(output_resolution / input_resolution)
        # 1. 创建有效值掩码（先于任何处理）
        valid_mask = (depthmap > 1e-6).astype(np.uint8)

        # 2. 仅处理有效区域 - 避免零值污染
        valid_depth = depthmap.copy()
        valid_depth[valid_mask == 0] = np.nan  # 将无效区域设为NaN

        # 3. 使用中值滤波替代最小值滤波
        kernel_size = max(1, int(0.5 / scale_factor))
        kernel_size = min(kernel_size, 15)  # 限制最大核尺寸

        # 使用中值滤波处理有效深度
        from scipy.ndimage import median_filter
        filtered_depth = median_filter(valid_depth, size=kernel_size, mode='constant', cval=np.nan)

        # 4. 缩放时使用最近邻插值
        depthmap_scaled = cv2.resize(
            filtered_depth,
            tuple(output_resolution),
            interpolation=cv2.INTER_NEAREST
        )

        # 5. 缩放掩码并重新应用
        mask_scaled = cv2.resize(
            valid_mask.astype(float),
            tuple(output_resolution),
            interpolation=cv2.INTER_NEAREST
        )
        depthmap = np.where(mask_scaled > 0.5, depthmap_scaled, 0)

        # 6. 后处理：移除可能残留的极小值
        depthmap[depthmap < 1e-6] = 0
        # depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
        #                       fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox
