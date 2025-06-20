import cv2
import numpy as np


def combine_and_save_masks(mask1_path, mask2_path, save_path):
    # 读取两张掩码图像
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 灰度图并归一化
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 灰度图并归一化

    if mask1 is None or mask2 is None:
        raise ValueError("Failed to load one or both of the mask images.")

    # 确保两张掩码图像的尺寸一致
    if mask1.shape != mask2.shape:
        raise ValueError("The two mask images must have the same shape.")

    # 对应像素位置进行逻辑 OR 操作，重合掩码
    combined_mask = np.maximum(mask1, mask2)  # 使用 numpy 的 maximum 函数进行元素-wise 最大值操作

    # 将合并后的掩码图像保存为图像文件
    cv2.imwrite(save_path, (combined_mask * 255).astype(np.uint8))  # 乘以255转换为[0, 255]范围并保存为图片

    print(f"Combined mask saved to {save_path}")


# 示例调用
mask1_path = '/mnt/data/JZD/3D2Fool/man/mask_half_sleeve.jpg'
mask2_path = '/mnt/data/JZD/3D2Fool/man/mask_jeans.jpg'
save_path = '/mnt/data/JZD/3D2Fool/man/mask.jpg'

combine_and_save_masks(mask1_path, mask2_path, save_path)
