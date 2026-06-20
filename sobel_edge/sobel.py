"""Sobel 边缘检测核心算法（纯 numpy 实现）"""

import numpy as np


# Sobel 卷积核
SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对灰度图像进行 2D 卷积（zero-padding，输出尺寸与输入一致），使用 numpy 切片加速"""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # zero-padding 保持输出尺寸与输入相同
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    oh, ow = image.shape

    output = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i+oh, j:j+ow]
    return output


def sobel_edge(
    image: np.ndarray,
    threshold: int | None = None,
    binarize: bool = False,
) -> np.ndarray:
    """对灰度图像执行 Sobel 边缘检测。

    Args:
        image: 输入灰度图像（2D numpy 数组，值域 0-255）
        threshold: 可选阈值（0-255），低于此值的梯度置为 0。默认 None（不设阈值）
        binarize: 是否输出二值图。必须与 threshold 同时使用；
                  threshold=None 时传 binarize=True 会抛出 ValueError。
                  True → 输出二值图（0 或 255）；
                  False → 仅抑制弱边缘，保留梯度强度。默认 False

    Returns:
        边缘梯度幅值图（uint8，0-255）。均匀/平坦区域的 padding 噪声会被抑制。
    """
    if image.ndim != 2:
        raise ValueError("输入必须是灰度图像（2D 数组）")

    if threshold is not None and not (0 <= threshold <= 255):
        raise ValueError("threshold 应在 [0, 255] 范围内")

    if binarize and threshold is None:
        raise ValueError("binarize=True 需要同时指定 threshold")

    img = image.astype(np.float64)

    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # 归一化到 0-255；若最大响应极小（均匀图的 padding 噪声），直接归零
    max_val = magnitude.max()
    if max_val > 1e-6:
        magnitude = magnitude / max_val * 255.0
    else:
        magnitude[:] = 0.0

    if threshold is not None:
        if binarize:
            magnitude = np.where(magnitude >= threshold, 255.0, 0.0)
        else:
            magnitude[magnitude < threshold] = 0.0

    return magnitude.astype(np.uint8)


def sobel_gradient_direction(image: np.ndarray) -> np.ndarray:
    """计算 Sobel 梯度方向（弧度）。

    Args:
        image: 输入灰度图像（2D numpy 数组）

    Returns:
        梯度方向角（弧度，范围 [-pi, pi]），与边缘幅值图同尺寸。
        平坦区域（gx 与 gy 均接近 0）对应方向无定义，标记为 NaN。
    """
    if image.ndim != 2:
        raise ValueError("输入必须是灰度图像（2D 数组）")

    img = image.astype(np.float64)

    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)

    angles = np.arctan2(gy, gx)

    # 平坦区域（gx, gy 均近零）方向无定义，标记为 NaN
    flat_mask = (np.abs(gx) < 1e-6) & (np.abs(gy) < 1e-6)
    angles[flat_mask] = np.nan

    return angles


def classify_direction(angles: np.ndarray) -> np.ndarray:
    """将梯度方向角离散分类为 4 个方向。

    Args:
        angles: arctan2 返回的梯度方向角（弧度）。NaN 视为无定义。

    Returns:
        分类标签数组（int8）：0=水平, 1=垂直, 2=对角线(+45°), 3=对角线(-45°)。
        NaN 输入位置返回 -1。
    """
    if angles.ndim < 1:
        raise ValueError("angles 必须是非空的 numpy 数组")

    # 先记录 NaN 位置，再对有效值做归一化分类
    nan_mask = np.isnan(angles)
    deg = np.degrees(np.where(nan_mask, 0.0, angles)) % 180

    result = np.zeros_like(deg, dtype=np.int8)

    # 水平方向 (0° ± 22.5°)
    result[(deg >= 0) & (deg < 22.5)] = 0
    result[(deg >= 157.5) & (deg < 180)] = 0

    # 垂直方向 (90° ± 22.5°)
    result[(deg >= 67.5) & (deg < 112.5)] = 1

    # +45° 对角线
    result[(deg >= 22.5) & (deg < 67.5)] = 2

    # -45° 对角线
    result[(deg >= 112.5) & (deg < 157.5)] = 3

    # 无定义位置标记为 -1
    result[nan_mask] = -1

    return result
