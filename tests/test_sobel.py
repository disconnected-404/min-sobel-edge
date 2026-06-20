"""sobel.py 的单元测试"""

import numpy as np
import pytest

from sobel_edge.sobel import (
    SOBEL_X,
    SOBEL_Y,
    _convolve2d,
    classify_direction,
    sobel_edge,
    sobel_gradient_direction,
)


class TestConvolve2d:
    def test_basic_convolution(self):
        """全 1 图像与 Sobel X 核卷积，内部区域应为 0（无水平梯度）"""
        image = np.ones((5, 5), dtype=np.float64)
        result = _convolve2d(image, SOBEL_X)
        assert result.shape == (5, 5)
        # 内部区域（排除 padding 边界）应为 0
        np.testing.assert_array_equal(result[1:-1, 1:-1], 0)

    def test_vertical_edge_detection(self):
        """左黑右白的图像，Sobel X 应检测到垂直边缘"""
        image = np.zeros((5, 5), dtype=np.float64)
        image[:, 3:] = 255.0
        result = _convolve2d(image, SOBEL_X)
        # 边缘位置（列2附近，真实边缘处）应有非零响应
        assert result[2, 2] > 0

    def test_output_shape(self):
        """卷积输出尺寸与输入尺寸一致（zero-padding）"""
        image = np.ones((10, 8), dtype=np.float64)
        result = _convolve2d(image, SOBEL_X)
        assert result.shape == (10, 8)


class TestSobelEdge:
    def test_uniform_image_interior_is_zero(self):
        """均匀图像内部（排除 padding 边界）不应有边缘"""
        image = np.full((10, 10), 128, dtype=np.uint8)
        result = sobel_edge(image)
        assert result.shape == (10, 10)
        # 内部区域（排除 1 圈边界）应全为 0
        np.testing.assert_array_equal(result[1:-1, 1:-1], 0)

    def test_output_dtype_and_range(self):
        """输出应为 uint8，值在 0-255"""
        image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        result = sobel_edge(image)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_threshold_filters_weak_edges(self):
        """阈值应严格减少非零像素"""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 100  # 柔和边缘
        result_no_thresh = sobel_edge(image)
        result_with_thresh = sobel_edge(image, threshold=200)
        # 高阈值下应显著过滤
        assert np.count_nonzero(result_with_thresh) < np.count_nonzero(result_no_thresh)

    def test_binarize_outputs_only_0_and_255(self):
        """binarize=True 时输出只能是 0 或 255"""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255
        result = sobel_edge(image, threshold=50, binarize=True)
        unique_vals = np.unique(result)
        assert all(v in (0, 255) for v in unique_vals)
        assert 255 in unique_vals  # 存在边缘

    def test_binarize_false_preserves_gradient(self):
        """binarize=False 时保留梯度强度"""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255
        result = sobel_edge(image, threshold=50, binarize=False)
        # 中间区域应有介于 0 和 255 之间的值
        nonzero = result[result > 0]
        assert np.any((nonzero > 0) & (nonzero < 255))

    def test_invalid_threshold_raises(self):
        """超出范围的 threshold 应报错"""
        image = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            sobel_edge(image, threshold=300)

    def test_rejects_color_image(self):
        """应拒绝彩色图像输入"""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="灰度图像"):
            sobel_edge(image)

    def test_sharp_edge_detected(self):
        """明显的黑白分界线应被检测到"""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255
        result = sobel_edge(image)
        # 边缘位置（第5列附近）应有非零值
        assert result[5, 4] > 0


class TestSobelGradientDirection:
    def test_vertical_edge_direction(self):
        """垂直边缘的梯度方向应接近水平（0 或 pi）"""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255
        angles = sobel_gradient_direction(image)
        # 取边缘附近的角度（避开 NaN 平坦区域）
        edge_angles = angles[1:-1, 4:6].flatten()  # 边缘附近
        valid = edge_angles[~np.isnan(edge_angles)]
        assert len(valid) > 0
        for a in valid:
            assert abs(a) < 0.5 or abs(abs(a) - np.pi) < 0.5

    def test_uniform_image_interior_is_nan(self):
        """均匀图像内部的梯度方向应全为 NaN（无定义）；padding 边界处有定义是正常的"""
        image = np.full((8, 8), 128, dtype=np.uint8)
        angles = sobel_gradient_direction(image)
        # 内部区域（排除 1 圈边界）应全为 NaN
        assert np.all(np.isnan(angles[1:-1, 1:-1]))

    def test_rejects_color_image(self):
        with pytest.raises(ValueError, match="灰度图像"):
            sobel_gradient_direction(np.zeros((5, 5, 3), dtype=np.uint8))


class TestClassifyDirection:
    def test_horizontal(self):
        """0° 和 180° 附近应分类为水平"""
        angles = np.array([0.0, np.radians(10), np.radians(170)])
        result = classify_direction(angles)
        assert all(r == 0 for r in result)

    def test_vertical(self):
        """90° 附近应分类为垂直"""
        angles = np.array([np.radians(80), np.radians(90), np.radians(100)])
        result = classify_direction(angles)
        assert all(r == 1 for r in result)

    def test_diagonal_45(self):
        """45° 附近应分类为 +45° 对角线"""
        angles = np.array([np.radians(30), np.radians(45), np.radians(60)])
        result = classify_direction(angles)
        assert all(r == 2 for r in result)

    def test_diagonal_135(self):
        """135° 附近应分类为 -45° 对角线"""
        angles = np.array([np.radians(120), np.radians(135), np.radians(150)])
        result = classify_direction(angles)
        assert all(r == 3 for r in result)

    def test_nan_returns_minus_one(self):
        """NaN 输入应分类为 -1（无定义）"""
        angles = np.array([np.nan, 0.0, np.radians(90), np.nan])
        result = classify_direction(angles)
        assert result[0] == -1
        assert result[3] == -1
        assert result[1] == 0
        assert result[2] == 1
