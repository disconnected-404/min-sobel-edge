"""io.py 的单元测试"""

import numpy as np
import pytest
from pathlib import Path

from sobel_edge.io import load_image, save_image


class TestLoadImage:
    def test_load_from_numpy_2d(self):
        """2D numpy 数组直接返回"""
        arr = np.zeros((10, 10), dtype=np.uint8)
        result = load_image(arr)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8

    def test_load_from_numpy_3d(self):
        """3D 彩色数组转为灰度"""
        arr = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = load_image(arr)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_load_from_float_2d_unit_range(self):
        """float 数组 [0, 1] 应线性映射到 [0, 255]"""
        arr = np.full((8, 8), 0.5, dtype=np.float64)
        result = load_image(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 127 or result[0, 0] == 128

    def test_load_from_float_2d_full_range(self):
        """float 数组超出 [0, 1] 范围时，应裁剪后正常返回"""
        arr = np.full((8, 8), 300.0, dtype=np.float64)
        result = load_image(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 255  # 超过 255 被裁剪

    def test_load_from_numpy_invalid_dims(self):
        """1D 数组应报错"""
        arr = np.zeros(10, dtype=np.uint8)
        with pytest.raises(ValueError, match="不支持的数组维度"):
            load_image(arr)

    def test_load_from_file(self, tmp_path):
        """从文件路径加载图片"""
        from PIL import Image
        img = Image.fromarray(np.full((20, 20), 100, dtype=np.uint8), mode="L")
        file_path = tmp_path / "test.png"
        img.save(file_path)

        result = load_image(str(file_path))
        assert result.shape == (20, 20)
        assert result.dtype == np.uint8

    def test_load_from_pathlib(self, tmp_path):
        """从 Path 对象加载图片"""
        from PIL import Image
        img = Image.fromarray(np.full((10, 10), 50, dtype=np.uint8), mode="L")
        file_path = tmp_path / "test2.png"
        img.save(file_path)

        result = load_image(file_path)
        assert result.shape == (10, 10)

    def test_file_not_found(self):
        """文件不存在应报错"""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image.png")


class TestSaveImage:
    def test_save_creates_file(self, tmp_path):
        """保存图片应创建文件"""
        arr = np.full((10, 10), 128, dtype=np.uint8)
        path = tmp_path / "output.png"
        result = save_image(arr, path)
        assert result.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        """保存时应自动创建父目录"""
        arr = np.full((5, 5), 200, dtype=np.uint8)
        path = tmp_path / "sub" / "dir" / "out.png"
        result = save_image(arr, path)
        assert result.exists()

    def test_save_and_reload_roundtrip(self, tmp_path):
        """保存后重新加载，内容应一致"""
        original = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        path = tmp_path / "roundtrip.png"
        save_image(original, path)
        reloaded = load_image(path)
        np.testing.assert_array_equal(original, reloaded)
