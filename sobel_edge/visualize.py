"""可视化对比模块"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compare_plot(
    original: np.ndarray,
    edge: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """生成原图与边缘检测结果的并排对比图。

    Args:
        original: 原始灰度图像（2D numpy 数组）
        edge: 边缘检测结果（2D numpy 数组）
        save_path: 可选，保存图片的路径
        show: 是否显示窗口（默认 False，避免脚本/CI 环境弹窗）。
              若后端为非交互式（如 Agg），即使 show=True 也不会弹窗。

    Returns:
        matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(edge, cmap="gray")
    axes[1].set_title("Sobel Edge")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
