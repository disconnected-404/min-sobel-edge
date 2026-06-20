from sobel_edge import load_image, save_image, sobel_edge, sobel_gradient_direction, classify_direction, compare_plot

# 加载图片（支持文件路径或 numpy 数组）
img = load_image("stones.png")

# 基础边缘检测
edge = sobel_edge(img)

# 带阈值的边缘检测（仅抑制弱边缘）
edge_filtered = sobel_edge(img, threshold=32)

# 带阈值的二值化边缘检测（输出仅 0 或 255）
edge_binary = sobel_edge(img, threshold=32, binarize=True)

# 保存边缘图
save_image(edge_filtered, "edge_output.png")

# 梯度方向（弧度），平坦区域为 NaN
angles = sobel_gradient_direction(img)

# 方向分类：0=水平, 1=垂直, 2=+45°对角线, 3=-45°对角线, -1=无定义
directions = classify_direction(angles)

# 可视化对比（原图 vs 边缘图）
compare_plot(img, edge_filtered, save_path="compare_output.png", show=False)