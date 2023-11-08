
import open3d as o3d
import numpy as np
import json

# 读取点云文件
pcd_path = '../doc/point_exh/1692688072199.pcd'
pcd = o3d.io.read_point_cloud(pcd_path)

# 加载JSON标签
json_path = '../doc/point_exh/1692688072199.json'
with open(json_path, 'r') as f:
    labels = json.load(f)["labels"]

# 为每个标签创建一个3D边界框
for label in labels:
    # 获取标签信息
    center = label["location"]
    extents = label["size"]
    rotation = label["rotation"]  # 注意这个旋转通常是四元数或者欧拉角，需要转换为旋转矩阵
    num_points = label["num_points"]

    # 创建一个3D边界框（这里假设rotation是欧拉角，你可能需要根据数据调整）
    bbox = o3d.geometry.OrientedBoundingBox(center, rotation, extents)

    # 如果需要，可以给边界框上色
    bbox.color = np.array([1, 0, 0])  # 红色

    # 将边界框添加到点云中显示
    pcd += bbox

# 可视化点云和边界框
o3d.visualization.draw_geometries([pcd])
