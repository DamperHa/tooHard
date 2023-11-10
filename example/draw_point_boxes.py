
import open3d as o3d
import numpy as np
import json
import transforms3d as tf3d
from open3d import geometry

# 读取点云文件
pcd_path = '../doc/point_exh/1692688072199.pcd'
pcd = o3d.io.read_point_cloud(pcd_path)

# 加载JSON标签
json_path = '../doc/point_exh/1692688072199.json'
with open(json_path, 'r') as f:
    labels = json.load(f)["labels"]
#

vis = o3d.visualization.Visualizer()

# # 为每个标签创建一个3D边界框
for label in labels:
    # 获取标签信息
    center = label["location"]
    extents = label["size"]
    rotation = label["rotation"]  # 注意这个旋转通常是四元数或者欧拉角，需要转换为旋转矩阵
    num_points = label["num_points"]


    # 创建一个3D边界框（这里假设rotation是欧拉角，你可能需要根据数据调整）
    bbox = o3d.geometry.OrientedBoundingBox(center, tf3d.euler.euler2mat(rotation[0], rotation[1], rotation[2]), extents)

    line_set = geometry.LineSet.create_from_oriented_bounding_box(bbox)
    #line_set.paint_uniform_color((0.5, 0.5, 0.5))
    # 在 Visualizer 中绘制 Box

    vis.add_geometry(line_set)

vis.add_geometry(pcd)
vis.run()
    # 可视化点云和边界框
