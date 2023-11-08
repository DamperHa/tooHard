import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 路径定义
image_path = '../doc/jpg_exh/1692688072199.jpg'
json_path = '../doc/jpg_exh/1692688072199.json'

# 加载图像
image = Image.open(image_path)

# 加载JSON文件
with open(json_path) as f:
    data = json.load(f)

# 创建图和坐标轴
fig, ax = plt.subplots(1)
ax.imshow(image)

# 在图像上绘制边界框
for item in data['label']:
    # 提取坐标并转换为float类型
    x1, y1, x2, y2 = map(float, item['box2d'])
    # 创建一个矩形patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    # 添加这个矩形到列表中
    ax.add_patch(rect)

# 显示图像和边界框
plt.show()

