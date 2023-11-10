from mmdet3d.models import Base3DDetector
from mmdet.models.builder import build_backbone, build_neck, build_head

class MultiModalDetector(Base3DDetector):
    def __init__(self, img_backbone, pc_backbone, img_bbox_head, pc_bbox_head, neck=None, train_cfg=None, test_cfg=None,
                 pretrained=None):
        super(MultiModalDetector, self).__init__()

        # 创建图像模态的骨干网络
        self.img_backbone = build_backbone(img_backbone)

        # 创建点云模态的骨干网络
        self.pc_backbone = build_backbone(pc_backbone)

        # 如果需要，可以添加颈部
        self.neck = build_neck(neck) if neck is not None else None

        # 创建图像和点云的检测头
        self.img_bbox_head = build_head(img_bbox_head) if img_bbox_head is not None else None
        self.pc_bbox_head = build_head(pc_bbox_head) if pc_bbox_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 初始化权重
        self.init_weights(pretrained=pretrained)

    def forward(self, img_data, pc_data, return_loss=True, **kwargs):
        # 提取图像特征
        img_feats = self.img_backbone(img_data)

        # 提取点云特征
        pc_feats = self.pc_backbone(pc_data)

        # 如果需要，将图像和点云的特征进行融合
        # fused_feats = fuse_features(img_feats, pc_feats)

        # 如果有颈部，则应用颈部
        if self.neck is not None:
            img_feats = self.neck(img_feats)
            pc_feats = self.neck(pc_feats)

        # 合并多模态特征
        # merged_feats = merge_features(img_feats, pc_feats)

        # 将合并后的特征传递给图像和点云的检测头
        img_outs = self.img_bbox_head(img_feats)
        pc_outs = self.pc_bbox_head(pc_feats)

        return img_outs, pc_outs
