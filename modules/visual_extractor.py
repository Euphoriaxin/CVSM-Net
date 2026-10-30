# modules/visual_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # 切掉模型最后2层（全连接分类层），只保留特征提取主干
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        # 自适应平均池化，避免固定 kernel_size 带来的尺寸不匹配：把任意尺寸特征图 → 压缩成 1x1 全局特征
        self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        # ================ 1. 处理多视图（双视图胸片）输入 ================
        # 如果输入是 5 维：说明是 [batch, 视图数(2), 通道, 高, 宽]
        if images.dim() == 5:
            batch_size, num_views, C, H, W = images.shape
            # 合并 batch 和 views 维度
            images = images.view(batch_size * num_views, C, H, W)
            patch_feats = self.model(images)  # [batch*views, feat_size, h, w]
            _, feat_size, h, w = patch_feats.shape
            # 拆分！！！恢复视图维度
            # patch_feats[:, 0, :, :, :]   →  正位胸片特征（fixed）
            # patch_feats[:, 1, :, :, :]   →  侧位胸片特征（moving）
            patch_feats = patch_feats.view(batch_size, num_views, feat_size, h, w)
            # ================ 视图融合：两张胸片 → 取平均融合 ================
            # 融合视图：取平均（也可用 max，根据任务选择）
            patch_feats = patch_feats.mean(dim=1)  # [batch, feat_size, h, w]
        else:
            # 如果是普通单视图，直接提取特征
            patch_feats = self.model(images)  # [batch, feat_size, h, w]

        # ================ 2. 提取全局特征 ================
        # 自适应池化 → 1x1 → 展平
        avg_feats = self.avg_fnt(patch_feats).squeeze(-1).squeeze(-1)  # [batch, feat_size]

        # ========= 3. 把特征图转成序列格式（供Transformer解码用） =======
        batch_size, feat_size, h, w = patch_feats.shape
        # reshape + 换维度 → [批次, 补丁数量(7×7=49), 特征维度(2048)]
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        # 返回：局部patch特征 + 全局特征
        return patch_feats, avg_feats