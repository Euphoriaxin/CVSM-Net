# models/models.py
import numpy as np
import torch
import torch.nn as nn
from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.crft_align import CRFTImageAligner  # 导入CRFT对齐模块


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # 视觉特征提取器
        self.visual_extractor = VisualExtractor(args)

        # CRFT图像对齐模块
        self.use_crft = getattr(args, 'use_crft', True)
        if self.use_crft:
            # 使用视觉特征维度d_vf（2048）而不是d_model（512）
            self.crft_aligner = CRFTImageAligner(embed_dim=args.d_vf)
            print(f"⚡ CRFT图像对齐模块已启用，输入维度: {args.d_vf}")

        # CMN编码器-解码器
        self.encoder_decoder = BaseCMN(args, tokenizer)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        crft_params = 0
        if self.use_crft:
            crft_params = sum([np.prod(p.size()) for p in self.crft_aligner.parameters()])
        return (super().__str__() +
                f'\nTrainable parameters: {params}' +
                f'\nCRFT parameters: {crft_params}')

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        """
        处理IU X-Ray数据集（两张图像）
        使用CRFT对两个视图的特征进行对齐和融合
        """
        # 提取两个视图的特征
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # 确保特征是三维的 [batch_size, num_patches, feature_dim]
        if att_feats_0.dim() == 4:
            # 如果是四维的 [B, C, H, W]，转换为三维
            B, C, H, W = att_feats_0.shape
            att_feats_0 = att_feats_0.flatten(2).transpose(1, 2)  # [B, H*W, C]
            att_feats_1 = att_feats_1.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 使用CRFT进行特征对齐和融合
        if self.use_crft and hasattr(self, 'crft_aligner'):
            # CRFT对齐：将第二视图的特征对齐到第一视图
            fused_features = self.crft_aligner(att_feats_0, att_feats_1)

            # 使用融合的特征
            att_feats_0 = fused_features
            att_feats_1 = fused_features

        # 特征融合（原始方式）
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(
                fc_feats, att_feats, mode='sample', update_opts=update_opts
            )
            return output, output_probs
        else:
            raise ValueError(f"未知模式: {mode}")

    # def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
    #     """
    #     处理MIMIC-CXR数据集（单张图像）
    #     对于单张图像，不需要CRFT对齐
    #     """
    #     att_feats, fc_feats = self.visual_extractor(images)

    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #         return output
    #     elif mode == 'sample':
    #         output, output_probs = self.encoder_decoder(
    #             fc_feats, att_feats, mode='sample', update_opts=update_opts
    #         )
    #         return output, output_probs
    #     else:
    #         raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        """
        处理 MIMIC-CXR 数据集（Dataset 已输出两个视图，可能相同或不同）
        提取两个视图的特征，进行 CRFT 对齐（如果启用），然后融合
        """
        # images 形状: [batch_size, 2, C, H, W]
        # 分别提取两个视图的视觉特征
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # 确保 att_feats 是三维 [batch_size, num_patches, feature_dim]
        if att_feats_0.dim() == 4:
            B, C, H, W = att_feats_0.shape
            att_feats_0 = att_feats_0.flatten(2).transpose(1, 2)  # [B, H*W, C]
            att_feats_1 = att_feats_1.flatten(2).transpose(1, 2)

        # 使用 CRFT 进行特征对齐和融合（如果启用）
        if self.use_crft and hasattr(self, 'crft_aligner'):
            # 将第二视图特征对齐到第一视图
            fused_features = self.crft_aligner(att_feats_0, att_feats_1)
            # 用对齐后的特征替换原始特征（也可以选择只替换其中一个，但这里保持后续拼接逻辑不变）
            att_feats_0 = fused_features
            att_feats_1 = fused_features

        # 特征融合：将两个视图的全局特征和局部特征分别拼接
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(
                fc_feats, att_feats, mode='sample', update_opts=update_opts
            )
            return output, output_probs
        else:
            raise ValueError(f"未知模式: {mode}")
