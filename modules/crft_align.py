# modules/crft_align.py
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CRFT 轻量版：Consistent-Recurrent Feature Flow Transformer
用于医学双视图/多模态图像 空间对齐 + 特征增强
"""


class FeatureFlowAttention(nn.Module):
    """CRFT 核心：差异引导注意力 + 空间流场对齐"""

    def __init__(self, dim=2048, num_heads=4):
        super().__init__()
        self.heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"维度{dim}必须能被头数{num_heads}整除"
        self.scale = self.head_dim ** -0.5  # 缩放因子，用于稳定注意力权重的梯度

        self.qkv = nn.Linear(dim, dim * 3)  # 线性层，将输入特征同时投影为 Query、Key、Value 三个向量（一次性生成，提高效率）
        self.flow_proj = nn.Linear(dim, dim)  # 对齐后的特征做一次线性变换，增加模型容量
        self.norm = nn.LayerNorm(dim)  # 层归一化，用于稳定训练、加速收敛

    def forward(self, feat1, feat2):
        B, N, C = feat1.shape
        # 线性变换
        # 将维度重新排列为 (3, B, heads, N, head_dim)
        qkv1 = self.qkv(feat1).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 在维度 0 上拆分，得到 q1, k1, v1，每个形状为 (B, heads, N, head_dim)
        q1, k1, v1 = qkv1.unbind(0)

        qkv2 = self.qkv(feat2).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)

        # 跨注意力（CRFT 核心：模态无关匹配）
        # q2（查询来自 moving 图像）与 k1（键来自 fixed 图像）的点积 (B, heads, N, head_dim) @ (B, heads, head_dim, N) → (B, heads, N, N)
        attn = (q2 @ k1.transpose(-2, -1)) * self.scale
        # 对每个查询（最后维度）做 softmax，得到注意力权重
        attn = attn.softmax(dim=-1)

        # 加权聚合Value并重组
        # 注意力权重与 feat1 的 Value 相乘，形状 (B, heads, N, head_dim)。——>相当于用来自 feat1 的信息去"重构"feat2 的每个位置，实现空间对齐
        aligned_feat = (attn @ v1).transpose(1, 2).reshape(B, N, C)  # aligned_feat 是 feat2 对齐到 feat1 空间后的特征

        # 投影与残差连接
        aligned_feat = self.flow_proj(aligned_feat)  # flow_proj：线性变换，增强对齐特征的表达能力
        return self.norm(feat2 + aligned_feat)  # 残差连接，保留原始 moving 特征，避免信息丢失


class CRFTImageAligner(nn.Module):
    """
    CRFT 医学图像对齐模块
    输入：两张医学图像（如胸片 AP + LAT）
    输出：空间对齐后的融合特征
    """

    def __init__(self, embed_dim=2048):
        super().__init__()
        self.align_net = FeatureFlowAttention(dim=embed_dim)
        # 一个两层的前馈网络
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            # 将 fixed 特征与对齐后的 moving 特征沿特征维度拼接，得到 2*embed_dim 维，再降维回 embed_dim，经过 LayerNorm 和 GELU 激活
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, feat_fixed, feat_moving):
        # 调用 FeatureFlowAttention，输入 feat_fixed 作为固定参考（充当 K/V 的来源），feat_moving 作为待对齐特征（充当 Q 的来源）
        aligned_feat = self.align_net(feat_fixed, feat_moving)
        # 特征融合：将 feat_fixed 与对齐后的 aligned_feat 拼接，通过融合网络得到最终的特征
        fused = self.fusion(torch.cat([feat_fixed, aligned_feat], dim=-1))
        return fused


