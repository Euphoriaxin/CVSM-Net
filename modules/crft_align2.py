import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 【原始核心代码 - 完全保留】 =====================
class FeatureFlowAttention(nn.Module):
    """CRFT 核心：差异引导注意力 + 空间流场对齐"""
    def __init__(self, dim=512, num_heads=2):
        super().__init__()
        self.heads = num_heads  # 多头注意力头数
        self.head_dim = dim // num_heads  # 每个头的特征维度
        # 断言：总维度必须能被头数整除（保证维度合法）
        assert self.head_dim * num_heads == dim, f"维度{dim}必须能被头数{num_heads}整除"
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子（防止点积数值过大）

        # 线性层：输入特征 → 生成 Q(查询)、K(键)、V(值) 3份特征（dim*3）
        self.qkv = nn.Linear(dim, dim * 3)
        # 流场投影层：对齐后的特征做线性变换，优化流场表达
        self.flow_proj = nn.Linear(dim, dim)
        # 层归一化：稳定训练，防止梯度消失
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat1, feat2):
        # 张量形状：B=批次, N=序列长度(图像展平后的像素数), C=特征维度
        B, N, C = feat1.shape

        # ========== 步骤1：为 feat1（固定/参考特征）生成 QKV ==========
        # 线性变换 → 重塑为 [B, N, 3(QKV), 头数, 单头维度]
        qkv1 = self.qkv(feat1).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)  # 拆分出 Q1、K1、V1（固定特征的查询、键、值）

        # ========== 步骤2：为 feat2（移动/待对齐特征）生成 QKV ==========
        qkv2 = self.qkv(feat2).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)  # 拆分出 Q2、K2、V2（移动特征的查询、键、值）

        # ========== 步骤3：核心空间对齐注意力计算 ==========
        # 用【移动特征的Q2】去匹配【固定特征的K1】，生成空间对齐权重
        attn = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 归一化注意力权重（0~1）

        # ========== 步骤4：加权融合固定特征，得到对齐结果 ==========
        # 注意力权重 × 固定特征的V1 → 对齐后的特征
        aligned_feat = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        aligned_feat = self.flow_proj(aligned_feat)  # 流场投影优化

        # ========== 步骤5：残差连接 + 归一化 ==========
        # 残差：移动特征 + 对齐后的固定特征（保留原始信息，仅修正空间偏移）
        return self.norm(feat2 + aligned_feat)


class CRFTImageAligner(nn.Module):
    """图像特征对齐模块：对齐 + 特征融合"""
    def __init__(self, embed_dim=512, num_heads=2):
        super().__init__()
        # 实例化核心对齐注意力层
        self.align_net = FeatureFlowAttention(dim=embed_dim, num_heads=num_heads)
        # 特征融合层：拼接后的特征 → 还原为原始维度
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),    # 拼接后维度×2，线性降维
            nn.LayerNorm(embed_dim),                # 归一化
            nn.GELU()                               # 激活函数（比ReLU更平滑）
        )

    def forward(self, feat_fixed, feat_moving):
        # 步骤1：核心对齐 → 固定特征对齐到移动特征
        aligned_feat = self.align_net(feat_fixed, feat_moving)
        # 步骤2：拼接【原始固定特征】+【对齐后的特征】
        fused = self.fusion(torch.cat([feat_fixed, aligned_feat], dim=-1))
        # 步骤3：输出融合对齐后的特征
        return fused


# ===================== 【增强模块】 =====================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 2个下采样卷积：3x3卷积，步幅2，填充1 → 尺寸减半
        self.down_4x4 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.down_2x2 = nn.Conv2d(dim, dim, 3, 2, 1)
        # 1x1卷积：统一多尺度特征维度
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        scale1 = x                  # 原始尺度（1x1）
        scale2 = self.down_4x4(x)   # 下采样1次（2x2）
        scale3 = self.down_2x2(scale2) # 下采样2次（4x4）
        # 投影后返回3个尺度的特征
        return self.proj(scale1), self.proj(scale2), self.proj(scale3)


class FineFlowRefiner(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 1x1卷积：融合3个尺度的特征（维度×3 → 原始维度）
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, scale1, scale2, scale3):
        # 双线性上采样：小尺度特征 → 还原为原始尺寸
        scale2_up = F.interpolate(scale2, size=scale1.shape[2:], mode='bilinear', align_corners=False)
        scale3_up = F.interpolate(scale3, size=scale1.shape[2:], mode='bilinear', align_corners=False)
        # 拼接3个尺度特征 + 融合
        fused = torch.cat([scale1, scale2_up, scale3_up], dim=1)
        return self.fusion(fused)


# ===================== 【最终通用版封装】 =====================
class FullCRFT(nn.Module):
    """
    通用配置：大小数据集共用一份代码
    小数据集：use_multiscale=False, iters=1, num_heads=2
    大数据集：use_multiscale=True, iters=2, num_heads=4
    """

    def __init__(self, dim=512, iters=1, use_multiscale=True, num_heads=2):
        super().__init__()
        self.dim = dim
        self.iters = min(iters, 2)  # 迭代对齐次数（最多2次）
        self.use_multiscale = use_multiscale  # 多尺度增强开关

        # 条件加载：开启多尺度则加载增强模块
        if self.use_multiscale:
            self.multiscale = MultiScaleFeatureExtractor(dim)
            self.refiner = FineFlowRefiner(dim)

        # 加载对齐器
        self.aligner = CRFTImageAligner(embed_dim=dim, num_heads=num_heads)

    def forward(self, feat_fixed, feat_moving):
        B, N, C = feat_fixed.shape
        H = W = int(N ** 0.5)  # 展平的序列特征 → 还原为正方形图像尺寸

        # ===================== 多尺度增强（可选） =====================
        if self.use_multiscale:
            # 1. 序列特征 [B,N,C] → 2D图像特征 [B,C,H,W]（适配卷积）
            f_fix = feat_fixed.transpose(1, 2).view(B, C, H, W)
            f_mov = feat_moving.transpose(1, 2).view(B, C, H, W)

            # 2. 多尺度特征提取
            s1_fix, s2_fix, s3_fix = self.multiscale(f_fix)
            s1_mov, s2_mov, s3_mov = self.multiscale(f_mov)

            # 3. 流场细化融合
            fix_enhanced = self.refiner(s1_fix, s2_fix, s3_fix)
            mov_enhanced = self.refiner(s1_mov, s2_mov, s3_mov)

            # 4. 2D图像特征 → 还原为序列特征 [B,N,C]
            fix_enhanced = fix_enhanced.flatten(2).transpose(1, 2)
            mov_enhanced = mov_enhanced.flatten(2).transpose(1, 2)
        else:
            # 关闭多尺度：直接使用原始特征
            fix_enhanced = feat_fixed
            mov_enhanced = feat_moving

        # ===================== 迭代对齐 =====================
        out = fix_enhanced
        for _ in range(self.iters):
            out = self.aligner(out, mov_enhanced)

        # 输出最终对齐融合后的特征
        return out