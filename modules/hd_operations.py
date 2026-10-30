# modules/hd_operations.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import ortho_group


class HDOperations:
    """超维计算的基本操作"""

    # 在超维计算中用于将两个超维向量结合成一个新的超维向量
    @staticmethod
    def binding(a, b):
        """
        绑定操作: a ⊗ b
        在双极向量中，绑定是逐元素乘法
        """
        return a * b

    # 用于将多个超维向量叠加成一个“集合”或“多集”向量。捆绑后的向量保留了所有输入向量的信息
    @staticmethod
    def bundling(*vectors):
        """
        捆绑操作: a ⊕ b ⊕ ...
        在实值向量中，捆绑是求和
        在推理时需要进行二值化
        """
        if len(vectors) == 0:
            return None
        result = vectors[0]
        for v in vectors[1:]:
            result = result + v
        return result

    # 用于计算两个“超维向量”之间的相似度
    @staticmethod
    def similarity(a, b, metric='cosine'):
        """
        计算超维向量之间的相似度
        a, b: [batch_size, hd_dim] 或 [hd_dim]
        """
        if metric == 'cosine':
            # 可微的相似度计算：cos_sim = (a·b) / (||a||·||b||)
            return F.cosine_similarity(a, b, dim=-1)
        elif metric == 'hamming':
            # 汉明距离（在推理时使用）
            # 注意：这需要二值化向量
            diff = (a != b).float()
            if diff.dim() > 1:
                return diff.mean(dim=-1)
            return 1 - diff.mean()
        # 不支持的度量方式
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    # 将实值向量转换为二值向量（通常为 ±1）
    @staticmethod
    def binarize(x, method='sign'):
        """
        将实值向量二值化
        method: 'sign' 或 'stochastic'
        """
        if method == 'sign':
            return torch.sign(x)
        elif method == 'stochastic':
            # 直通估计器，在训练时保持梯度
            return HDOperations.straight_through_sign(x)
        else:
            raise ValueError(f"Unsupported binarization method: {method}")

    # 前向传播返回 sign(x)；反向传播时，梯度直接从 x 流经，忽略 sign 的零梯度问题（因为 detach() 切断了梯度，而 x 部分保留了梯度）
    @staticmethod
    def straight_through_sign(x):
        """
        直通估计器：前向传播使用sign，反向传播使用梯度
        """
        with torch.no_grad():
            sign = torch.sign(x)
        return x + (sign - x).detach()


class LSHEncoder(nn.Module):
    """
    位置敏感哈希编码器
    将实值特征映射到超维空间
    """

    def __init__(self, input_dim, hd_dim=10000, num_projections=None):
        super(LSHEncoder, self).__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.hd_dim = hd_dim  # 输出超维向量的维度

        # 使用随机投影矩阵（可学习或固定）
        if num_projections is None:
            num_projections = hd_dim

        # 初始化随机投影矩阵
        self.projection = nn.Parameter(
            torch.randn(num_projections, input_dim) / np.sqrt(input_dim),
            requires_grad=True  # 设为True允许学习
        )

        # 位置编码（用于图像块）
        self.position_codes = nn.Parameter(
            torch.randn(1, 1024, hd_dim),  # 假设1024个图像块
            requires_grad=False
        )

    def forward(self, x, positions=None):
        """
        x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
        返回: [batch_size, seq_len, hd_dim] 或 [batch_size, hd_dim]
        """
        batch_size = x.shape[0]

        if x.dim() == 2:
            # 2D：(batch_size, input_dim) → 输出 (batch_size, hd_dim)
            x_proj = torch.matmul(x, self.projection.t())
        elif x.dim() == 3:
            # 3D：(batch_size, seq_len, input_dim) → 输出 (batch_size, seq_len, hd_dim)
            seq_len = x.shape[1]
            x_flat = x.reshape(-1, self.input_dim)
            x_proj_flat = torch.matmul(x_flat, self.projection.t())
            x_proj = x_proj_flat.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        # 应用位置编码（如果是图像块）
        if positions is not None and x.dim() == 3:
            pos_codes = self.position_codes[:, :positions, :]
            x_proj = HDOperations.binding(x_proj, pos_codes)

        # 二值化（使用直通估计器以便训练）
        hd_vector = HDOperations.straight_through_sign(x_proj)

        return hd_vector