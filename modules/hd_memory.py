# modules/hd_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HDMemoryModule(nn.Module):
    """
    超维计算记忆模块
    与原始CMN的MultiThreadMemory并行工作
    """

    def __init__(self, args, tokenizer):
        super(HDMemoryModule, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # 获取词汇表大小
        try:
            self.vocab_size = tokenizer.get_vocab_size()
            print(f"HDMemory: 从 tokenizer.get_vocab_size() 获取词汇表大小 = {self.vocab_size}")
        except AttributeError as e:
            self.vocab_size = getattr(args, 'vocab_size', 3000)
            print(f"警告: 无法从 tokenizer 获取词汇表大小，使用默认值: {self.vocab_size}")

        # HD参数
        self.hd_dim = getattr(args, 'hd_dim', 512)  # 默认值改为512
        self.hd_weight = getattr(args, 'hd_memory_weight', 0.3)
        self.use_hd = getattr(args, 'use_hd_memory', True)
        self.topk = getattr(args, 'hd_topk', 32)

        # 减小记忆库大小以节省内存
        self.memory_size = min(args.cmm_size, 512)  # 最大512个记忆项

        # 投影层：将原始视觉和文本特征映射到HD空间
        self.visual_to_hd = nn.Linear(args.d_model, self.hd_dim)
        self.text_to_hd = nn.Linear(args.d_model, self.hd_dim)

        # 反向投影：将超维响应反向投影回模型空间（d_model），以便与原始特征相加或拼接
        self.hd_to_model = nn.Linear(self.hd_dim, args.d_model)

        # HD记忆库（可学习的原型向量）形状：[memory_size, hd_dim]
        # 每个记忆项代表一个超维原型
        self.hd_memory = nn.Parameter(
            torch.randn(self.memory_size, self.hd_dim) * 0.01
        )

        # 每个记忆项被访问/更新的次数
        self.usage_count = nn.Parameter(
            torch.zeros(self.memory_size),
            requires_grad=False
        )

        # 初始化
        self._init_weights()

    # 权重初始化：使用 Xavier 均匀初始化，保证投影层输出方差稳定
    def _init_weights(self):
        nn.init.xavier_uniform_(self.visual_to_hd.weight)
        nn.init.xavier_uniform_(self.text_to_hd.weight)
        nn.init.xavier_uniform_(self.hd_to_model.weight)

    # 根据查询（query）检索最相似的记忆项，加权聚合后返回映射回模型空间的响应
    def forward(self, query, mode='visual', update=False, targets=None, position_idx=None):
        """
        前向传播：查询HD记忆
        """
        # 处理 HD 记忆禁用情况的逻辑
        if not self.use_hd:
            if query.dim() == 3:
                return torch.zeros_like(query), None
            else:
                return torch.zeros_like(query.unsqueeze(1)).squeeze(1), None

        batch_size = query.size(0)

        # 1. 将查询映射到HD空间
        if mode == 'visual':
            hd_query = torch.tanh(self.visual_to_hd(query))
        else:  # text
            hd_query = torch.tanh(self.text_to_hd(query))

        # 2. 优化内存的相似度计算
        if hd_query.dim() == 3:
            # 序列输入
            seq_len = hd_query.size(1)
            hd_query_flat = hd_query.reshape(-1, self.hd_dim)  # [batch*seq_len, hd_dim]

            # 分批计算相似度以避免内存爆炸
            similarity = self._batch_cosine_similarity(
                hd_query_flat,  # [batch*seq_len, hd_dim]
                self.hd_memory  # [memory_size, hd_dim]
            )

            # 获取top-k最相似的记忆
            topk_sim, topk_idx = torch.topk(similarity, min(self.topk, similarity.size(1)), dim=1)

            # 3. 加权聚合记忆响应
            weights = F.softmax(topk_sim, dim=1)

            # 获取topk记忆向量
            topk_memory = self.hd_memory[topk_idx]  # [batch*seq_len, topk, hd_dim]

            # 加权求和
            weights_expanded = weights.unsqueeze(-1)  # [batch*seq_len, topk, 1]
            hd_response_flat = (topk_memory * weights_expanded).sum(dim=1)  # [batch*seq_len, hd_dim]

            # 重塑回原始形状
            hd_response = hd_response_flat.reshape(batch_size, seq_len, self.hd_dim)
            hd_similarity = topk_sim.reshape(batch_size, seq_len, -1)

        else:
            # 单个向量输入
            similarity = self._batch_cosine_similarity(
                hd_query,  # [batch_size, hd_dim]
                self.hd_memory  # [memory_size, hd_dim]
            )

            topk_sim, topk_idx = torch.topk(similarity, min(self.topk, similarity.size(1)), dim=1)
            weights = F.softmax(topk_sim, dim=1)

            topk_memory = self.hd_memory[topk_idx]  # [batch_size, topk, hd_dim]
            weights_expanded = weights.unsqueeze(-1)
            hd_response = (topk_memory * weights_expanded).sum(dim=1)  # [batch_size, hd_dim]
            hd_similarity = topk_sim

        # 4. 将HD响应映射回模型空间
        hd_response_model = self.hd_to_model(hd_response)

        # 5. 可选：更新记忆（训练时）
        if update and self.training and targets is not None:
            self._update_memory(hd_query, targets, position_idx)

        return hd_response_model, hd_similarity

    def _batch_cosine_similarity(self, query, memory, batch_size=32):
        """
        分批计算余弦相似度，避免内存爆炸
        query: [n_query, hd_dim]
        memory: [n_memory, hd_dim]
        返回: [n_query, n_memory]
        """
        n_query = query.size(0)
        n_memory = memory.size(0)

        # 归一化查询向量
        query_norm = F.normalize(query, p=2, dim=1)
        memory_norm = F.normalize(memory, p=2, dim=1)

        # 分批计算
        all_similarities = []
        for i in range(0, n_query, batch_size):
            end = min(i + batch_size, n_query)
            query_batch = query_norm[i:end]  # [batch, hd_dim]

            # 计算相似度: query_batch @ memory_norm.T
            sim_batch = torch.mm(query_batch, memory_norm.t())  # [batch, n_memory]
            all_similarities.append(sim_batch)

        return torch.cat(all_similarities, dim=0)

    # 当前实现仅为每个查询找到最相似的记忆项（torch.max），并将对应 usage_count 加 1
    def _update_memory(self, hd_query, targets, position_idx):
        """
        更新HD记忆
        """
        with torch.no_grad():
            batch_size = hd_query.size(0)
            if hd_query.dim() == 3:
                hd_query_flat = hd_query.reshape(-1, self.hd_dim)
                # 使用分批相似度计算
                similarity = self._batch_cosine_similarity(hd_query_flat, self.hd_memory)
                _, most_similar = torch.max(similarity, dim=1)

                # 更新计数器
                for idx in most_similar:
                    self.usage_count.data[idx] += 1

    def get_hd_logits(self, visual_features, text_features, position_idx=None):
        """
        基于HD记忆计算词汇logits
        用于在推理时增强原始logits
        """
        if not self.use_hd:
            return None

        batch_size = visual_features.size(0)

        # 将视觉和文本特征映射到HD空间
        visual_hd = torch.tanh(self.visual_to_hd(visual_features.mean(dim=1)))
        text_hd = torch.tanh(self.text_to_hd(text_features))

        # 绑定视觉和文本特征
        bound_vector = visual_hd * text_hd

        # 计算与所有记忆的相似度
        similarity = self._batch_cosine_similarity(bound_vector, self.hd_memory)

        # 简化：将相似度平均值作为基础logits
        mean_similarity = similarity.mean(dim=1, keepdim=True)  # [batch_size, 1]
        hd_logits = mean_similarity.expand(-1, self.vocab_size + 1) * 0.1

        return hd_logits

    def reset_memory(self):
        """重置记忆"""
        with torch.no_grad():
            self.hd_memory.data.normal_(0, 0.01)
            self.usage_count.data.zero_()