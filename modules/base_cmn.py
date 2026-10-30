# 256维度
# modules/base_cmn.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel
from .hd_memory import HDMemoryModule  # 新增导入


# 深度复制模块 N 次，用于构建多层堆叠
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 生成一个上三角掩码，用于解码器中的自回归（防止看到未来 token）
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# 标准缩放点积注意力，返回加权后的值和注意力权重
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


"""
MultiThreadMemory 的核心操作：
仅关注与查询最相似的少数几个内存槽（top‑k），来降低计算量并提高可解释性
"""


# 计算 query 与 key 的相似度分数
# 只保留 top‑k 个最高分数的索引
# 根据这些索引从 value 中取出对应的向量
# 对这些选出的 value 做 softmax 加权求和
# 相比标准注意力（对所有 key 做 softmax），top‑k 稀疏化可以降低计算量，并强制模型聚焦于最相关的记忆项
def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    # 计算原始注意力分数
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 应用掩码
    if mask is not None:
        # 当 mask == 0 的位置，用 -inf 填充对应的 scores 元素。这样在后续 softmax 中，这些位置的注意力权重会变为 0
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # 选择 Top‑k 最高分数
    selected_scores, idx = scores.topk(topk)
    # 根据索引收集对应的值（value）:目的是从 value 中取出与 idx 对应的 topk 个值向量
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    # 对选中的分数计算 softmax 注意力权重
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn
        # 添加一个投影层，用于将文本嵌入从512维投影到256维
        self.text_proj = None

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)

        """动态创建投影层（文本嵌入 → 记忆维度）"""
        # 使目标嵌入的维度与记忆矩阵的特征维度匹配，才能将 embeddings 作为查询与记忆矩阵计算相似度
        if embeddings.size(-1) != memory_matrix.size(-1):
            if self.text_proj is None or self.text_proj.in_features != embeddings.size(
                    -1) or self.text_proj.out_features != memory_matrix.size(-1):
                self.text_proj = nn.Linear(embeddings.size(-1), memory_matrix.size(-1)).to(embeddings.device)
            embeddings_proj = self.text_proj(embeddings)
        else:
            embeddings_proj = embeddings
        # 记忆查询与响应
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1))
        responses = self.cmn(embeddings_proj, dummy_memory_matrix, dummy_memory_matrix)
        # 将响应投影回原始维度
        if responses.size(-1) != embeddings.size(-1):
            if not hasattr(self, 'text_proj_back'):
                self.text_proj_back = nn.Linear(responses.size(-1), embeddings.size(-1)).to(responses.device)
            responses = self.text_proj_back(responses)
        # 残差连接与解码器输出
        embeddings = embeddings + responses
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32, input_dim=None):
        """
        修改：添加 input_dim 参数
        """
        super(MultiThreadMemory, self).__init__()
        # 如果提供了 input_dim，使用它，否则使用 d_model
        self.input_dim = input_dim if input_dim is not None else d_model
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # 修改：创建4个不同的线性层
        # 前3个用于 query, key, value，从 input_dim 投影到 d_model
        # 最后一个用于输出，从 d_model 投影到 d_model
        self.query_linear = nn.Linear(self.input_dim, d_model)
        self.key_linear = nn.Linear(self.input_dim, d_model)
        self.value_linear = nn.Linear(self.input_dim, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.query_linear(query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query = self.query_linear(query)
            key = self.key_linear(key)
            value = self.value_linear(value)
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.output_linear(x), present
        else:
            return self.output_linear(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseCMN(AttModel):
    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk

        tgt_vocab = self.vocab_size + 1

        # 修改：将 input_dim 设置为256
        # 因为 att_feats 会被投影到 memory_matrix 的维度（256）
        visual_feat_dim = 256

        self.cmn = MultiThreadMemory(args.num_heads, args.d_model,
                                     topk=args.topk, input_dim=visual_feat_dim)

        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        # memory_matrix 维度是 (cmm_size, cmm_dim=256)
        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)

        # HD记忆模块
        self.hd_memory = HDMemoryModule(args, tokenizer)

        self.current_epoch = 0

    def _save_attns(self, start=False):
        """
        保存注意力权重
        这是一个简单实现，如果不需要保存注意力可以留空
        """
        # 如果父类有这个方法，调用父类的
        try:
            super()._save_attns(start)
        except AttributeError:
            # 父类没有这个方法，我们自己实现一个简单版本
            if start:
                self.attention_weights = []
            # 这里可以添加保存注意力权重的逻辑
            # 但为了简化，我们只记录一个标记
            if not hasattr(self, 'attn_saved'):
                self.attn_saved = True
            return

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    # 记忆查询与特征增强
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        if att_feats.size(-1) != self.memory_matrix.size(-1):
            # 创建投影层，将512维投影到256维
            proj = nn.Linear(att_feats.size(-1), self.memory_matrix.size(-1)).to(att_feats.device)
            att_feats_proj = proj(att_feats)
        else:
            att_feats_proj = att_feats

        # 原始CMN记忆查询
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(
            att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))

        responses = self.cmn(att_feats_proj, dummy_memory_matrix, dummy_memory_matrix)

        # HD记忆查询（注意：HD记忆可能期望512维输入）
        if getattr(self.args, 'use_hd_memory', True):
            hd_responses, _ = self.hd_memory(
                att_feats, mode='visual', update=False
            )
            # 注意：hd_responses 维度可能与 responses 不同
            # 如果需要，将 hd_responses 投影到与 responses 相同的维度
            if hd_responses.size(-1) != responses.size(-1):
                hd_proj = nn.Linear(hd_responses.size(-1), responses.size(-1)).to(hd_responses.device)
                hd_responses = hd_proj(hd_responses)

            # 融合原始响应和HD响应
            hd_weight = getattr(self.args, 'hd_memory_weight', 0.3)
            responses = (1 - hd_weight) * responses + hd_weight * hd_responses

        # 将响应加回原始特征
        # 注意：att_feats 是512维，responses 是256维，需要投影
        if att_feats.size(-1) != responses.size(-1):
            proj_back = nn.Linear(responses.size(-1), att_feats.size(-1)).to(responses.device)
            responses = proj_back(responses)

        att_feats = att_feats + responses

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        # 获取原始输出
        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix)

        # ========== 新增：在训练时更新HD记忆 ==========
        if self.training and getattr(self.args, 'update_hd_memory', True):
            # 提取视觉特征（全局平均）
            visual_global = att_feats.mean(dim=1)  # [batch_size, d_model]

            # 为每个时间步更新记忆
            batch_size, seq_len = seq.shape
            for t in range(seq_len - 1):
                # 当前文本特征
                current_text = out[:, t, :]  # [batch_size, d_model]
                next_token = seq[:, t + 1]  # 下一个目标token

                # 更新HD记忆
                self.hd_memory(
                    current_text,
                    mode='text',
                    update=True,
                    targets=next_token,
                    position_idx=t
                )
        # ===========================================
        outputs = F.log_softmax(self.logit(out), dim=-1)
        # ========== 新增：用HD记忆增强logits ==========
        if getattr(self.args, 'use_hd_memory', True) and self.training:
            # 获取视觉特征（用于计算HD logits）
            visual_global = att_feats.mean(dim=1)
            # 为每个时间步获取HD logits
            hd_enhanced_outputs = []
            for t in range(seq_len):
                current_text = out[:, t, :]
                hd_logits = self.hd_memory.get_hd_logits(
                    visual_global.unsqueeze(1),  # 添加序列维度
                    current_text,
                    position_idx=t
                )
                if hd_logits is not None:
                    # 融合原始logits和HD logits
                    hd_weight = getattr(self.args, 'hd_memory_weight', 0.3)
                    combined = (1 - hd_weight) * outputs[:, t, :] + hd_weight * hd_logits
                    hd_enhanced_outputs.append(combined)
                else:
                    hd_enhanced_outputs.append(outputs[:, t, :])

            outputs = torch.stack(hd_enhanced_outputs, dim=1)
        # =============================================
        return outputs
    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        # 原始解码
        out, past = self.model.decode(memory, mask, ys,
                                      subsequent_mask(ys.size(1)).to(memory.device),
                                      past=past, memory_matrix=self.memory_matrix)
        # 保存注意力权重
        if not self.training:
            try:
                self._save_attns(start=len(state) == 0)
            except:
                # 如果_save_attns出错，跳过
                pass
        # ========== 新增：在推理时用HD记忆增强 ==========
        if getattr(self.args, 'use_hd_memory', True) and not self.training:
            # 获取当前时间步的输出
            current_out = out[:, -1:, :]  # 只取最后一个时间步
            # 获取视觉特征
            visual_global = fc_feats_ph.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
            # 获取HD记忆响应
            hd_response, _ = self.hd_memory(
                current_out, mode='text', update=False
            )
            # 融合原始输出和HD响应
            hd_weight = getattr(self.args, 'hd_memory_weight', 0.3)
            out_enhanced = (1 - hd_weight) * current_out + hd_weight * hd_response
            # 将增强后的输出放回去
            out = torch.cat([out[:, :-1, :], out_enhanced], dim=1)
        # =============================================
        return out[:, -1], [ys.unsqueeze(0)] + past