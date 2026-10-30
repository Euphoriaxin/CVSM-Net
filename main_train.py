import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tokenizers import Tokenizer
from modules.loss import compute_loss
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from models.models import BaseCMNModel


def parse_args():
    parser = argparse.ArgumentParser()

    # 定义统一的布尔解析函数，完美避开 argparse 字符串陷阱
    str2bool = lambda x: (str(x).lower() == 'true')

    # 数据参数
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='图像目录')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='标注文件路径')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'])

    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--d_vf', type=int, default=2048, help='视觉特征维度')
    parser.add_argument('--num_layers', type=int, default=3, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--logit_layers', type=int, default=1, help='logit层数')

    # CMN参数
    parser.add_argument('--cmm_size', type=int, default=512, help='记忆矩阵大小')
    parser.add_argument('--cmm_dim', type=int, default=256, help='记忆矩阵维度')
    parser.add_argument('--topk', type=int, default=32, help='top-k记忆查询')

    # HD记忆参数
    parser.add_argument('--hd_dim', type=int, default=256, help='HD维度')
    parser.add_argument('--use_hd_memory', type=str2bool, default=True, help='是否使用HD记忆')
    parser.add_argument('--hd_memory_weight', type=float, default=0.8, help='HD记忆权重')
    parser.add_argument('--update_hd_memory', type=str2bool, default=True, help='是否更新HD记忆')
    parser.add_argument('--hd_topk', type=int, default=16, help='HD记忆top-k')

    # CRFT参数
    parser.add_argument('--use_crft', type=str2bool, default=True, help='是否使用CRFT配准')

    # 数据增强参数
    parser.add_argument('--augment', type=str2bool, default=False, help='是否启用数据增强')
    parser.add_argument('--aug_intensity', type=float, default=0.1, help='增强强度')
    parser.add_argument('--aug_rotation', type=int, default=5, help='旋转角度范围（度）')
    parser.add_argument('--aug_crop_scale', type=float, default=0.9, help='随机裁剪缩放范围下限')
    parser.add_argument('--aug_flip', type=str2bool, default=False, help='是否启用水平翻转')

    # 特殊标记索引
    parser.add_argument('--bos_idx', type=int, default=0, help='<start>标记索引')
    parser.add_argument('--eos_idx', type=int, default=1, help='<end>标记索引')
    parser.add_argument('--pad_idx', type=int, default=3, help='<pad>标记索引')
    parser.add_argument('--unk_idx', type=int, default=2, help='<unk>标记索引')
    parser.add_argument('--vocab_size', type=int, default=655, help='词汇表大小')

    # 模型架构参数
    parser.add_argument('--use_bn', type=int, default=0, help='是否使用BatchNorm (0: 不使用, 1: 使用)')

    # 训练参数
    parser.add_argument('--max_seq_length', type=int, default=60, help='最大序列长度')
    parser.add_argument('--threshold', type=int, default=3, help='词频阈值')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='视觉编码器学习率')
    parser.add_argument('--lr_ed', type=float, default=5e-4, help='编解码器学习率')
    parser.add_argument('--step_size', type=int, default=10, help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.8, help='学习率衰减率')
    parser.add_argument('--seed', type=int, default=7580, help='随机种子')

    # 数据加载器参数
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载的工作进程数')

    # Dropout相关参数
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='语言模型的dropout概率')

    # 优化器参数
    parser.add_argument('--optim', type=str, default='Adam', help='优化器类型')
    parser.add_argument('--adam_betas', type=str, default='(0.9, 0.999)', help='Adam优化器的betas参数')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam优化器的epsilon参数')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--amsgrad', type=str2bool, default=True, help='是否使用AMSGrad')

    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='学习率调度器类型')

    # Noam优化器参数
    parser.add_argument('--noamopt_warmup', type=int, default=2000, help='Noam优化器的warmup步数')
    parser.add_argument('--noamopt_factor', type=float, default=1.0, help='Noam优化器的因子')

    # ReduceLROnPlateau参数
    parser.add_argument('--reduce_on_plateau_factor', type=float, default=0.1, help='Plateau调度器的衰减因子')
    parser.add_argument('--reduce_on_plateau_patience', type=int, default=10, help='Plateau调度器的耐心值')

    # 视觉特征提取器参数
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='视觉特征提取器')
    parser.add_argument('--visual_extractor_pretrained', type=str2bool, default=True, help='是否使用预训练特征提取器')

    # 推理参数
    parser.add_argument('--beam_size', type=int, default=3, help='束搜索大小')
    parser.add_argument('--sample_method', type=str, default='beam_search', help='采样方法')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--sample_n', type=int, default=1, help='采样数量')

    # 其他参数
    parser.add_argument('--save_dir', type=str, default='results/iu_xray_crft_hd/', help='保存目录')
    parser.add_argument('--record_dir', type=str, default='records/', help='记录目录')
    parser.add_argument('--log_period', type=int, default=50, help='日志记录周期')
    parser.add_argument('--save_period', type=int, default=1, help='模型保存周期')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='监控指标')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='监控模式')
    parser.add_argument('--early_stop', type=int, default=20, help='早停轮数')
    parser.add_argument('--n_gpu', type=int, default=1, help='GPU数量')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练路径')

    args = parser.parse_args()

    # 将字符串形式的adam_betas转换为元组
    if hasattr(args, 'adam_betas') and isinstance(args.adam_betas, str):
        try:
            betas_str = args.adam_betas.strip('()').replace(' ', '')
            betas = tuple(map(float, betas_str.split(',')))
            args.adam_betas = betas
        except Exception as e:
            print(f"警告: 无法解析adam_betas参数 '{args.adam_betas}'，使用默认值(0.9, 0.999)")
            args.adam_betas = (0.9, 0.999)

    return args


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 保存配置
    config_path = os.path.join(args.save_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    print(f"✅ 训练配置已保存到: {config_path}")

    # 创建tokenizer
    tokenizer = Tokenizer(args)
    print(f"✅ Tokenizer创建完成，词汇表大小: {tokenizer.get_vocab_size()}")

    # 【提前固化词表】确保测试时词表索引与训练完全一致
    vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
    tokenizer.save_vocab(vocab_path)
    print(f"✅ 词表已提前固化到: {vocab_path}")

    # 更新词汇表大小到args
    args.vocab_size = tokenizer.get_vocab_size()

    # 创建数据加载器
    print("🔄 创建数据加载器...")

    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    print(f"✅ 数据加载器创建完成")
    print(f"   - 训练集: {len(train_dataloader.dataset)} 个样本")
    print(f"   - 验证集: {len(val_dataloader.dataset)} 个样本")
    print(f"   - 测试集: {len(test_dataloader.dataset)} 个样本")

    # 创建模型
    model = BaseCMNModel(args, tokenizer)
    print(f"✅ 模型创建完成")
    print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"   - 总参数: {sum(p.numel() for p in model.parameters())}")

    # CRFT配置
    if args.use_crft:
        print(f"⚡ CRFT配准模块已启用")
    else:
        print(f"❌ CRFT配准模块已禁用 (消融状态)")

    # HD记忆配置
    if args.use_hd_memory:
        print(f"⚡ HD记忆模块已启用")
        print(f"   - HD维度: {args.hd_dim}")
        print(f"   - HD权重: {args.hd_memory_weight}")
    else:
        print(f"❌ HD记忆模块已禁用 (消融状态)")

    # 创建损失函数
    criterion = compute_loss

    # 创建评估函数
    metric_ftns = compute_scores

    # 创建优化器
    optimizer = build_optimizer(args, model)
    print(f"✅ 优化器创建完成")
    print(f"   - 优化器类型: {args.optim}")
    print(f"   - 初始学习率: VE={args.lr_ve}, ED={args.lr_ed}")
    print(f"   - 权重衰减: {args.weight_decay}")
    print(f"   - Adam betas: {args.adam_betas}")
    print(f"   - Adam eps: {args.adam_eps}")

    # 创建学习率调度器
    lr_scheduler = build_lr_scheduler(args, optimizer)
    print(f"✅ 学习率调度器创建完成")
    print(f"   - 调度器类型: {args.lr_scheduler}")
    print(f"   - 步长: {args.step_size}, 衰减率: {args.gamma}")

    # 创建训练器
    trainer = Trainer(
        model, criterion, metric_ftns, optimizer, args, lr_scheduler,
        train_dataloader, val_dataloader, test_dataloader
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()