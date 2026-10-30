# modules/trainer.py
import logging
import os
import time
from datetime import datetime
from abc import abstractmethod

import torch
from numpy import inf
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        # 设置日志格式
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        # 设置文件日志
        self._setup_file_logging()

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        # 初始化最佳模型记录器
        self.best_model_info = {
            'epoch': 0,
            'metric_name': self.mnt_metric,
            'metric_value': self.mnt_best,
            'path': '',
            'timestamp': ''
        }

        # 初始化最佳记录器，包含所有验证和测试指标
        self._init_best_recorder()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # 记录开始时间
        self.start_time = time.time()

    def _setup_file_logging(self):
        """设置文件日志"""
        if not hasattr(self.args, 'record_dir'):
            self.args.record_dir = 'records/'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        # 创建时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.args.record_dir, f'training_{timestamp}.log')

        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)

        # 保存日志文件路径
        self.log_file = log_file
        self.logger.info(f"日志文件保存到: {log_file}")

    def _init_best_recorder(self):
        """初始化最佳记录器，包含所有指标"""
        # 初始化验证集最佳记录
        val_metrics = {
            'val_BLEU_1': -inf if self.mnt_mode == 'max' else inf,
            'val_BLEU_2': -inf if self.mnt_mode == 'max' else inf,
            'val_BLEU_3': -inf if self.mnt_mode == 'max' else inf,
            'val_BLEU_4': -inf if self.mnt_mode == 'max' else inf,
            'val_ROUGE_L': -inf if self.mnt_mode == 'max' else inf,
            'val_METEOR': -inf if self.mnt_mode == 'max' else inf,
            'val_CIDEr': -inf if self.mnt_mode == 'max' else inf
        }

        # 初始化测试集最佳记录
        test_metrics = {
            'test_BLEU_1': -inf if self.mnt_mode == 'max' else inf,
            'test_BLEU_2': -inf if self.mnt_mode == 'max' else inf,
            'test_BLEU_3': -inf if self.mnt_mode == 'max' else inf,
            'test_BLEU_4': -inf if self.mnt_mode == 'max' else inf,
            'test_ROUGE_L': -inf if self.mnt_mode == 'max' else inf,
            'test_METEOR': -inf if self.mnt_mode == 'max' else inf,
            'test_CIDEr': -inf if self.mnt_mode == 'max' else inf
        }

        self.best_recorder = {'val': val_metrics, 'test': test_metrics}
        # 设置监控指标的特殊值
        self.best_recorder['val'][self.mnt_metric] = self.mnt_best
        self.best_recorder['test'][self.mnt_metric_test] = self.mnt_best

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """主训练循环"""
        not_improved_count = 0

        # 打印训练配置
        self._print_training_config()

        # 训练循环
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 记录epoch开始时间
            epoch_start_time = time.time()

            # 训练一个epoch
            result = self._train_epoch(epoch)

            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time

            # 保存日志信息
            log = {'epoch': epoch, 'epoch_time': epoch_time}
            log.update(result)
            self._record_best(log)

            # 打印详细训练信息
            self._print_epoch_info(epoch, log, epoch_time)

            # 检查是否是最佳模型
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"监控指标 '{self.mnt_metric}' 未找到，停止监控")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True

                    # 更新最佳模型信息
                    self.best_model_info = {
                        'epoch': epoch,
                        'metric_name': self.mnt_metric.replace('val_', ''),
                        'metric_value': self.mnt_best,
                        'path': os.path.join(self.checkpoint_dir, 'model_best.pth'),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # 保存最佳模型
                    self._save_checkpoint(epoch, save_best=True)

                    # 打印最佳模型信息
                    self._print_best_model_info()
                else:
                    not_improved_count += 1

                # 早停检查
                if not_improved_count > self.early_stop:
                    total_time = time.time() - self.start_time
                    self.logger.info(f"🚨 训练早停！在 {self.early_stop} 个epoch内验证指标未提升")
                    self.logger.info(f"⏱️  总训练时间: {total_time / 3600:.2f} 小时")
                    break

            # 定期保存检查点
            if epoch % self.save_period == 0 and not best:
                self._save_checkpoint(epoch, save_best=False)

            # 记录到文件
            self._log_to_file(epoch, log, best)

        # 训练结束
        total_time = time.time() - self.start_time
        self._print_training_summary(total_time)
        self._print_best_recorder()

        # 保存最终模型
        self._save_checkpoint(self.epochs, save_best=False, prefix='final_')

    def _print_training_config(self):
        """打印训练配置"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始 CMN + HDFLIM 训练流程")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 数据集: {self.args.dataset_name}")
        self.logger.info(f"📈 批次大小: {self.args.batch_size}")
        self.logger.info(f"📊 训练轮数: {self.args.epochs}")
        self.logger.info(f"🎯 监控指标: {self.args.monitor_metric} ({self.mnt_mode})")
        self.logger.info(f"⏰ 早停轮数: {self.early_stop}")
        self.logger.info(f"💾 保存目录: {self.checkpoint_dir}")
        self.logger.info(f"📝 日志文件: {self.log_file}")
        self.logger.info("-" * 80)

        # HD相关配置
        if hasattr(self.args, 'use_hd_memory') and self.args.use_hd_memory:
            self.logger.info("⚡ HD记忆配置:")
            self.logger.info(f"  - HD维度: {getattr(self.args, 'hd_dim', 256)}")
            self.logger.info(f"  - HD权重: {getattr(self.args, 'hd_memory_weight', 0.3)}")
            self.logger.info(f"  - 更新HD记忆: {getattr(self.args, 'update_hd_memory', True)}")
        self.logger.info("=" * 80)

    def _print_epoch_info(self, epoch, log, epoch_time):
        """打印epoch详细信息"""
        train_loss = log.get('train_loss', 0)

        # 验证集指标
        val_bleu1 = log.get('val_BLEU_1', 0)
        val_bleu2 = log.get('val_BLEU_2', 0)
        val_bleu3 = log.get('val_BLEU_3', 0)
        val_bleu4 = log.get('val_BLEU_4', 0)
        val_rouge = log.get('val_ROUGE_L', 0)
        val_meteor = log.get('val_METEOR', 0)
        val_cider = log.get('val_CIDEr', 0)

        # 测试集指标
        test_bleu1 = log.get('test_BLEU_1', 0)
        test_bleu2 = log.get('test_BLEU_2', 0)
        test_bleu3 = log.get('test_BLEU_3', 0)
        test_bleu4 = log.get('test_BLEU_4', 0)
        test_rouge = log.get('test_ROUGE_L', 0)
        test_meteor = log.get('test_METEOR', 0)
        test_cider = log.get('test_CIDEr', 0)

        # 计算进度
        progress = f"[{epoch:3d}/{self.epochs:3d}]"
        progress_bar = self._create_progress_bar(epoch, self.epochs, width=30)

        self.logger.info("")
        self.logger.info("▬" * 60)
        self.logger.info(f"训练轮数 {progress}: {progress_bar} {100 * epoch / self.epochs:.1f}%")
        self.logger.info(f"第{epoch:3d}轮训练平均损失: {train_loss:.4f}")
        self.logger.info(f"训练耗时: {epoch_time:.1f}s")
        self.logger.info("")
        self.logger.info("▬" * 60)

        # 验证集指标表格
        self.logger.info("📊 验证集指标:")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'指标':<10} │ {'BLEU-1':<8} │ {'BLEU-2':<8} │ {'BLEU-3':<8} │ {'BLEU-4':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(
            f"  │ {'分数':<10} │ {val_bleu1:8.4f} │ {val_bleu2:8.4f} │ {val_bleu3:8.4f} │ {val_bleu4:8.4f} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'其他指标':<10} │ {'ROUGE-L':<8} │ {'METEOR':<8} │ {'CIDEr':<8} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {val_rouge:8.4f} │ {val_meteor:8.4f} │ {val_cider:8.4f} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)

        # 测试集指标表格
        self.logger.info("")
        self.logger.info("📈 测试集指标:")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'指标':<10} │ {'BLEU-1':<8} │ {'BLEU-2':<8} │ {'BLEU-3':<8} │ {'BLEU-4':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(
            f"  │ {'分数':<10} │ {test_bleu1:8.4f} │ {test_bleu2:8.4f} │ {test_bleu3:8.4f} │ {test_bleu4:8.4f} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'其他指标':<10} │ {'ROUGE-L':<8} │ {'METEOR':<8} │ {'CIDEr':<8} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {test_rouge:8.4f} │ {test_meteor:8.4f} │ {test_cider:8.4f} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)

        # 显示最佳指标对比
        self.logger.info("")
        self.logger.info("🏆 当前最佳 vs 历史最佳:")
        self.logger.info("  [验证集]")
        self.logger.info(f"    BLEU-4: 当前 {val_bleu4:.4f} | 最佳 {self.best_recorder['val'].get('val_BLEU_4', 0):.4f}")
        self.logger.info(f"    ROUGE-L: 当前 {val_rouge:.4f} | 最佳 {self.best_recorder['val'].get('val_ROUGE_L', 0):.4f}")

        self.logger.info("  [测试集]")
        self.logger.info(f"    BLEU-4: 当前 {test_bleu4:.4f} | 最佳 {self.best_recorder['test'].get('test_BLEU_4', 0):.4f}")
        self.logger.info(
            f"    ROUGE-L: 当前 {test_rouge:.4f} | 最佳 {self.best_recorder['test'].get('test_ROUGE_L', 0):.4f}")

        self.logger.info("-" * 60)

    def _print_best_model_info(self):
        """打印最佳模型信息"""
        info = self.best_model_info
        self.logger.info("🌟" + "=" * 60)
        self.logger.info("🌟 发现新的最佳模型！")
        self.logger.info(f"🌟 最佳模型保存: {info['path']}")
        self.logger.info(f"🌟 最佳epoch: {info['epoch']}, {info['metric_name']}: {info['metric_value']:.4f}")
        self.logger.info(f"🌟 保存时间: {info['timestamp']}")
        self.logger.info("🌟" + "=" * 60)

    def _print_training_summary(self, total_time):
        """打印训练总结"""
        self.logger.info("")
        self.logger.info("🎉" + "=" * 60)
        self.logger.info("🎉 训练完成！")
        self.logger.info(f"🎉 总训练时间: {total_time / 3600:.2f} 小时")
        self.logger.info(f"🎉 最终保存目录: {self.checkpoint_dir}")
        self.logger.info("🎉" + "=" * 60)

    def _print_best_recorder(self):
        """打印最佳记录"""
        self.logger.info("")
        self.logger.info("🏆 最佳结果记录:")

        # 验证集最佳记录
        self.logger.info("  [验证集]")
        val_metrics = [
            ('BLEU-1', 'val_BLEU_1'),
            ('BLEU-2', 'val_BLEU_2'),
            ('BLEU-3', 'val_BLEU_3'),
            ('BLEU-4', 'val_BLEU_4'),
            ('ROUGE-L', 'val_ROUGE_L'),
            ('METEOR', 'val_METEOR'),
            ('CIDEr', 'val_CIDEr')
        ]

        for display_name, metric_key in val_metrics:
            if metric_key in self.best_recorder['val']:
                value = self.best_recorder['val'][metric_key]
                if value not in [inf, -inf]:
                    self.logger.info(f"    {display_name:10s}: {value:.4f}")

        # 测试集最佳记录
        self.logger.info("  [测试集]")
        test_metrics = [
            ('BLEU-1', 'test_BLEU_1'),
            ('BLEU-2', 'test_BLEU_2'),
            ('BLEU-3', 'test_BLEU_3'),
            ('BLEU-4', 'test_BLEU_4'),
            ('ROUGE-L', 'test_ROUGE_L'),
            ('METEOR', 'test_METEOR'),
            ('CIDEr', 'test_CIDEr')
        ]

        for display_name, metric_key in test_metrics:
            if metric_key in self.best_recorder['test']:
                value = self.best_recorder['test'][metric_key]
                if value not in [inf, -inf]:
                    self.logger.info(f"    {display_name:10s}: {value:.4f}")

    def _create_progress_bar(self, current, total, width=30):
        """创建进度条"""
        progress = int(width * current / total)
        bar = "█" * progress + "░" * (width - progress)
        return bar

    def _record_best(self, log):
        """记录最佳结果"""
        # 更新验证集最佳结果
        for key, value in log.items():
            if key.startswith('val_'):
                if key in self.best_recorder['val']:
                    if (self.mnt_mode == 'min' and value <= self.best_recorder['val'][key]) or \
                            (self.mnt_mode == 'max' and value >= self.best_recorder['val'][key]):
                        self.best_recorder['val'][key] = value
                else:
                    self.best_recorder['val'][key] = value

        # 更新测试集最佳结果
        for key, value in log.items():
            if key.startswith('test_'):
                if key in self.best_recorder['test']:
                    if (self.mnt_mode == 'min' and value <= self.best_recorder['test'][key]) or \
                            (self.mnt_mode == 'max' and value >= self.best_recorder['test'][key]):
                        self.best_recorder['test'][key] = value
                else:
                    self.best_recorder['test'][key] = value

    def _log_to_file(self, epoch, log, is_best=False):
        """记录到文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}:\n")
            f.write(f"  Train Loss: {log.get('train_loss', 0):.4f}\n")

            # 验证集指标
            f.write("  Validation Metrics:\n")
            for key, value in log.items():
                if key.startswith('val_'):
                    f.write(f"    {key}: {value:.4f}\n")

            # 测试集指标
            f.write("  Test Metrics:\n")
            for key, value in log.items():
                if key.startswith('test_'):
                    f.write(f"    {key}: {value:.4f}\n")

            if is_best:
                f.write(
                    f"  ✅ Best Model Saved: {self.best_model_info['metric_name']}={self.best_model_info['metric_value']:.4f}\n")
            f.write("-" * 50 + "\n")

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("警告: 无可用GPU，将使用CPU训练")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"警告: 配置使用 {n_gpu_use} 个GPU，但只有 {n_gpu} 个可用")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, prefix=''):
        """保存检查点"""
        try:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
                'best_model_info': self.best_model_info,
                'best_recorder': self.best_recorder
            }

            if save_best:
                # 保存最佳模型
                filename = os.path.join(self.checkpoint_dir, 'model_best.pth')

                # 先保存到临时文件，然后重命名
                temp_filename = filename + '.tmp'

                try:
                    # 使用新的序列化格式
                    torch.save(state, temp_filename, _use_new_zipfile_serialization=True)
                    os.rename(temp_filename, filename)
                    self.logger.info(f"💾 已保存最佳模型: {filename}")
                except Exception as e:
                    self.logger.error(f"❌ 保存最佳模型失败: {e}")

                    # 尝试只保存模型参数
                    try:
                        simple_state = {
                            'epoch': epoch,
                            'state_dict': self.model.state_dict()
                        }
                        torch.save(simple_state, filename + '.simple.pth', _use_new_zipfile_serialization=True)
                        self.logger.info(f"✅ 已保存简化版最佳模型: {filename}.simple.pth")
                    except Exception as e2:
                        self.logger.error(f"❌ 简化版保存也失败: {e2}")

                # 不保存额外的指标副本，减少I/O

            else:
                # 保存检查点
                if prefix:
                    filename = os.path.join(self.checkpoint_dir, f'{prefix}checkpoint_epoch{epoch}.pth')
                else:
                    filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')

                # 只保存关键检查点
                if epoch % 5 == 0 or epoch == self.epochs:
                    temp_filename = filename + '.tmp'
                    try:
                        # 使用新的序列化格式
                        torch.save(state, temp_filename, _use_new_zipfile_serialization=True)
                        os.rename(temp_filename, filename)
                        self.logger.info(f"💾 已保存检查点: {filename}")
                    except Exception as e:
                        self.logger.error(f"❌ 保存检查点失败: {e}")

        except Exception as e:
            self.logger.error(f"❌ 保存检查点时发生未知错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"从检查点恢复: {resume_path}")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'best_model_info' in checkpoint:
            self.best_model_info = checkpoint['best_model_info']

        if 'best_recorder' in checkpoint:
            self.best_recorder = checkpoint['best_recorder']

        self.logger.info(f"从 epoch {self.start_epoch} 恢复训练")
        self.logger.info(f"之前的最佳指标: {self.mnt_best}")


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # 记录数据集大小
        self.train_size = len(train_dataloader.dataset)
        self.val_size = len(val_dataloader.dataset)
        self.test_size = len(test_dataloader.dataset)

        self.logger.info(f"📊 数据集大小: 训练集={self.train_size}, 验证集={self.val_size}, 测试集={self.test_size}")

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.logger.info(f"[{epoch}/{self.epochs}] 开始训练...")

        train_loss = 0
        self.model.train()

        # 训练进度条
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"训练 [{epoch}/{self.epochs}]",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )

        for batch_idx, (images_id, images, reports_ids, reports_masks) in pbar:
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)

            # 前向传播
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新进度条
            avg_loss = train_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch': f'{batch_idx + 1}/{len(self.train_dataloader)}'})

            # 每log_period个batch记录一次
            if (batch_idx + 1) % self.args.log_period == 0:
                self.logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Loss: {avg_loss:.4f}")

        pbar.close()

        avg_train_loss = train_loss / len(self.train_dataloader)
        log = {'train_loss': avg_train_loss}

        # 验证阶段
        val_metrics = self._evaluate_epoch(epoch, 'validation', self.val_dataloader)
        log.update(val_metrics)

        # 测试阶段
        test_metrics = self._evaluate_epoch(epoch, 'test', self.test_dataloader)
        log.update(test_metrics)

        # 学习率调整
        self.lr_scheduler.step()

        return log

    def _evaluate_epoch(self, epoch, mode, dataloader):
        """评估阶段"""
        if mode == 'validation':
            desc = f"验证 [{epoch}/{self.epochs}]"
            prefix = 'val_'
            dataset_size = self.val_size
        else:
            desc = f"测试 [{epoch}/{self.epochs}]"
            prefix = 'test_'
            dataset_size = self.test_size

        self.model.eval()
        with torch.no_grad():
            gts, res = [], []

            # 评估进度条
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=desc,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                ncols=100
            )

            for batch_idx, (images_id, images, reports_ids, reports_masks) in pbar:
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                res.extend(reports)
                gts.extend(ground_truths)

                # 更新进度条
                processed = len(gts)
                pbar.set_postfix({'已处理': f'{processed}/{dataset_size}'})

            pbar.close()

            # 计算指标
            metrics = self.metric_ftns({i: [gt] for i, gt in enumerate(gts)},
                                       {i: [re] for i, re in enumerate(res)})

            # 添加前缀
            prefixed_metrics = {f'{prefix}{k}': v for k, v in metrics.items()}

            return prefixed_metrics