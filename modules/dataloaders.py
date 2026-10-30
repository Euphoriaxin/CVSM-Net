# modules/dataloaders.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # 训练集的图像变换（支持增强）
        if split == 'train':
            if getattr(args, 'augment', False):
                print(f"✅ 启用数据增强 - 强度: {args.aug_intensity}, 旋转: ±{args.aug_rotation}°, 裁剪缩放: {args.aug_crop_scale}")

                # 构建变换列表
                transform_list = [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(args.aug_crop_scale, 1.0)),
                    transforms.RandomRotation(degrees=(-args.aug_rotation, args.aug_rotation)),
                    transforms.ColorJitter(
                        brightness=args.aug_intensity * 0.5,
                        contrast=args.aug_intensity,
                        saturation=0.0,
                        hue=0.0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ]

                # ✅ 根据 aug_flip 决定是否添加水平翻转
                if getattr(args, 'aug_flip', False):
                    print(f"✅ 启用随机水平翻转")
                    transform_list.insert(3, transforms.RandomHorizontalFlip(p=0.5))

                self.transform = transforms.Compose(transform_list)

            else:
                # 原始训练变换（无额外增强）
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ])
        else:
            # 验证集 / 测试集：固定尺寸，无增强
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    # 自定义批次打包函数：把多个长度不一的样本，打包成统一长度、模型可直接训练的批次数据
    @staticmethod
    def collate_fn(data):
        # 1. 解压单个样本返回的 5 个字段
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
        # 2. 图像批次堆叠：[batch_size, 2, 3, 224, 224]
        image_batch = torch.stack(image_batch, 0)
        # 3. 找到当前批次中最长的文本长度
        max_seq_length = max(seq_lengths_batch)
        # 4. 创建全0数组，用于填充对齐
        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        # 5. 把每个报告填充到最大长度
        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks
        # 6. 转成PyTorch张量，返回最终批次
        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


# class R2DataLoader(DataLoader):
#     def __init__(self, args, tokenizer, split, shuffle):
#         self.args = args
#         self.dataset_name = args.dataset_name
#         self.batch_size = args.batch_size
#         self.shuffle = shuffle
#         self.num_workers = args.num_workers
#         self.tokenizer = tokenizer
#         self.split = split

#         # 训练集的图像变换（支持增强）
#         if split == 'train':
#             if getattr(args, 'augment', False):
#                 print(f"✅ 启用数据增强 - 强度: {args.aug_intensity}, 旋转: ±{args.aug_rotation}°, 裁剪缩放: {args.aug_crop_scale}")
#                 self.transform = transforms.Compose([
#                     transforms.Resize((256, 256)),     # 统一缩放到256×256
#                     transforms.RandomResizedCrop(
#                         224,
#                         scale=(args.aug_crop_scale, 1.0)    # 随机裁剪
#                     ),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.RandomRotation(degrees=(-args.aug_rotation, args.aug_rotation)),     # 随机旋转
#                     # 颜色抖动：亮度、对比度变化（胸片不改变饱和度、色调）
#                     transforms.ColorJitter(
#                         brightness=args.aug_intensity * 0.5,
#                         contrast=args.aug_intensity,
#                         saturation=0.0,
#                         hue=0.0
#                     ),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.485, 0.456, 0.406),
#                                          (0.229, 0.224, 0.225))
#                 ])
#             else:
#                 # 原始训练变换（无额外增强）
#                 self.transform = transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.RandomCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.485, 0.456, 0.406),
#                                          (0.229, 0.224, 0.225))
#                 ])
#         else:
#             # 验证集 / 测试集：固定尺寸，无增强
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))
#             ])

#         if self.dataset_name == 'iu_xray':
#             self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
#         else:
#             self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers
#         }
#         super().__init__(**self.init_kwargs)

#     # 自定义批次打包函数：把多个长度不一的样本，打包成统一长度、模型可直接训练的批次数据
#     @staticmethod
#     def collate_fn(data):
#         # 1. 解压单个样本返回的 5 个字段
#         image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
#         # 2. 图像批次堆叠：[batch_size, 2, 3, 224, 224]
#         image_batch = torch.stack(image_batch, 0)
#         # 3. 找到当前批次中最长的文本长度
#         max_seq_length = max(seq_lengths_batch)
#         # 4. 创建全0数组，用于填充对齐
#         target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
#         target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
#          # 5. 把每个报告填充到最大长度
#         for i, report_ids in enumerate(report_ids_batch):
#             target_batch[i, :len(report_ids)] = report_ids

#         for i, report_masks in enumerate(report_masks_batch):
#             target_masks_batch[i, :len(report_masks)] = report_masks
#         # 6. 转成PyTorch张量，返回最终批次
#         return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)