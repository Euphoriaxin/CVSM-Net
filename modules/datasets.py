# modules/datasets.py
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            # ===================== 【唯一修改点】 =====================
            # 旧代码：self.examples[i]['report']
            # 新代码：读取你JSON中干净的 reports 字段
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            # ==========================================================
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


# 专门处理 IU X-Ray 数据集（多视图，通常包含正位和侧位两张图像）
class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        # 确保image_path是列表
        if isinstance(image_path, str):
            image_path = [image_path]
        # 加载图像（固定返回2张图：适配模型输入要求）
        if len(image_path) == 1:
            # 单视图：复制同一张图片两次
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        else:
            # 如果有多个图片路径（IU X-Ray）：取前两个视图
            # 双视图：直接加载正位、侧位两张图
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # 拼接成张量：shape = [2, C, H, W]
        # 2=两张图，C=通道数(3)，H=高度，W=宽度
        image = torch.stack((image_1, image_2), 0)  # shape: [2, C, H, W]

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


# 专门处理 MIMIC-CXR 数据集
class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_paths = example['image_path']  # 这是一个列表，可能包含1-3个图片路径

        # 循环加载图像，最多加载2张
        loaded_images = []
        for i, img_path in enumerate(image_paths):
            if i >= 2:  # 最多只取前2个视图
                break
            # # 打开图像 + 预处理
            image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            loaded_images.append(image)

        # 处理成模型需要的固定格式（2个视图）
        if len(loaded_images) == 1:
            # 单视图：复制同一张图片
            image_final = torch.stack([loaded_images[0], loaded_images[0]], dim=0)
        elif len(loaded_images) >= 2:
            # 多视图：取前两个视图
            image_final = torch.stack([loaded_images[0], loaded_images[1]], dim=0)
        else:
            raise ValueError(f"无效的图片数量: {len(loaded_images)}")

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image_final, report_ids, report_masks, seq_length)
        return sample