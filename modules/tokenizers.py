import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr

        # 加载标注文件
        with open(self.ann_path, 'r') as f:
            self.ann = json.load(f)

        print(f"✅ 标注文件加载成功，共有 {len(self.ann)} 个样本")

        # 检查标注文件结构
        if isinstance(self.ann, dict):
            if 'train' in self.ann:
                # 格式: {'train': [...], 'val': [...], 'test': [...]}
                train_data = self.ann['train']
            else:
                # 格式: 可能是其他格式
                train_data = self.ann
        else:
            # 格式: 直接是列表
            train_data = self.ann

        # 采样查看第一个样本的结构
        if train_data and isinstance(train_data, list) and len(train_data) > 0:
            first_sample = train_data[0]
            print(f"📊 样本结构示例: {list(first_sample.keys())}")

        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        counter = Counter()

        # 确定训练数据
        if isinstance(self.ann, dict) and 'train' in self.ann:
            train_data = self.ann['train']
        else:
            train_data = self.ann

        print(f"📈 处理训练数据，共 {len(train_data)} 个样本")

        for i, example in enumerate(train_data):
            # 调试：打印前几个样本
            if i < 3:
                print(f"样本 {i} 的键: {list(example.keys())}")

            # 尝试不同的键名
            report_text = None

            # 可能的键名列表
            possible_keys = ['report', 'reports', 'caption', 'text', 'findings', 'impression']

            for key in possible_keys:
                if key in example:
                    if isinstance(example[key], list):
                        reports_list = example[key]
                    else:
                        reports_list = [example[key]]

                    for report in reports_list:
                        if isinstance(report, str):
                            report_text = self.clean_report(report)
                            break
                    if report_text:
                        break

            if report_text is None:
                # 如果还没找到，尝试其他可能的格式
                if 'sentences' in example:
                    reports_list = example['sentences']
                    for report in reports_list:
                        if isinstance(report, str):
                            report_text = self.clean_report(report)
                            break

                # 如果仍然找不到，跳过
                if report_text is None:
                    print(f"⚠️  警告: 样本 {i} 没有找到报告文本")
                    continue

            tokens = report_text.split()
            counter.update(tokens)

        # 创建词汇表
        tokens = [item for item in counter.items() if item[1] > self.threshold]
        tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
        vocab = ['<start>', '<end>', '<unk>', '<pad>'] + [item[0] for item in tokens]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}

        print(f"✅ 词汇表创建完成，大小: {len(vocab)}")
        print(f"📊 前10个高频词: {list(token2idx.keys())[4:14]}")

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        if not isinstance(report, str):
            return ""

        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub(r'[^\w\s]', '',
                                        t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                      '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        if not isinstance(report, str):
            return ""

        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub(r'[^\w\s]', '',
                                        t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                      '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        tokens = ['<start>'] + tokens + ['<end>']
        unk_id = self.token2idx['<unk>']
        ids = [self.token2idx.get(token, unk_id) for token in tokens]
        return ids

    def decode(self, ids):
        idx2word = self.idx2token
        words = []
        for i in ids:
            word = idx2word[i]
            if word not in ['<start>', '<pad>', '<unk>']:
                words.append(word)
            if word == '<end>':
                break
        return ' '.join(words)

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
