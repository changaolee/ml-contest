from paddlenlp.transformers import BertTokenizer
from paddlenlp.datasets import MapDataset
from paddle.io import DataLoader, BatchSampler
from paddle.optimizer import AdamW
from paddle import nn
from bunch import Bunch
from functools import partial
import paddle.nn.functional as F
import numpy as np


class DataFountain529SentaTrainer(object):
    def __init__(self, model: nn.Layer, train_data: MapDataset, dev_data: MapDataset, config: Bunch):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.pretrain_model)

    def train(self):
        # 转换至模型的输入
        self.train_data = self.train_data.map(partial(
            self._convert_sample, tokenizer=self.tokenizer, max_len=self.config.max_len
        ))
        self.dev_data = self.dev_data.map(partial(
            self._convert_sample, tokenizer=self.tokenizer, max_len=self.config.max_len
        ))

        train_sampler = BatchSampler(dataset=self.train_data, batch_size=self.config.train_batch_size, shuffle=True)
        train_data_loader = DataLoader(dataset=self.train_data, batch_sampler=train_sampler)

        # 定义优化器
        optimizer = AdamW(learning_rate=self.config.learning_rate, parameters=self.model.parameters())

        # 定义损失函数
        criterion = nn.loss.CrossEntropyLoss()

        # 模型训练
        for input_ids, attention_mask, labels in train_data_loader():
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    @staticmethod
    def _convert_sample(sample, tokenizer, max_len):
        encoded_inputs = tokenizer(text=sample["text"], max_seq_len=max_len)
        input_ids, attention_mask = encoded_inputs["input_ids"], [1] * len(encoded_inputs["input_ids"])
        input_ids += [0] * max(max_len - len(input_ids), 0)
        attention_mask += [0] * max(max_len - len(attention_mask), 0)
        return tuple([np.array(x, dtype="int64") for x in [input_ids, attention_mask, [sample["label"]]]])
