import logging

from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle import nn
from bunch import Bunch
from functools import partial
from utils.utils import create_data_loader, mkdir_if_not_exist
import paddle.nn.functional as F
import numpy as np
import paddle
import os
import csv


class DataFountain529SentaInfer(object):
    label_map = {0: "0", 1: "1", 2: "2"}

    def __init__(self,
                 model: nn.Layer,
                 tokenizer: PretrainedTokenizer,
                 test_ds: MapDataset,
                 config: Bunch):
        self.model = model
        self.tokenizer = tokenizer
        self.test_ds = test_ds
        self.config = config
        self.logger = self.config.logger
        self._gen_data_loader()
        self._load_model()

    def _gen_data_loader(self):
        # 将数据处理成模型可读入的数据格式
        trans_func = partial(
            self.convert_example,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Stack()  # qid
        ): [data for data in fn(samples)]

        self.test_data_loader = create_data_loader(
            self.test_ds,
            mode='test',
            batch_size=self.config.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

    def _load_model(self):
        model_params_path = os.path.join(self.config.model_path, "model.pdparams")
        if os.path.isfile(model_params_path):
            # 加载模型参数
            state_dict = paddle.load(model_params_path)
            self.model.set_dict(state_dict)
            self.logger.info("Loaded parameters from {}".format(model_params_path))

    def predict(self):
        result = []
        # 切换 model 模型为评估模式，关闭 dropout 等随机因素
        self.model.eval()
        for batch in self.test_data_loader:
            input_ids, token_type_ids, qids = batch
            # 喂数据给 model
            logits = self.model(input_ids, token_type_ids)
            # 预测分类
            probs = F.softmax(logits, axis=-1)
            idx = paddle.argmax(probs, axis=1).numpy()
            idx = idx.tolist()
            labels = [self.label_map[i] for i in idx]
            qids = qids.numpy().tolist()
            result.extend(zip(qids, labels))
        self.save_result(result)

    def save_result(self, result):
        res_dir = os.path.join(self.config.res_dir, self.config.model_name)
        mkdir_if_not_exist(res_dir)
        # 写入预测结果
        with open(os.path.join(res_dir, "result.csv"), "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "class"])
            for line in result:
                qid, label = line
                writer.writerow([qid[0], label])

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len=512):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["qid"]]]])
