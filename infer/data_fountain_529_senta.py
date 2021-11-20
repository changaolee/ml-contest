from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle import nn
from bunch import Bunch
from functools import partial
from utils.utils import create_data_loader
import paddle.nn.functional as F
import numpy as np
import paddle
import os


class DataFountain529SentaInfer(object):

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
        model_params_path = self.config.model_path if os.path.isfile(self.config.model_path) \
            else os.path.join(self.config.model_path, "model.pdparams")
        if os.path.isfile(model_params_path):
            # 加载模型参数
            state_dict = paddle.load(model_params_path)
            self.model.set_dict(state_dict)
            self.logger.info("Loaded parameters from {}".format(model_params_path))
        else:
            self.logger.error("Loaded parameters error from {}".format(model_params_path))

    @paddle.no_grad()
    def predict(self):
        result = []
        # 切换 model 模型为评估模式，关闭 dropout 等随机因素
        self.model.eval()
        for step, batch in enumerate(self.test_data_loader, start=1):
            input_ids, token_type_ids, qids = batch
            # 喂数据给 model
            logits = self.model(input_ids, token_type_ids)
            # 预测分类
            probs = F.softmax(logits, axis=-1)
            qids = qids.flatten().numpy().tolist()
            probs = probs.numpy().tolist()
            result.extend(zip(qids, probs))
        return result

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len=512):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["qid"]]]])
