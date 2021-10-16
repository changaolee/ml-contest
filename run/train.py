from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import DataFountain529SentaBaselineModel
from trainer.data_fountain_529_senta import DataFountain529SentaTrainer
from paddlenlp.transformers import BertTokenizer
from paddlenlp.datasets import MapDataset
from utils.config_utils import get_config
from functools import partial
import numpy as np
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))

    # 获取原始训练、验证数据集
    train_ds = MapDataset(DataFountain529SentaDataset(config, "train"))
    dev_ds = MapDataset(DataFountain529SentaDataset(config, "dev"))

    def convert_sample(_sample, _tokenizer):
        _encoded_inputs = _tokenizer(text=_sample["text"], max_seq_len=config.max_len)
        _input_ids, _attention_mask = _encoded_inputs["input_ids"], [1] * len(_encoded_inputs["input_ids"])
        _input_ids += [0] * max(config.max_len - len(_input_ids), 0)
        _attention_mask += [0] * max(config.max_len - len(_attention_mask), 0)
        return tuple([np.array(x, dtype="int64") for x in [_input_ids, _attention_mask, [_sample["label"]]]])

    # 转换至模型的输入
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model)
    train_ds = train_ds.map(partial(convert_sample, tokenizer=tokenizer))
    dev_ds = dev_ds.map(partial(convert_sample, tokenizer=tokenizer))

    # 获取模型
    model = DataFountain529SentaBaselineModel(config)  # baseline

    # 获取训练器
    trainer = DataFountain529SentaTrainer(model, train_data=train_ds, dev_data=dev_ds, config=config)
    trainer.train()


if __name__ == "__main__":
    train()
