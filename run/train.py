from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import DataFountain529SentaBaselineModel
from trainer.data_fountain_529_senta import DataFountain529SentaTrainer
from paddlenlp.datasets import MapDataset
from utils.config_utils import get_config
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))

    # 获取训练集、验证集
    train_ds = MapDataset(DataFountain529SentaDataset(config, "train"))
    dev_ds = MapDataset(DataFountain529SentaDataset(config, "dev"))

    # 获取模型
    model = DataFountain529SentaBaselineModel.from_pretrained(config.pretrain_model, config=config)  # baseline

    # 获取训练器
    trainer = DataFountain529SentaTrainer(model, train_data=train_ds, dev_data=dev_ds, config=config)
    trainer.train()


if __name__ == "__main__":
    train()
