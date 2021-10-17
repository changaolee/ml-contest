from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import DataFountain529SentaBertBaselineModel, DataFountain529SentaSkepBaselineModel
from trainer.data_fountain_529_senta import DataFountain529SentaTrainer
from paddlenlp.datasets import MapDataset
from utils.config_utils import get_config
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))
    logger = config.logger

    # 获取训练集、验证集
    train_ds = MapDataset(DataFountain529SentaDataset(config, "train"))
    dev_ds = MapDataset(DataFountain529SentaDataset(config, "dev"))

    # 加载预训练模型
    pretrained_model_name = config.pretrained_model_name
    if pretrained_model_name in ["bert-wwm-chinese"]:
        model = DataFountain529SentaBertBaselineModel.from_pretrained(  # bert baseline
            pretrained_model_name, config=config
        )
    elif pretrained_model_name in ["skep_ernie_1.0_large_ch"]:
        model = DataFountain529SentaSkepBaselineModel.from_pretrained(  # skep baseline
            pretrained_model_name, config=config
        )
    else:
        logger.error("load pretrain_model {} error.".format(pretrained_model_name))
        return False

    # 获取训练器
    trainer = DataFountain529SentaTrainer(model, train_data=train_ds, dev_data=dev_ds, config=config)
    trainer.train()


if __name__ == "__main__":
    train()
