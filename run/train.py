from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
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

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()
    config = data_processor.config

    # 获取训练集、开发集
    train_ds, dev_ds = DataFountain529SentaDataset(config).load_data(splits=['train', 'dev'], lazy=False)

    # 加载模型
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
        return

    # 获取训练器
    trainer = DataFountain529SentaTrainer(model, train_ds=train_ds, dev_ds=dev_ds, config=config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    train()
