from paddlenlp.transformers import BertTokenizer, SkepTokenizer
from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import DataFountain529SentaBertBaselineModel, DataFountain529SentaSkepBaselineModel
from trainer.data_fountain_529_senta import DataFountain529SentaTrainer
from utils.config_utils import get_config, CONFIG_PATH
from bunch import Bunch
import os


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()
    config = data_processor.config

    # 获取训练集、开发集
    train_ds, dev_ds = DataFountain529SentaDataset(config).load_data(splits=['train', 'dev'], lazy=False)

    # 加载 model 和 tokenizer
    model, tokenizer = get_model_and_tokenizer(config.pretrained_model_name, config)

    # 获取训练器
    trainer = DataFountain529SentaTrainer(model, tokenizer=tokenizer, train_ds=train_ds, dev_ds=dev_ds, config=config)

    # 开始训练
    trainer.train()


def get_model_and_tokenizer(pretrained_model_name: str, config: Bunch):
    model, tokenizer = None, None
    logger = config.logger
    if pretrained_model_name in ["bert-wwm-chinese"]:
        model = DataFountain529SentaBertBaselineModel.from_pretrained(  # bert baseline
            pretrained_model_name, config=config
        )
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    elif pretrained_model_name in ["skep_ernie_1.0_large_ch"]:
        model = DataFountain529SentaSkepBaselineModel.from_pretrained(  # skep baseline
            pretrained_model_name, config=config
        )
        tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name)
    else:
        logger.error("load pretrain_model {} error.".format(pretrained_model_name))
    return model, tokenizer


if __name__ == "__main__":
    train()
