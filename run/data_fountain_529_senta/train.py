from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification
from paddlenlp.transformers import SkepTokenizer, SkepForSequenceClassification
from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
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

    folds = [1] if config.k_fold == 0 else range(1, config.k_fold + 1)
    for fold in folds:
        config.fold = fold

        # 获取训练集、开发集
        train_ds, dev_ds = DataFountain529SentaDataset(config).load_data(
            fold=config.fold, splits=['train', 'dev'], lazy=False
        )

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(config.model_name, config)

        # 获取训练器
        trainer = DataFountain529SentaTrainer(
            model=model, tokenizer=tokenizer, train_ds=train_ds, dev_ds=dev_ds, config=config
        )

        # 开始训练
        trainer.train()


def get_model_and_tokenizer(model_name: str, config: Bunch):
    model, tokenizer = None, None
    logger = config.logger
    if model_name == "bert_baseline":
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=config.num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "skep_baseline":
        model = SkepForSequenceClassification.from_pretrained("skep_ernie_1.0_large_ch", num_classes=config.num_classes)
        tokenizer = SkepTokenizer.from_pretrained("skep_ernie_1.0_large_ch")
    else:
        logger.error("load model error: {}.".format(model_name))
    return model, tokenizer


if __name__ == "__main__":
    train()
