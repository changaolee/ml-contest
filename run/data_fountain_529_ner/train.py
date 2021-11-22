from paddlenlp.transformers import BertTokenizer, BertForTokenClassification
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from model.data_fountain_529_ner import BertCrfForTokenClassification
from data_process.data_fountain_529_ner import DataFountain529NerDataProcessor
from dataset.data_fountain_529_ner import DataFountain529NerDataset
from trainer.data_fountain_529_ner import DataFountain529NerTrainer
from utils.config_utils import get_config, CONFIG_PATH
from bunch import Bunch
import os


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_ner.json"))

    # 原始数据预处理
    data_processor = DataFountain529NerDataProcessor(config)
    data_processor.process()
    config = data_processor.config

    # 获取全部分类标签
    config.label_list = DataFountain529NerDataset.get_labels()

    folds = [1] if config.k_fold == 0 else range(1, config.k_fold + 1)
    for fold in folds:
        config.fold = fold

        # 获取训练集、开发集
        train_ds, dev_ds = DataFountain529NerDataset(config).load_data(
            fold=config.fold, splits=['train', 'dev'], lazy=False
        )

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(config.model_name, config)

        # 获取训练器
        trainer = DataFountain529NerTrainer(
            model=model, tokenizer=tokenizer, train_ds=train_ds, dev_ds=dev_ds, config=config
        )

        # 开始训练
        trainer.train()


def get_model_and_tokenizer(model_name: str, config: Bunch):
    model, tokenizer = None, None
    logger = config.logger
    if model_name == "bert_base":
        model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "bert_crf":
        bert = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        model = BertCrfForTokenClassification(bert)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "ernie_base":
        model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(config.label_list))
        tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    else:
        logger.error("load model error: {}.".format(model_name))
    return model, tokenizer


if __name__ == "__main__":
    train()
