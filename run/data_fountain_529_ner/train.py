from data_process.data_fountain_529_ner import DataFountain529NerDataProcessor
from dataset.data_fountain_529_ner import DataFountain529NerDataset
from model.data_fountain_529_ner import get_model_and_tokenizer
from trainer.data_fountain_529_ner import DataFountain529NerTrainer
from utils.config_utils import get_config, CONFIG_PATH
import os


def train():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_ner.json"), "train")

    # 原始数据预处理
    data_processor = DataFountain529NerDataProcessor(config)
    data_processor.process()

    # 获取全部分类标签
    config.label_list = DataFountain529NerDataset.get_labels()

    folds = [0] if config.k_fold == 0 else range(config.k_fold)
    for fold in folds:
        config.fold = fold

        # 获取训练集、开发集
        train_ds, dev_ds = DataFountain529NerDataset(config).load_data(
            fold=config.fold, splits=['train', 'dev'], lazy=False
        )

        # 加载 model 和 tokenizer
        model, tokenizer, config = get_model_and_tokenizer(config.model_name, config)

        # 获取训练器
        trainer = DataFountain529NerTrainer(
            model=model, tokenizer=tokenizer, train_ds=train_ds, dev_ds=dev_ds, config=config
        )

        # 开始训练
        trainer.train()


if __name__ == "__main__":
    train()
