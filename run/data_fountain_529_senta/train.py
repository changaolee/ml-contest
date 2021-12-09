from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import get_model_and_tokenizer
from trainer.data_fountain_529_senta import DataFountain529SentaTrainer
from utils.config_utils import get_config, CONFIG_PATH
from dotmap import DotMap
import argparse
import os


def train(opt):
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"), "train")

    # baseline 配置重写
    if opt.baseline:
        config = DotMap({**config.toDict(), **config.baseline.toDict()})

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()

    folds = [0] if config.k_fold == 0 else range(config.k_fold)
    for fold in folds:
        config.fold = fold

        # 获取训练集、开发集
        train_ds, dev_ds = DataFountain529SentaDataset(config).load_data(
            fold=config.fold, splits=['train', 'dev'], lazy=False
        )

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(config)

        # 获取训练器
        trainer = DataFountain529SentaTrainer(
            model=model, tokenizer=tokenizer, train_ds=train_ds, dev_ds=dev_ds, config=config
        )

        # 开始训练
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', nargs='?', const=True, default=False, help='start baseline training')
    train(parser.parse_args())
