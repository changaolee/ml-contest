from base.dataset_base import DatasetBase
from utils.config_utils import get_config
from utils.utils import dataset_split
from bunch import Bunch
import pandas as pd
import os

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../data"))
CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


class DataFountain529Senta(DatasetBase):
    """
    情感分类（1：正面，0：负面，2：中立）
    ref: https://www.datafountain.cn/competitions/529/datasets
    """

    def __init__(self, config: Bunch, mode: str, shuffle: bool):
        super().__init__(config)

        def load_data_from_source(path):
            dataset = []
            df = pd.read_csv(path)
            for idx, line in df.iterrows():
                text, label = line.get("text", ""), line.get("class", "")
                sample = {"text": text, "label": label}
                dataset.append(sample)
            return dataset

        train_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.train_filename)
        test_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.test_filename)

        self.train_dataset, self.dev_dataset = dataset_split(
            dataset=load_data_from_source(train_data_path),
            dev_prop=config.dev_prop,
            shuffle=shuffle
        )
        self.test_dataset = load_data_from_source(test_data_path)
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset)
        elif self.mode == 'dev':
            return len(self.dev_dataset)
        return len(self.test_dataset)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_dataset[idx]
        elif self.mode == 'dev':
            return self.dev_dataset[idx]
        return self.test_dataset[idx]


if __name__ == "__main__":
    conf = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))
    df529 = DataFountain529Senta(conf, "train", True)
    print(len(df529))
    for i in range(5):
        print(df529[i])
