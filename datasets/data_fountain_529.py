from base.data_loader_base import DataLoaderBase
from paddlenlp.datasets import MapDataset
from utils.config_utils import get_config
import pandas as pd
import os

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../data"))
CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


class DataFountain529(DataLoaderBase):
    """
    情感分类（1：正面，0：负面，2：中立）
    ref: https://www.datafountain.cn/competitions/529/datasets
    """

    def __init__(self, config):
        super().__init__(config)

        def load_data_from_source(path):
            dataset = []
            df = pd.read_csv(path)
            for idx, line in df.iterrows():
                sample = {"text": line["text"], "label": line["class"]}
                dataset.append(sample)
            return dataset

        self.logger = self.config.logger
        self.label_list = ['0', '1', '2']
        self.dataset = load_data_from_source(
            os.path.join(DATA_PATH, self.config.exp_name, self.config.train_filename)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    df529 = DataFountain529(
        get_config(os.path.join(CONFIG_PATH, "data_fountain_529.json"))
    )
    print(len(df529))
    for i in range(5):
        print(df529[i])
