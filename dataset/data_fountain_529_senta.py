from paddle.io import Dataset
from utils.config_utils import get_config
from utils.utils import dataset_split
from bunch import Bunch
import pandas as pd
import os

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../data"))
CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


class DataFountain529SentaDataset(Dataset):
    """
    df-529 情感分类数据集（1：正面，0：负面，2：中立）
    """

    def __init__(self, config: Bunch, mode: str, shuffle: bool = False):
        super().__init__()

        self.config = config
        self.mode = mode
        self.shuffle = shuffle

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

        if self.mode == "train":
            self.dataset, _ = dataset_split(
                dataset=load_data_from_source(train_data_path),
                dev_prop=self.config.dev_prop,
                shuffle=self.shuffle
            )
        elif self.mode == "dev":
            _, self.dataset = dataset_split(
                dataset=load_data_from_source(train_data_path),
                dev_prop=self.config.dev_prop,
                shuffle=self.shuffle
            )
        else:
            self.dataset = load_data_from_source(test_data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    conf = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))
    df529 = DataFountain529SentaDataset(conf, "dev", True)
    print(len(df529))
    for i in range(5):
        print(df529[i])
