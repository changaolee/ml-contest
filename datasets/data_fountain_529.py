from base.data_loader_base import DataLoaderBase
from utils.config_utils import get_config
import pandas as pd
import os

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../data"))
CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "../config"))


class DataFountain529(DataLoaderBase):

    def __init__(self, config):
        super().__init__(config)

        def load_data_from_source(path):
            dataset = []
            df = pd.read_csv(path)
            for idx, line in df.iterrows():
                sample = {"text": line["text"], "label": line["class"]}
                dataset.append(sample)
            return dataset

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
