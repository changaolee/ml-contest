from paddlenlp.datasets import load_dataset
from dotmap import DotMap
import pandas as pd


class DataFountain529SentaDataset(object):
    """
    df-529 情感分类数据集（1：正面，0：负面，2：中立）
    """

    def __init__(self, config: DotMap):
        super().__init__()
        self.config = config

    def load_data(self, fold=1, splits=None, lazy=None):
        result = []
        for split in splits:
            path = self.config.splits.get(split).format(fold)
            ds = load_dataset(self.read, data_path=path, lazy=lazy)
            result.append(ds)
        return result

    @staticmethod
    def read(data_path):
        df = pd.read_csv(data_path, encoding="utf-8")
        for idx, line in df.iterrows():
            text, label, qid = line.get("text", ""), line.get("label", ""), line.get("id", "")
            yield {"text": text, "label": label, "qid": qid}
