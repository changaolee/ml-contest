from paddlenlp.datasets import load_dataset
from dotmap import DotMap
import pandas as pd


class DataFountain529NerDataset(object):
    """
    df-529 NER 数据集
    """

    def __init__(self, config: DotMap):
        super().__init__()
        self.config = config

    def load_data(self, fold=0, splits=None, lazy=None):
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
            text, bio, qid = line.get("text", ""), line.get("BIO_anno", ""), line.get("id", "")
            yield {"tokens": list(text), "labels": bio.split(), "qid": qid}

    @staticmethod
    def get_labels():
        return ["B-BANK", "I-BANK",
                "B-PRODUCT", "I-PRODUCT",
                "B-COMMENTS_N", "I-COMMENTS_N",
                "B-COMMENTS_ADJ", "I-COMMENTS_ADJ", "O"]
