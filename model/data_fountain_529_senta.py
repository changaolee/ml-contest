from base.model_base import ModelBase
from bunch import Bunch


class DataFountain529SentaModel(ModelBase):
    """
    df-529 情感分类模型
    """

    def __init__(self, config: Bunch):
        super().__init__(config)

    def build_model(self):
        pass
