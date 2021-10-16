from paddle.io import Dataset
from paddle.nn import Layer


class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model: Layer, train_data: Dataset, dev_data: Dataset, config):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.config = config

    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError
