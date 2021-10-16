from bunch import Bunch


class DatasetBase(object):
    """
    数据加载的基类
    """

    def __init__(self, config: Bunch = None):
        self.config = config

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
