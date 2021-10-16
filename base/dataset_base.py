from paddle.io import Dataset
from bunch import Bunch


class DatasetBase(Dataset):
    """
    数据集基类
    """

    def __init__(self, config: Bunch, mode: str = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode
