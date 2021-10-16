from paddle.io import Dataset


class DatasetBase(Dataset):
    """
    数据集基类
    """

    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
