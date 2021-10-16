from paddle.io import Dataset


class DatasetBase(Dataset):
    """
    自定义数据集的基类
    """

    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
