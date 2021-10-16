from paddle.nn import Layer
from bunch import Bunch


class ModelBase(Layer):
    """
    模型基类
    """

    def __init__(self, config: Bunch):
        super().__init__()
        self.config = config
