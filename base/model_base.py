from paddle.nn import Layer


class ModelBase(Layer):
    """
    模型基类
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
