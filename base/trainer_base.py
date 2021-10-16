class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError
