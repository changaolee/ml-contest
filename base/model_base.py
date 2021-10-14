class ModelBase(object):
    """
    模型基类
    """

    def __init__(self, config):
        self.config = config  # 配置
        self.model = None  # 模型

    def save(self, checkpoint_path):
        """
        存储 checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model ...")
        self.model.save_weights(checkpoint_path)
        print("[INFO] Model saved")

    def load(self, checkpoint_path):
        """
        加载 checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("[INFO] Model loaded")

    def build_model(self):
        """
        构建模型
        """
        raise NotImplementedError
