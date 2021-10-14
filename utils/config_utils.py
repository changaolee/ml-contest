from bunch import Bunch
from utils.utils import mkdir_if_not_exist
import json
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))


def get_config(json_file):
    """
    获取配置类
    :param json_file: 配置 json 文件
    :return: 配置类
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    config.tb_dir = os.path.join(ROOT_PATH, "experiments", config.exp_name, "logs")  # 日志
    config.cp_dir = os.path.join(ROOT_PATH, "experiments", config.exp_name, "checkpoints")  # 模型

    mkdir_if_not_exist(config.tb_dir)
    mkdir_if_not_exist(config.cp_dir)

    return config
