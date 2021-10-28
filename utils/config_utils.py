from bunch import Bunch
from utils.utils import mkdir_if_not_exist, get_logger
import json
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))

DATA_PATH = os.path.join(ROOT_PATH, "data")
CONFIG_PATH = os.path.join(ROOT_PATH, "config")


def get_config(json_file):
    """
    获取配置类
    :param json_file: 配置 json 文件
    :return: 配置类
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    config.tb_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "tensorboard")  # 训练可视化
    config.ckpt_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "checkpoint")  # 模型
    config.log_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "log")  # 日志

    mkdir_if_not_exist(config.tb_dir)
    mkdir_if_not_exist(config.ckpt_dir)
    mkdir_if_not_exist(config.log_dir)

    config.logger = get_logger(config.log_dir, config.exp_name)

    return config
