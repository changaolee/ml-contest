from dotmap import DotMap
from utils.utils import mkdir_if_not_exist, get_logger
import json
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))

DATA_PATH = os.path.join(ROOT_PATH, "data")
CONFIG_PATH = os.path.join(ROOT_PATH, "config")
RESOURCE_PATH = os.path.join(ROOT_PATH, "resource")


def get_config(json_file: str, mode: str = ""):
    """
    获取配置类
    :param json_file: 配置 json 文件
    :param mode: train / predict
    :return: 配置类
    """
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)
    if mode:
        config = DotMap({**config.toDict(), **config[mode].toDict()})

    config.ckpt_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "checkpoint")  # 模型
    config.log_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "log")  # 日志
    config.vis_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "visual_log")  # 可视化
    config.res_dir = os.path.join(ROOT_PATH, "experiment", config.exp_name, "result")  # 结果

    mkdir_if_not_exist(config.ckpt_dir)
    mkdir_if_not_exist(config.log_dir)
    mkdir_if_not_exist(config.vis_dir)
    mkdir_if_not_exist(config.res_dir)

    config.logger = get_logger(config.log_dir, config.exp_name)

    return config
