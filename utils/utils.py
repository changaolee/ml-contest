import os
import shutil
import logging
import random

logger = logging.getLogger(__name__)


def get_logger(log_dir: str, exp_name: str):
    """
    获取日志对象
    :param log_dir: 日志文件路径
    :param exp_name: 记录对象名称
    :return: 日志对象
    """
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename="{0}/{1}.log".format(log_dir, exp_name), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(exp_name)


def mkdir_if_not_exist(dir_name: str, is_delete: bool = False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            logger.info("文件夹 {} 已存在，删除文件夹".format(dir_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info("文件夹 {} 不存在，创建文件夹".format(dir_name))
        return True
    except Exception as e:
        logger.error("mkdir_if_not_exist:error:{}".format(e))
        return False


def dataset_split(dataset: list, dev_prop: float, shuffle: bool):
    """
    将原始数据集划分为测试集和验证集
    :param dataset: 原始数据集
    :param dev_prop: 验证集占比
    :param shuffle: 是否打乱顺序
    :return: 测试集和验证集
    """
    if shuffle:
        random.shuffle(dataset)
    idx = int(len(dataset) * dev_prop)
    train_dataset, dev_dataset = dataset[idx:], dataset[:idx]
    return train_dataset, dev_dataset
