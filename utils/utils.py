import os
import shutil
import logging
import paddle
import json
import hashlib

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


def create_data_loader(dataset,
                       trans_fn=None,
                       mode='train',
                       batch_size=1,
                       batchify_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    data_loader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return data_loader


def load_label_vocab(labels: list) -> dict:
    vocab = {}
    for i, label in enumerate(labels):
        vocab[label] = i
    return vocab


def md5(data):
    if isinstance(data, dict):
        data = json.dumps(data)
    return hashlib.md5(data.encode(encoding='utf-8')).hexdigest()
