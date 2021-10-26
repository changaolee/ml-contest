from bunch import Bunch
from utils.utils import mkdir_if_not_exist
import pandas as pd
import csv
import random
import os


class DataFountain529SentaDataProcessor(object):
    DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../data"))

    def __init__(self, config: Bunch):
        self.config = config

        # 原始数据集路径
        self.train_data_path = os.path.join(self.DATA_PATH, self.config.exp_name, self.config.train_filename)
        self.test_data_path = os.path.join(self.DATA_PATH, self.config.exp_name, self.config.test_filename)

        # 处理后的数据集路径
        processed_path = os.path.join(self.DATA_PATH, self.config.exp_name, "processed")
        mkdir_if_not_exist(processed_path)
        self.config.train_path = os.path.join(processed_path, "train.csv")
        self.config.dev_path = os.path.join(processed_path, "dev.csv")
        self.config.test_path = os.path.join(processed_path, "test.csv")

        # 开发集占比
        self.dev_prop = config.dev_prop

    def process(self, override=False):
        assert 0 < self.dev_prop < 1, "开发集占比应在 0 到 1 之间"

        if not override and \
                os.path.isfile(self.config.train_path) and \
                os.path.isfile(self.config.dev_path) and \
                os.path.isfile(self.config.test_path):
            return

        # 训练集、开发集划分
        self.train_dev_dataset_split()

        # TODO: 数据增强

        # 测试集保存
        self.test_dataset_save()

    def train_dev_dataset_split(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")

        with open(self.config.train_path, "w", encoding="utf-8") as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(["text", "label"])

            with open(self.config.dev_path, "w", encoding="utf-8") as dev_f:
                dev_writer = csv.writer(dev_f)
                dev_writer.writerow(["text", "label"])

                for idx, line in df.iterrows():
                    text, label = line.get("text", ""), line.get("class", "")
                    if random.random() <= self.dev_prop:
                        dev_writer.writerow([text, label])
                    else:
                        train_writer.writerow([text, label])

    def test_dataset_save(self):
        df = pd.read_csv(self.test_data_path, encoding="utf-8")

        with open(self.config.test_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                test_writer.writerow([_id, text])
