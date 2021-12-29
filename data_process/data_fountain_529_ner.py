from sklearn.model_selection import KFold
from dotmap import DotMap
from utils.utils import mkdir_if_not_exist, md5
from utils.config_utils import DATA_PATH
import pandas as pd
import csv
import os


class DataFountain529NerDataProcessor(object):
    def __init__(self, config: DotMap):
        self.config = config

        # 原始数据集路径
        self.train_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.train_filename)
        self.test_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.test_filename)

        # 根据配置文件设置数据处理后的存储路径
        self._set_processed_data_path(config)

        # 数据集划分配置
        self.k_fold = config.k_fold
        self.random_state = config.random_state
        self.dev_prop = config.dev_prop

        self.mode = config.mode
        self.logger = config.logger

    def _set_processed_data_path(self, config):
        data_process_config = config.data_process
        unique_dir_name = md5({**data_process_config.toDict(), **{"k_fold": config.k_fold,
                                                                  "random_state": config.random_state,
                                                                  "dev_prop": config.dev_prop}})
        # 处理后的数据集路径
        self.processed_path = os.path.join(DATA_PATH, config.exp_name, "processed", unique_dir_name)
        mkdir_if_not_exist(self.processed_path)
        self.train_path = os.path.join(self.processed_path, "train_{}.csv")
        self.dev_path = os.path.join(self.processed_path, "dev_{}.csv")
        self.test_path = os.path.join(self.processed_path, "test.csv")

        # 补充配置信息
        config.splits = {
            "train": self.train_path,
            "dev": self.dev_path,
            "test": self.test_path,
        }

    def process(self):
        # 训练集、开发集划分
        if self.mode != "predict":
            self._train_dev_dataset_split()

        # 测试集保存
        self._test_dataset_save()

    def _train_dev_dataset_split(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")

        X, y = df.drop(['BIO_anno'], axis=1), df['BIO_anno']

        k_fold = int(1 / self.dev_prop) if self.k_fold == 0 else self.k_fold
        kf = KFold(n_splits=k_fold,
                   shuffle=True,
                   random_state=self.random_state).split(X, y)

        for k, (train_idx, dev_idx) in enumerate(kf):

            train_path = self.train_path.format(k + 1)
            with open(train_path, "w", encoding="utf-8") as train_f:
                train_writer = csv.writer(train_f)
                train_writer.writerow(["text", "BIO_anno"])

                for i in range(len(train_idx)):
                    cur_X, cur_y = X.iloc[train_idx[i]], y.iloc[train_idx[i]]
                    train_writer.writerow([cur_X["text"], cur_y])

            dev_path = self.dev_path.format(k + 1)
            with open(dev_path, "w", encoding="utf-8") as dev_f:
                dev_writer = csv.writer(dev_f)
                dev_writer.writerow(["text", "BIO_anno"])

                for i in range(len(dev_idx)):
                    cur_X, cur_y = X.iloc[dev_idx[i]], y.iloc[dev_idx[i]]
                    dev_writer.writerow([cur_X["text"], cur_y])

            if self.k_fold == 0:
                break

    def _test_dataset_save(self):
        df = pd.read_csv(self.test_data_path, encoding="utf-8")

        with open(self.test_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                test_writer.writerow([_id, text])
