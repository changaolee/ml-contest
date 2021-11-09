from sklearn.model_selection import StratifiedKFold
from bunch import Bunch
from utils.utils import mkdir_if_not_exist
from utils.config_utils import DATA_PATH
import pandas as pd
import csv
import os


class DataFountain529SentaDataProcessor(object):
    def __init__(self, config: Bunch):
        self.config = config

        # 原始数据集路径
        self.train_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.train_filename)
        self.test_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.test_filename)

        # 获取类别分类占比
        self.config.label_dist = self.get_label_dist()

        # 处理后的数据集路径
        processed_path = os.path.join(DATA_PATH, self.config.exp_name, "processed")
        mkdir_if_not_exist(processed_path)
        self.train_path = os.path.join(processed_path, "train_{}.csv")
        self.dev_path = os.path.join(processed_path, "dev_{}.csv")
        self.test_path = os.path.join(processed_path, "test.csv")

        self.config.splits = {
            "train": self.train_path,
            "dev": self.dev_path,
            "test": self.test_path
        }

        # 数据集划分配置
        self.k_fold = config.k_fold
        self.random_state = config.random_state

    def get_label_dist(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")
        label_prop = df["class"].value_counts() / len(df)
        dist = [0.] * len(label_prop)
        for label, prop in label_prop.iteritems():
            dist[label] = prop
        return dist

    def process(self, override=False):
        if not override and \
                os.path.isfile(self.train_path) and \
                os.path.isfile(self.dev_path) and \
                os.path.isfile(self.test_path):
            return

        # 训练集、开发集划分
        self.train_dev_dataset_split()

        # TODO: 数据增强

        # 测试集保存
        self.test_dataset_save()

    def train_dev_dataset_split(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")
        X, y = df.drop(['id', 'class'], axis=1), df['class']

        k_fold = int(1 / self.config.dev_prop) if self.k_fold == 0 else self.k_fold
        skf = StratifiedKFold(n_splits=k_fold,
                              shuffle=True,
                              random_state=self.random_state).split(X, y)

        for k, (train_idx, dev_idx) in enumerate(skf):

            train_path = self.train_path.format(k + 1)
            with open(train_path, "w", encoding="utf-8") as train_f:
                train_writer = csv.writer(train_f)
                train_writer.writerow(["text", "label"])

                for i in range(len(train_idx)):
                    cur_X, cur_y = X.iloc[train_idx[i]], y.iloc[train_idx[i]]
                    train_writer.writerow([cur_X["text"], cur_y])

            dev_path = self.dev_path.format(k + 1)
            with open(dev_path, "w", encoding="utf-8") as dev_f:
                dev_writer = csv.writer(dev_f)
                dev_writer.writerow(["text", "label"])

                for i in range(len(dev_idx)):
                    cur_X, cur_y = X.iloc[dev_idx[i]], y.iloc[dev_idx[i]]
                    dev_writer.writerow([cur_X["text"], cur_y])

            if self.k_fold == 0:
                break

    def test_dataset_save(self):
        df = pd.read_csv(self.test_data_path, encoding="utf-8")

        with open(self.test_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                test_writer.writerow([_id, text])
