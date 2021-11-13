from sklearn.model_selection import StratifiedKFold
from bunch import Bunch
from utils.utils import mkdir_if_not_exist
from utils.config_utils import DATA_PATH, RESOURCE_PATH
from utils.nlp_da import NlpDA
import pandas as pd
import string
import random
import csv
import os
import re


class DataFountain529SentaDataProcessor(object):
    def __init__(self, config: Bunch):
        self.config = config

        # 原始数据集路径
        self.train_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.train_filename)
        self.test_data_path = os.path.join(DATA_PATH, self.config.exp_name, self.config.test_filename)

        # 获取类别分类占比
        self.config.label_dist = self.get_label_dist()

        # 数据增强路径
        self.data_augmentation_path = os.path.join(DATA_PATH, self.config.exp_name, "data_augmentation")
        mkdir_if_not_exist(self.data_augmentation_path)
        self.da_train_data_path = os.path.join(self.data_augmentation_path, "train.csv")
        self.da_test_data_path = os.path.join(self.data_augmentation_path, "test.csv")

        # 处理后的数据集路径
        self.processed_path = os.path.join(DATA_PATH, self.config.exp_name, "processed")
        mkdir_if_not_exist(self.processed_path)
        self.train_path = os.path.join(self.processed_path, "train_{}.csv")
        self.dev_path = os.path.join(self.processed_path, "dev_{}.csv")
        self.test_path = os.path.join(self.processed_path, "test.csv")

        self.config.splits = {
            "train": self.train_path,
            "dev": self.dev_path,
            "test": self.test_path
        }

        self.logger = self.config.logger

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
        # 非强制覆盖，且文件夹非空，跳过处理
        if not override and os.listdir(self.processed_path):
            self.logger.info("skip data process")
            return

        # 训练集、开发集划分
        self.train_dev_dataset_split()

        # 测试集保存
        self.test_dataset_save()

    def data_augmentation(self):
        # 文件夹非空，跳过数据增强
        if os.path.isfile(self.da_train_data_path) and os.path.isfile(self.da_test_data_path):
            self.logger.info("skip data augmentation")
            return

        # 银行实体信息
        bank_entity_path = os.path.join(RESOURCE_PATH, "entity/bank.txt")

        # 翻译缓存文件
        da_cache_path = os.path.join(self.data_augmentation_path, "cache")
        mkdir_if_not_exist(da_cache_path)
        trans_cache_path = os.path.join(da_cache_path, "trans.json")

        # 翻译路径：zh -> en -> zh
        trans_path = [("zh", "en"), ("en", "zh")]

        # 数据增强对象
        nlp_da = NlpDA(
            # random_word_options={"base_file": bank_entity_path, "create_num": 2, "change_rate": 0.3},
            # similar_word_options={"create_num": 2, "change_rate": 0.3},
            # random_delete_char_options={"create_num": 2, "change_rate": 0.1},
            # translate_options={"domain": "finance", "trans_path": trans_path, "trans_cache_file": trans_cache_path}
        )

        # 训练数据
        train_df = pd.read_csv(self.train_data_path, encoding="utf-8")
        with open(self.da_train_data_path, "w", encoding="utf-8") as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(["id", "text", "class"])  # 保持与原始数据一致

            for idx, line in train_df.iterrows():
                _id, text, label = line.get("id", ""), line.get("text", ""), line.get("class", "")
                if str(label) != "2":
                    for da_text in nlp_da.generate(text):
                        train_writer.writerow([_id, da_text, label])
                else:
                    train_writer.writerow([_id, text, label])

        # 测试数据
        test_df = pd.read_csv(self.test_data_path, encoding="utf-8")
        with open(self.da_test_data_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in test_df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                for da_text in nlp_da.generate(text):
                    test_writer.writerow([_id, da_text])

    def train_dev_dataset_split(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")
        da_df = pd.read_csv(self.da_train_data_path, encoding="utf-8")

        X, y = df.drop(['class'], axis=1), df['class']

        k_fold = int(1 / self.config.dev_prop) if self.k_fold == 0 else self.k_fold
        skf = StratifiedKFold(n_splits=k_fold,
                              shuffle=True,
                              random_state=self.random_state).split(X, y)

        for k, (train_idx, dev_idx) in enumerate(skf):

            train_path = self.train_path.format(k + 1)
            with open(train_path, "w", encoding="utf-8") as train_f:
                train_writer = csv.writer(train_f)
                train_writer.writerow(["text", "label"])

                rows = []
                for i in range(len(train_idx)):
                    cur_X, cur_y = X.iloc[train_idx[i]], y.iloc[train_idx[i]]
                    da_texts = da_df.loc[da_df["id"] == cur_X["id"]]["text"].values.tolist()
                    for da_text in da_texts:
                        da_text = self.text_clean(da_text)
                        rows.append([da_text, cur_y])
                random.shuffle(rows)
                train_writer.writerows(rows)

            dev_path = self.dev_path.format(k + 1)
            with open(dev_path, "w", encoding="utf-8") as dev_f:
                dev_writer = csv.writer(dev_f)
                dev_writer.writerow(["text", "label"])

                for i in range(len(dev_idx)):
                    cur_X, cur_y = X.iloc[dev_idx[i]], y.iloc[dev_idx[i]]
                    text = self.text_clean(cur_X["text"])
                    dev_writer.writerow([text, cur_y])

            if self.k_fold == 0:
                break

    def test_dataset_save(self):
        df = pd.read_csv(self.test_data_path, encoding="utf-8")

        with open(self.test_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                text = self.text_clean(text)
                test_writer.writerow([_id, text])

    @classmethod
    def text_clean(cls, text):
        # 去除空白字符及首尾标点
        text = text.strip().strip(string.punctuation).replace(" ", "")

        # 去除数字内部的逗号
        p = re.compile(r"\d+,\d+?")
        for m in p.finditer(text):
            mm = m.group()
            text = text.replace(mm, mm.replace(",", ""))

        # 去除数字小数点
        p = re.compile(r"\d+.\d+")
        for m in p.finditer(text):
            mm = m.group()
            text = text.replace(mm, mm[0:mm.find(".")])

        # 数字转换 k、K -> 千，w、W -> 万
        p = re.compile(r"\d+[kKＫ千]?")
        for m in p.finditer(text):
            mm = m.group()
            text = text.replace(mm, mm.replace("k", "000").replace("K", "000").replace("千", "000"))
        p = re.compile(r"\d+[wWＷ万]?")
        for m in p.finditer(text):
            mm = m.group()
            text = text.replace(mm, mm.replace("w", "0000").replace("W", "0000").replace("万", "0000"))

        # 英文标点转中文标点
        e_pun = u',.!?%[]()<>"\''
        c_pun = u'，。！？％【】（）《》“‘'
        table = {ord(f): ord(t) for f, t in zip(e_pun, c_pun)}
        text = text.translate(table)

        return text


if __name__ == "__main__":
    tests = [
        '''   "  如果  你亏了钱，独家利息,2 千、2 万，20 万，10w，3k, 14.27W通常会有50%的折扣，而我只借360利息，期限为100,000个月。"'''
    ]
    for test_text in tests:
        print(DataFountain529SentaDataProcessor.text_clean(test_text))
