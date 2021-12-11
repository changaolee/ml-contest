import json

from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import get_model_and_tokenizer
from infer.data_fountain_529_senta import DataFountain529SentaInfer
from sklearn.model_selection import StratifiedKFold
from dotmap import DotMap
from utils.utils import mkdir_if_not_exist, md5
from utils.config_utils import DATA_PATH
from utils.nlp_da import NlpDA
import pandas as pd
import string
import random
import csv
import os
import re


class DataFountain529SentaDataProcessor(object):

    def __init__(self, config: DotMap):
        self.config = config

        # 原始数据集路径
        self.train_data_path = os.path.join(DATA_PATH, config.exp_name, config.train_filename)
        self.test_data_path = os.path.join(DATA_PATH, config.exp_name, config.test_filename)

        # 根据配置文件设置数据处理后的存储路径
        self._set_processed_data_path(config)

        self.da_options = config.data_process.data_augmentation
        self.enable_text_clean = config.data_process.enable_text_clean
        self.sp_options = config.data_process.data_split_options

        self.enable_da = bool(self.da_options)
        self.enable_sp = bool(self.sp_options)

        # 数据集划分配置
        self.k_fold = config.k_fold
        self.random_state = config.random_state
        self.dev_prop = config.dev_prop

        self.logger = config.logger

    def _set_processed_data_path(self, config):
        data_process_config = config.data_process
        unique_dir_name = md5({**data_process_config.toDict(), **{"k_fold": config.k_fold,
                                                                  "random_state": config.random_state,
                                                                  "dev_prop": config.dev_prop}})
        # 数据难度评估结果存储路径
        self.data_difficulty_assessment_path = os.path.join(
            DATA_PATH, config.exp_name, "data_difficulty_assessment", unique_dir_name
        )
        mkdir_if_not_exist(self.data_difficulty_assessment_path)
        self.assessed_path = os.path.join(self.data_difficulty_assessment_path, "assessed_data.csv")
        self.data_difficulty_score_path = os.path.join(
            self.data_difficulty_assessment_path, "data_difficulty_score.json"
        )

        # 数据增强路径
        self.data_augmentation_path = os.path.join(DATA_PATH, config.exp_name, "data_augmentation", unique_dir_name)
        mkdir_if_not_exist(self.data_augmentation_path)
        self.da_train_data_path = os.path.join(self.data_augmentation_path, "train.csv")
        self.da_test_data_path = os.path.join(self.data_augmentation_path, "test.csv")

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
            "assessed": self.assessed_path,
        }

    def process(self):
        # 数据难度评估
        if self.enable_sp:
            self._data_difficulty_assessment()

        # 数据增强
        if self.enable_da:
            self._data_augmentation()

        # 训练集、开发集划分
        self._train_dev_dataset_split()

        # 测试集保存
        self._test_dataset_save()

    def _data_difficulty_assessment(self):
        # 相应文件都存在，跳过处理
        if os.path.isfile(self.assessed_path) and os.path.isfile(self.data_difficulty_score_path):
            self.logger.info("skip data difficulty assessment")
            return

        # 待评估数据集保存
        self._assessed_dataset_save()

        # 获取待评估数据集
        [assessed_ds] = DataFountain529SentaDataset(self.config).load_data(splits=['assessed'], lazy=False)

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(self.sp_options.model_name, self.config)

        # 获取推断器
        model_params_path = self.sp_options.model_params_path
        infer = DataFountain529SentaInfer(
            model, tokenizer=tokenizer, test_ds=assessed_ds, config=self.config, model_params_path=model_params_path)

        # 开始预测
        result = infer.predict()

        # 提取数据难度打分（预测概率，值越小难度越大）
        difficulty_score = {}
        df = pd.read_csv(self.assessed_path, encoding="utf-8")
        for qid, probs in result:
            label = df.loc[df["id"] == qid].iloc[0]["label"]
            difficulty_score[qid] = probs[int(label)]

        # 保存数据难度打分
        with open(self.data_difficulty_score_path, "w", encoding="utf-8") as score_f:
            json.dump(difficulty_score, score_f)

    def _assessed_dataset_save(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")

        with open(self.assessed_path, "w", encoding="utf-8") as assessed_f:
            assessed_writer = csv.writer(assessed_f)
            assessed_writer.writerow(["id", "text", "label"])

            rows = []
            for idx, line in df.iterrows():
                _id, text, label = line.get("id", ""), line.get("text", ""), line.get("class", "")
                rows.append([_id, text, label])
            assessed_writer.writerows(rows)

    def _data_augmentation(self):
        # 相应文件都存在，跳过处理
        if os.path.isfile(self.da_train_data_path) and os.path.isfile(self.da_test_data_path):
            self.logger.info("skip data augmentation")
            return

        # 数据增强对象
        nlp_da = NlpDA(**self.da_options)

        # 训练数据
        train_df = pd.read_csv(self.train_data_path, encoding="utf-8")
        with open(self.da_train_data_path, "w", encoding="utf-8") as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(["id", "text", "class"])  # 保持与原始数据一致

            for idx, line in train_df.iterrows():
                _id, text, label = line.get("id", ""), line.get("text", ""), line.get("class", "")
                for da_text in nlp_da.generate(text):
                    train_writer.writerow([_id, da_text, label])

        # 测试数据
        test_df = pd.read_csv(self.test_data_path, encoding="utf-8")
        with open(self.da_test_data_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            for idx, line in test_df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                for da_text in nlp_da.generate(text):
                    test_writer.writerow([_id, da_text])

    def _train_dev_dataset_split(self):
        df = pd.read_csv(self.train_data_path, encoding="utf-8")
        X, y = df.drop(["class"], axis=1), df["class"]

        if self.enable_sp:
            def _get_difficulty_class(_class, _score):
                return int("{}{}".format(_class, int(_score * 10) // 3))

            with open(self.data_difficulty_score_path, "r", encoding="utf-8") as difficulty_score_f:
                difficulty_score = json.load(difficulty_score_f)
            df["difficulty_class"] = df.apply(lambda x: _get_difficulty_class(
                x["class"], difficulty_score[str(x["id"])]
            ), axis=1)
            X, y = df.drop(["difficulty_class"], axis=1), df["difficulty_class"]

        da_df = pd.read_csv(self.da_train_data_path, encoding="utf-8") if self.enable_da else None

        k_fold = int(1 / self.dev_prop) if self.k_fold == 0 else self.k_fold
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state).split(X, y)
        for k, (train_idx, dev_idx) in enumerate(skf):
            train_path = self.train_path.format(k)
            with open(train_path, "w", encoding="utf-8") as train_f:
                train_writer = csv.writer(train_f)
                train_writer.writerow(["text", "label"])

                rows = []
                for i in range(len(train_idx)):
                    cur_X, cur_y = X.iloc[train_idx[i]], y.iloc[train_idx[i]]
                    if self.enable_sp:
                        cur_y = cur_X["class"]
                    if da_df:
                        da_texts = da_df.loc[da_df["id"] == cur_X["id"]]["text"].values.tolist()
                        for da_text in da_texts:
                            if self.enable_text_clean:
                                da_text = self.text_clean(da_text)
                            rows.append([da_text, cur_y])
                    else:
                        text = cur_X["text"]
                        if self.enable_text_clean:
                            text = self.text_clean(text)
                        rows.append([text, cur_y])
                random.shuffle(rows)
                train_writer.writerows(rows)

            dev_path = self.dev_path.format(k)
            with open(dev_path, "w", encoding="utf-8") as dev_f:
                dev_writer = csv.writer(dev_f)
                dev_writer.writerow(["text", "label"])

                for i in range(len(dev_idx)):
                    cur_X, cur_y = X.iloc[dev_idx[i]], y.iloc[dev_idx[i]]
                    if self.enable_sp:
                        cur_y = cur_X["class"]
                    text = cur_X["text"]
                    if self.enable_text_clean:
                        text = self.text_clean(text)
                    dev_writer.writerow([text, cur_y])

            if self.k_fold == 0:
                break

    def _test_dataset_save(self):
        df = pd.read_csv(self.test_data_path, encoding="utf-8")
        da_df = pd.read_csv(self.da_test_data_path, encoding="utf-8") \
            if self.enable_da and self.da_options.enable_tta else None

        with open(self.test_path, "w", encoding="utf-8") as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow(["id", "text"])

            rows = []
            for idx, line in df.iterrows():
                _id, text = line.get("id", ""), line.get("text", "")
                if da_df:
                    da_texts = da_df.loc[da_df["id"] == _id]["text"].values.tolist()
                    for da_text in da_texts:
                        if self.enable_text_clean:
                            da_text = self.text_clean(da_text)
                        rows.append([_id, da_text])
                else:
                    if self.enable_text_clean:
                        text = self.text_clean(text)
                    rows.append([_id, text])
            test_writer.writerows(rows)

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
