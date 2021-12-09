from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import get_model_and_tokenizer
from infer.data_fountain_529_senta import DataFountain529SentaInfer
from utils.config_utils import get_config, CONFIG_PATH
from utils.utils import mkdir_if_not_exist
import numpy as np
import csv
import os


def predict():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"), "predict")

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()

    k_fold_result = []
    k_fold_models = {
        1: "",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "",
        8: "",
        9: "",
        10: ""
    }
    for fold, model_path in k_fold_models.items():
        # 获取测试集
        [test_ds] = DataFountain529SentaDataset(config).load_data(splits=['test'], lazy=False)

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(config.model_name, config)

        # 获取推断器
        model_path = os.path.join(config.ckpt_dir, config.model_name, "fold_{}/{}/model.pdparams".format(fold, model_path))
        infer = DataFountain529SentaInfer(model, tokenizer=tokenizer, test_ds=test_ds, config=config, model_params_path=model_path)

        # 开始预测
        fold_result = infer.predict()
        fold_result = merge_tta_result(fold_result)
        k_fold_result.append(fold_result)

    # 融合 k 折模型的预测结果
    result = merge_k_fold_result(k_fold_result)

    # 写入预测结果
    res_dir = os.path.join(config.res_dir, config.model_name)
    mkdir_if_not_exist(res_dir)
    with open(os.path.join(res_dir, "result.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class"])
        for line in result:
            qid, label = line
            writer.writerow([qid, label])


def merge_tta_result(tta_result):
    tta = {}
    for record in tta_result:
        qid, probs = record[0], record[1]
        if qid not in tta:
            tta[qid] = {'probs': [0.] * 3, 'num': 0}
        tta[qid]['probs'] = np.sum([tta[qid]['probs'], probs], axis=0)
        tta[qid]['num'] += 1

    result = []
    for qid, sum_probs in tta.items():
        probs = np.divide(sum_probs['probs'], sum_probs['num']).tolist()
        result.append([qid, probs])
    return result


def merge_k_fold_result(k_fold_result):
    merge_result = {}
    for fold_result in k_fold_result:
        for line in fold_result:
            qid, probs = line[0], line[1]
            if qid not in merge_result:
                merge_result[qid] = [0.] * 3
            merge_result[qid] = np.sum([merge_result[qid], probs], axis=0)
    merge_result = sorted(merge_result.items(), key=lambda x: x[0], reverse=False)

    result = []
    for line in merge_result:
        qid, probs = line[0], line[1]
        label = np.argmax(probs)
        result.append([qid, label])
    return result


if __name__ == "__main__":
    predict()
