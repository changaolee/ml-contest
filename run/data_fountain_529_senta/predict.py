from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import get_model_and_tokenizer
from infer.data_fountain_529_senta import DataFountain529SentaInfer
from utils.config_utils import get_config, CONFIG_PATH
from dotmap import DotMap
import numpy as np
import csv
import os


def predict():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"), "predict")

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()

    # 使用配置中的所有模型进行融合
    fusion_result = []
    for model_name, weight in config.model_params.items():

        # 计算单模型 K 折交叉验证的结果
        k_fold_result = []
        for fold in range(config.k_fold or 1):
            model_path = os.path.join(config.base_path, model_name, 'model_{}.pdparams'.format(fold))
            fold_result = single_model_predict(config, model_name, model_path)
            k_fold_result.append(fold_result)

        # 融合 k 折模型的预测结果
        merge_result = merge_k_fold_result(k_fold_result)

        # 将当前模型及对应权重保存
        fusion_result.append([merge_result, weight])

    # 融合所有模型的预测结果
    result = merge_fusion_result(fusion_result)

    # 写入预测结果
    with open(os.path.join(config.base_path, "result.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class"])
        for line in result:
            qid, label = line
            writer.writerow([qid, label])


def merge_fusion_result(fusion_result):
    merge_result = {}
    for single_result, weight in fusion_result:
        for line in single_result:
            qid, label = line[0], line[1]
            if qid not in merge_result:
                merge_result[qid] = [0.] * 3
            merge_result[qid][label] += 1. * weight
    merge_result = sorted(merge_result.items(), key=lambda x: x[0], reverse=False)

    result = []
    for line in merge_result:
        qid, scores = line[0], line[1]
        label = np.argmax(scores)
        result.append([qid, label])
    return result


def single_model_predict(config: DotMap, model_name: str, model_path: str):
    # 获取测试集
    [test_ds] = DataFountain529SentaDataset(config).load_data(splits=['test'], lazy=False)

    # 加载 model 和 tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name, config)

    # 获取推断器
    infer = DataFountain529SentaInfer(model,
                                      tokenizer=tokenizer,
                                      test_ds=test_ds,
                                      config=config,
                                      model_params_path=model_path)

    # 开始预测
    result = infer.predict()

    # 合并 TTA 后的结果
    result = merge_tta_result(result)

    return result


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
