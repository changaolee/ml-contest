from model.data_fountain_529_ner import get_model_and_tokenizer
from data_process.data_fountain_529_ner import DataFountain529NerDataProcessor
from dataset.data_fountain_529_ner import DataFountain529NerDataset
from infer.data_fountain_529_ner import DataFountain529NerInfer
from utils.config_utils import get_config, CONFIG_PATH
from utils.utils import mkdir_if_not_exist
from dotmap import DotMap
from scipy import stats
import csv
import os


def predict():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_ner.json"), "predict")

    # 原始数据预处理
    data_processor = DataFountain529NerDataProcessor(config)
    data_processor.process()

    # 获取全部分类标签
    config.label_list = DataFountain529NerDataset.get_labels()

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
    res_dir = os.path.join(config.res_dir, config.model_name)
    mkdir_if_not_exist(res_dir)
    with open(os.path.join(res_dir, "result.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "BIO_anno"])
        for line in result:
            qid, tags = line
            writer.writerow([qid, " ".join(tags)])


def merge_fusion_result(fusion_result):
    # TODO: 尝试不同的模型融合方法
    return fusion_result[0]


def single_model_predict(config: DotMap, model_name: str, model_path: str):
    # 获取测试集
    [test_ds] = DataFountain529NerDataset(config).load_data(splits=['test'], lazy=False)

    # 加载 model 和 tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name, config)

    # 获取推断器
    infer = DataFountain529NerInfer(model,
                                    tokenizer=tokenizer,
                                    test_ds=test_ds,
                                    config=config,
                                    model_params_path=model_path)
    # 开始预测
    result = infer.predict()

    return result


def merge_k_fold_result(k_fold_result):
    merge_result = {}
    for fold_result in k_fold_result:
        for line in fold_result:
            qid, tags = line[0], line[1]
            if qid not in merge_result:
                merge_result[qid] = []
            merge_result[qid].append(tags)

    result = []
    for qid, tags in merge_result.items():
        merge_tags = stats.mode(tags)[0][0]
        result.append([qid, merge_tags])
    return result


if __name__ == "__main__":
    predict()
