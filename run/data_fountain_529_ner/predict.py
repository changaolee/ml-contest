from paddlenlp.transformers import BertTokenizer, BertForTokenClassification
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from model.data_fountain_529_ner import BertCrfForTokenClassification, ErnieCrfForTokenClassification
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
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_ner.json"))

    # 原始数据预处理
    data_processor = DataFountain529NerDataProcessor(config)
    data_processor.process()
    config = data_processor.config

    # 获取全部分类标签
    config.label_list = DataFountain529NerDataset.get_labels()

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
        [test_ds] = DataFountain529NerDataset(config).load_data(splits=['test'], lazy=False)

        # 加载 model 和 tokenizer
        model, tokenizer = get_model_and_tokenizer(config.model_name, config)

        # 获取推断器
        config.model_path = os.path.join(config.ckpt_dir, config.model_name, "fold_{}/{}".format(fold, model_path))
        infer = DataFountain529NerInfer(model, tokenizer=tokenizer, test_ds=test_ds, config=config)

        # 开始预测
        fold_result = infer.predict()
        k_fold_result.append(fold_result)

    # 融合 k 折模型的预测结果
    result = merge_k_fold_result(k_fold_result)

    # 写入预测结果
    res_dir = os.path.join(config.res_dir, config.model_name)
    mkdir_if_not_exist(res_dir)
    with open(os.path.join(res_dir, "result.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "BIO_anno"])
        for line in result:
            qid, tags = line
            writer.writerow([qid, " ".join(tags)])


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


def get_model_and_tokenizer(model_name: str, config: DotMap):
    model, tokenizer = None, None
    logger = config.logger
    if model_name == "bert_base":
        model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "bert_crf":
        bert = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        model = BertCrfForTokenClassification(bert)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "ernie_base":
        model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(config.label_list))
        tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    elif model_name == "ernie_crf":
        ernie = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(config.label_list))
        model = ErnieCrfForTokenClassification(ernie)
        tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    else:
        logger.error("load model error: {}.".format(model_name))
    return model, tokenizer


if __name__ == "__main__":
    predict()
