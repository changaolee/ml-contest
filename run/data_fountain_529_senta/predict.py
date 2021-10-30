from paddlenlp.transformers import BertTokenizer
from data_process.data_fountain_529_senta import DataFountain529SentaDataProcessor
from dataset.data_fountain_529_senta import DataFountain529SentaDataset
from model.data_fountain_529_senta import DataFountain529SentaBertBaselineModel
from infer.data_fountain_529_senta import DataFountain529SentaInfer
from utils.config_utils import get_config, CONFIG_PATH
from bunch import Bunch
import os


def predict():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"))

    # 原始数据预处理
    data_processor = DataFountain529SentaDataProcessor(config)
    data_processor.process()
    config = data_processor.config

    # 获取测试集
    [test_ds] = DataFountain529SentaDataset(config).load_data(splits=['test'], lazy=False)

    # 加载 model 和 tokenizer
    model, tokenizer = get_model_and_tokenizer(config.model_name, config)

    # 获取推断器
    config.model_path = os.path.join(config.ckpt_dir, config.model_name, "model_600")
    infer = DataFountain529SentaInfer(model, tokenizer=tokenizer, test_ds=test_ds, config=config)

    # 开始预测
    infer.predict()


def get_model_and_tokenizer(model_name: str, config: Bunch):
    model, tokenizer = None, None
    logger = config.logger
    if model_name == "bert_baseline":
        model = DataFountain529SentaBertBaselineModel.from_pretrained("bert-wwm-chinese", config=config)
        tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")
    else:
        logger.error("load model error.".format(model_name))
    return model, tokenizer


if __name__ == "__main__":
    predict()
