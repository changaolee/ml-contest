from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle import nn
from bunch import Bunch
from functools import partial
from utils.utils import create_data_loader, load_label_vocab
import paddle.nn.functional as F
import numpy as np
import paddle
import os


class DataFountain529NerInfer(object):

    def __init__(self,
                 model: nn.Layer,
                 tokenizer: PretrainedTokenizer,
                 test_ds: MapDataset,
                 config: Bunch):
        self.model = model
        self.tokenizer = tokenizer
        self.test_ds = test_ds
        self.config = config
        self.logger = self.config.logger
        self._gen_data_loader()
        self._load_model()

    def _gen_data_loader(self):
        label_vocab = load_label_vocab(self.config.label_list)
        self.no_entity_id = label_vocab["O"]

        # 将数据处理成模型可读入的数据格式
        trans_func = partial(
            self.convert_example,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int32'),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype='int32'),  # token_type_ids
            Stack(dtype='int64'),  # seq_len
            Stack(dtype='int64'),  # qid
        ): [data for data in fn(samples)]

        self.test_data_loader = create_data_loader(
            self.test_ds,
            mode='test',
            batch_size=self.config.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

    def _load_model(self):
        model_params_path = self.config.model_path if os.path.isfile(self.config.model_path) \
            else os.path.join(self.config.model_path, "model.pdparams")
        if os.path.isfile(model_params_path):
            # 加载模型参数
            state_dict = paddle.load(model_params_path)
            self.model.set_dict(state_dict)
            self.logger.info("Loaded parameters from {}".format(model_params_path))
        else:
            self.logger.error("Loaded parameters error from {}".format(model_params_path))

    @paddle.no_grad()
    def predict(self):
        # 切换 model 模型为评估模式，关闭 dropout 等随机因素
        self.model.eval()

        result = []
        id2label = dict(enumerate(self.config.label_list))
        for step, batch in enumerate(self.test_data_loader):
            input_ids, token_type_ids, lens, qids = batch
            logits = self.model(input_ids, token_type_ids)
            pred = paddle.argmax(logits, axis=-1)

            for i, end in enumerate(lens):
                tags = [id2label[x.numpy()[0]] for x in pred[i][1:end - 1]]
                qid = qids[i].numpy()[0]
                result.append([qid, tags])
        return result

    @staticmethod
    def parse_decodes(input_words, id2label, decodes, lens):
        decodes = [x for batch in decodes for x in batch]
        lens = [x for batch in lens for x in batch]

        outputs = []
        for idx, end in enumerate(lens):
            sent = "".join(input_words[idx]['tokens'])
            tags = [id2label[x] for x in decodes[idx][1:end - 1]]
            outputs.append([sent, tags])

        return outputs

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len):
        tokens, qid = example["tokens"], example["qid"]
        encoded_inputs = tokenizer(text=tokens,
                                   max_seq_len=max_seq_len,
                                   return_length=True,
                                   is_split_into_words=True)

        return tuple([np.array(x, dtype="int64") for x in [encoded_inputs["input_ids"],
                                                           encoded_inputs["token_type_ids"],
                                                           encoded_inputs["seq_len"],
                                                           qid]])
