from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle import nn
from bunch import Bunch
from functools import partial
from utils.utils import create_data_loader, evaluate
import paddle.nn.functional as F
import numpy as np
import time
import paddle
import os


class DataFountain529SentaTrainer(object):
    train_data_loader = None
    dev_data_loader = None
    epochs = None
    ckpt_dir = None
    num_training_steps = None
    optimizer = None
    criterion = None
    metric = None

    def __init__(self,
                 model: nn.Layer,
                 tokenizer: PretrainedTokenizer,
                 train_ds: MapDataset,
                 dev_ds: MapDataset,
                 config: Bunch):
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.config = config
        self.logger = self.config.logger

    def gen_data_loader(self):
        # 将数据处理成模型可读入的数据格式
        trans_func = partial(
            self.convert_example,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len)

        # 将数据组成批量式数据，如
        # 将不同长度的文本序列 padding 到批量式数据中最大长度
        # 将每条数据 label 堆叠在一起
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Stack()  # labels
        ): [data for data in fn(samples)]

        self.train_data_loader = create_data_loader(
            self.train_ds,
            mode='train',
            batch_size=self.config.train_batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)
        self.dev_data_loader = create_data_loader(
            self.dev_ds,
            mode='dev',
            batch_size=self.config.train_batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

    def prepare(self):
        # 训练轮次
        self.epochs = self.config.train_epochs
        # 训练过程中保存模型参数的文件夹
        self.ckpt_dir = self.config.ckpt_dir
        # 训练所需要的总 step 数
        self.num_training_steps = len(self.train_data_loader) * self.epochs
        # Adam 优化器
        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.config.learning_rate,
            parameters=self.model.parameters())
        # 交叉熵损失函数
        self.criterion = paddle.nn.loss.CrossEntropyLoss()
        # accuracy评价指标
        self.metric = paddle.metric.Accuracy()

    def train(self):
        self.gen_data_loader()
        self.prepare()

        # 开启训练
        global_step = 0
        tic_train = time.time()
        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(self.train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                # 喂数据给 model
                logits = self.model(input_ids, token_type_ids)
                # 计算损失函数值
                loss = self.criterion(logits, labels)
                # 预测分类概率值
                probs = F.softmax(logits, axis=1)
                # 计算 acc
                correct = self.metric.compute(probs, labels)
                self.metric.update(correct)
                acc = self.metric.accumulate()

                global_step += 1
                if global_step % 10 == 0:
                    self.logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()

                # 反向梯度回传，更新参数
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                if global_step % 100 == 0:
                    save_dir = os.path.join(self.ckpt_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # 评估当前训练的模型
                    evaluate(self.model, self.criterion, self.metric, self.dev_data_loader)
                    # 保存当前模型参数等
                    self.model.save_pretrained(save_dir)
                    # 保存 tokenizer 的词表等
                    self.tokenizer.save_pretrained(save_dir)

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len=512):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])
