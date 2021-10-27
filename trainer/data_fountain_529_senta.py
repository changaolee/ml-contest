from paddlenlp.transformers import BertTokenizer, SkepTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.datasets import MapDataset
from paddle.io import DataLoader, BatchSampler
from paddle.optimizer import AdamW
from paddle.metric import Accuracy
from paddle import nn
from bunch import Bunch
from functools import partial
import numpy as np
import time


class DataFountain529SentaTrainer(object):
    def __init__(self, model: nn.Layer, train_ds: MapDataset, dev_ds: MapDataset, config: Bunch):
        self.model = model
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.config = config
        self.logger = self.config.logger

    def train(self):
        # 加载预训练模型对应的 tokenizer
        pretrained_model_name = self.config.pretrained_model_name
        if pretrained_model_name in ["bert-wwm-chinese"]:
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        elif pretrained_model_name in ["skep_ernie_1.0_large_ch"]:
            tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name)
        else:
            self.logger.error("load pretrain_model {} tokenizer error.".format(pretrained_model_name))
            return False

        # 转换至模型的输入
        self.train_ds = self.train_ds.map(partial(
            self.convert_example, tokenizer=tokenizer, max_seq_len=self.config.max_seq_len
        ))
        self.dev_ds = self.dev_ds.map(partial(
            self.convert_example, tokenizer=tokenizer, max_seq_len=self.config.max_seq_len
        ))

        # 构建训练集合的 data_loader
        train_batch_sampler = BatchSampler(dataset=self.train_ds, batch_size=self.config.train_batch_size, shuffle=True)
        train_data_loader = DataLoader(dataset=self.train_ds, batch_sampler=train_batch_sampler, return_list=True)

        # 训练参数读取
        num_train_epochs = self.config.train_epochs
        num_training_steps = len(train_data_loader) * num_train_epochs

        # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
        lr_scheduler = LinearDecayWithWarmup(self.config.learning_rate, num_training_steps, 0.0)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        # 定义 Optimizer
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params)
        # 交叉熵损失
        criterion = nn.loss.CrossEntropyLoss()
        # 评估的时候采用准确率指标
        metric = Accuracy()

        # 开启训练
        global_step = 0
        tic_train = time.time()
        for epoch in range(1, num_train_epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):

                input_ids, token_type_ids, labels = batch
                probs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
                loss = criterion(probs, labels)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1

                # 每间隔 100 step 输出训练指标
                if global_step % 100 == 0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                          % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])
