from paddlenlp.transformers import BertTokenizer, SkepTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.datasets import MapDataset
from paddle.io import DataLoader, BatchSampler
from paddle.optimizer import AdamW
from paddle.metric import Accuracy
from paddle import nn
from bunch import Bunch
from functools import partial
import paddle.nn.functional as F
import numpy as np
import time


class DataFountain529SentaTrainer(object):
    def __init__(self, model: nn.Layer, train_data: MapDataset, dev_data: MapDataset, config: Bunch):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
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
        self.train_data = self.train_data.map(partial(
            self._convert_sample, tokenizer=tokenizer, max_len=self.config.max_len
        ))
        self.dev_data = self.dev_data.map(partial(
            self._convert_sample, tokenizer=tokenizer, max_len=self.config.max_len
        ))

        train_sampler = BatchSampler(dataset=self.train_data, batch_size=self.config.train_batch_size, shuffle=True)
        train_data_loader = DataLoader(dataset=self.train_data, batch_sampler=train_sampler)

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
                input_ids, attention_mask, labels = batch
                # 喂数据给 model
                logits = self.model(input_ids, attention_mask=attention_mask)
                # 计算损失函数值
                loss = criterion(logits, labels)
                # 预测分类概率值
                probs = F.softmax(logits, axis=1)
                # 计算acc
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1
                if global_step % 10 == 0:
                    self.logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()

                # 反向梯度回传，更新参数
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                # if global_step % 100 == 0:
                #     save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     # 评估当前训练的模型
                #     evaluate(model, criterion, metric, dev_data_loader)
                #     # 保存当前模型参数等
                #     model.save_pretrained(save_dir)
                #     # 保存tokenizer的词表等
                #     tokenizer.save_pretrained(save_dir)

        # # 模型训练
        # for input_ids, attention_mask, labels in train_data_loader():
        #     logits = self.model(input_ids, attention_mask=attention_mask)
        #     loss = criterion(logits, labels)
        #     probs = F.softmax(logits, axis=1)
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.clear_grad()

    @staticmethod
    def _convert_sample(sample, tokenizer, max_len):
        encoded_inputs = tokenizer(text=sample["text"], max_seq_len=max_len)
        input_ids, attention_mask = encoded_inputs["input_ids"], [1] * len(encoded_inputs["input_ids"])
        input_ids += [0] * max(max_len - len(input_ids), 0)
        attention_mask += [0] * max(max_len - len(attention_mask), 0)
        return tuple([np.array(x, dtype="int64") for x in [input_ids, attention_mask, [sample["label"]]]])
