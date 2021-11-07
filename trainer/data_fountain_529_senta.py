from paddlenlp.transformers import PretrainedTokenizer, LinearDecayWithWarmup
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle import nn
from bunch import Bunch
from functools import partial
from visualdl import LogWriter
from utils.utils import create_data_loader, mkdir_if_not_exist
from utils.metric import Kappa
from utils.loss import FocalLoss
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
        self._gen_data_loader()
        self._prepare()

    def _gen_data_loader(self):
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
            trans_fn=trans_func,
            mode='train',
            batch_size=self.config.batch_size,
            batchify_fn=batchify_fn)
        self.dev_data_loader = create_data_loader(
            self.dev_ds,
            trans_fn=trans_func,
            mode='dev',
            batch_size=self.config.batch_size,
            batchify_fn=batchify_fn)

    def _prepare(self):
        # 当前训练折次
        self.fold = self.config.fold
        # 训练折数
        self.total_fold = self.config.k_fold
        # 训练轮次
        self.epochs = self.config.train_epochs

        # 训练过程中保存模型参数的文件夹
        self.ckpt_dir = os.path.join(self.config.ckpt_dir, self.config.model_name, "fold_{}".format(self.fold))
        mkdir_if_not_exist(self.ckpt_dir)

        # 可视化日志的文件夹
        self.vis_dir = os.path.join(self.config.vis_dir, self.config.model_name, "fold_{}".format(self.fold))
        mkdir_if_not_exist(self.vis_dir)

        # 训练所需要的总 step 数
        self.num_training_steps = len(self.train_data_loader) * self.epochs

        # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
        self.lr_scheduler = LinearDecayWithWarmup(self.config.learning_rate, self.num_training_steps, 0.0)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        # 定义 Optimizer
        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params)

        # 交叉熵损失函数
        self.criterion = paddle.nn.loss.CrossEntropyLoss()
        # kappa 评价指标
        self.metric = Kappa(self.config.num_classes)

        # TODO: 使用准确率评价
        self.metric = paddle.metric.Accuracy()

    def train_v2(self):
        global_step = 0
        tic_train = time.time()
        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(self.train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                probs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
                loss = self.criterion(probs, labels)
                correct = self.metric.compute(probs, labels)
                self.metric.update(correct)
                acc = self.metric.accumulate()

                global_step += 1

                # 每间隔 10 step 输出训练指标
                if global_step % 100 == 0:
                    self.logger.info(
                        "「%d/%d」global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                        % (self.fold, self.total_fold, global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()

                    with LogWriter(logdir=self.vis_dir) as writer:
                        writer.add_scalar(tag="acc_train", step=global_step, value=acc)
                        writer.add_scalar(tag="loss_train", step=global_step, value=loss)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                if global_step % 100 == 0 or global_step == self.num_training_steps:
                    save_dir = os.path.join(self.ckpt_dir, "model_%d" % global_step)
                    mkdir_if_not_exist(save_dir)
                    # 评估当前训练的模型
                    self.evaluate_v2(global_step=global_step)
                    # 保存当前模型参数等
                    paddle.save(self.model.state_dict(), os.path.join(save_dir, "model.pdparams"))
                    paddle.save(self.optimizer.state_dict(), os.path.join(save_dir, "opt.optparams"))
                    # 保存 tokenizer 的词表等
                    self.tokenizer.save_pretrained(save_dir)

    def train(self):
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
                preds = paddle.argmax(probs, axis=1, keepdim=True)

                # 计算 kappa
                self.metric.update(preds, labels)
                kappa = self.metric.accumulate()

                global_step += 1
                if global_step % 10 == 0:
                    self.logger.info(
                        "「%d/%d」global step %d, epoch: %d, batch: %d, loss: %.5f, kappa: %.5f, speed: %.2f step/s"
                        % (self.fold, self.total_fold, global_step, epoch, step, loss, kappa,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()

                    with LogWriter(logdir=self.vis_dir) as writer:
                        writer.add_scalar(tag="kappa_train", step=global_step, value=kappa)
                        writer.add_scalar(tag="loss_train", step=global_step, value=loss)

                # 反向梯度回传，更新参数
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                if global_step % 100 == 0 or global_step == self.num_training_steps:
                    save_dir = os.path.join(self.ckpt_dir, "model_%d" % global_step)
                    mkdir_if_not_exist(save_dir)
                    # 评估当前训练的模型
                    self.evaluate(global_step=global_step)
                    # 保存当前模型参数等
                    paddle.save(self.model.state_dict(), os.path.join(save_dir, "model.pdparams"))
                    paddle.save(self.optimizer.state_dict(), os.path.join(save_dir, "opt.optparams"))
                    # 保存 tokenizer 的词表等
                    self.tokenizer.save_pretrained(save_dir)

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len=512):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])

    @paddle.no_grad()
    def evaluate(self, global_step):
        self.model.eval()
        self.metric.reset()
        losses, kappa = [], 0.0
        for batch in self.dev_data_loader:
            input_ids, token_type_ids, labels = batch
            logits = self.model(input_ids, token_type_ids)
            loss = self.criterion(logits, labels)
            losses.append(loss.numpy())
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1, keepdim=True)
            self.metric.update(preds, labels)
            kappa = self.metric.accumulate()
        self.logger.info("「%d/%d」eval loss: %.5f, kappa: %.5f"
                         % (self.fold, self.total_fold, float(np.mean(losses)), kappa))
        self.model.train()
        self.metric.reset()

        with LogWriter(logdir=self.vis_dir) as writer:
            writer.add_scalar(tag="kappa_dev", step=global_step, value=kappa)
            writer.add_scalar(tag="loss_dev", step=global_step, value=float(np.mean(losses)))

    @paddle.no_grad()
    def evaluate_v2(self, global_step):
        self.model.eval()
        self.metric.reset()
        losses, acc = [], 0.0
        for batch in self.dev_data_loader:
            input_ids, token_type_ids, labels = batch
            probs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = self.criterion(probs, labels)
            losses.append(loss.numpy())
            correct = self.metric.compute(probs, labels)
            self.metric.update(correct)
            acc = self.metric.accumulate()
        self.logger.info("「%d/%d」eval loss: %.5f, kappa: %.5f"
                         % (self.fold, self.total_fold, float(np.mean(losses)), acc))
        self.model.train()
        self.metric.reset()

        with LogWriter(logdir=self.vis_dir) as writer:
            writer.add_scalar(tag="acc_dev", step=global_step, value=acc)
            writer.add_scalar(tag="loss_dev", step=global_step, value=float(np.mean(losses)))
