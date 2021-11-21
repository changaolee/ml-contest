from paddlenlp.transformers import PretrainedTokenizer, LinearDecayWithWarmup
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import ChunkEvaluator
from paddle import nn
from bunch import Bunch
from functools import partial
from visualdl import LogWriter
from utils.utils import create_data_loader, mkdir_if_not_exist, load_label_vocab
from utils.loss import FocalLoss
import numpy as np
import time
import paddle
import os


class DataFountain529NerTrainer(object):
    train_data_loader = None
    dev_data_loader = None
    epochs = None
    ckpt_dir = None
    num_training_steps = None
    optimizer = None
    criterion = None
    metric = None
    ignore_label = -100

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
        label_vocab = load_label_vocab(self.config.label_list)
        self.no_entity_id = label_vocab["O"]

        # 将数据处理成模型可读入的数据格式
        trans_func = partial(
            self.convert_example,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            label_vocab=label_vocab)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int32'),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype='int32'),  # token_type_ids
            Stack(dtype='int64'),  # seq_len
            Pad(axis=0, pad_val=self.no_entity_id, dtype='int64')  # labels
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
        self.train_vis_dir = os.path.join(
            self.config.vis_dir, self.config.model_name, "fold_{}/train".format(self.fold)
        )
        self.dev_vis_dir = os.path.join(
            self.config.vis_dir, self.config.model_name, "fold_{}/dev".format(self.fold)
        )
        mkdir_if_not_exist(self.train_vis_dir)
        mkdir_if_not_exist(self.dev_vis_dir)

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
            epsilon=self.config.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params)

        # 交叉熵损失函数
        self.criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=self.no_entity_id)
        self.eval_criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=self.no_entity_id)

        # # Focal Loss
        # self.criterion = FocalLoss(num_classes=len(self.config.label_list), ignore_index=self.no_entity_id)
        # self.eval_criterion = FocalLoss(num_classes=len(self.config.label_list), ignore_index=self.no_entity_id)

        # 评价指标
        self.metric = ChunkEvaluator(label_list=self.config.label_list)
        self.eval_metric = ChunkEvaluator(label_list=self.config.label_list)

    def train(self):
        # 开启训练
        global_step = 0
        tic_train = time.time()

        with LogWriter(logdir=self.train_vis_dir) as train_writer:
            with LogWriter(logdir=self.dev_vis_dir) as dev_writer:
                for epoch in range(1, self.epochs + 1):
                    for step, batch in enumerate(self.train_data_loader):
                        input_ids, token_type_ids, lens, labels = batch

                        # 喂数据给 model
                        logits = self.model(input_ids, token_type_ids)

                        # 计算损失函数值
                        loss = paddle.mean(self.criterion(logits, labels))

                        # 预测分类概率值
                        preds = logits.argmax(axis=2)

                        n_infer, n_label, n_correct = self.metric.compute(lens, preds, labels)
                        self.metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
                        precision, recall, f1_score = self.metric.accumulate()

                        global_step += 1
                        if global_step % 10 == 0:
                            self.logger.info(
                                "「%d/%d」global step %d, epoch: %d, batch: %d, loss: %.5f, precision: %.5f, recall: %.5f, f1: %.5f, speed: %.2f step/s"
                                % (self.fold, self.total_fold, global_step, epoch, step,
                                   loss, precision, recall, f1_score, 10 / (time.time() - tic_train)))
                            tic_train = time.time()

                            train_writer.add_scalar(tag="precision", step=global_step, value=precision)
                            train_writer.add_scalar(tag="recall", step=global_step, value=recall)
                            train_writer.add_scalar(tag="f1", step=global_step, value=f1_score)
                            train_writer.add_scalar(tag="loss", step=global_step, value=loss)

                        # 反向梯度回传，更新参数
                        loss.backward()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.clear_grad()

                        if global_step % 10 == 0 or global_step == self.num_training_steps:
                            # 评估当前训练的模型
                            loss_dev, precision_dev, recall_dev, f1_score_dev = self.evaluate()

                            dev_writer.add_scalar(tag="precision", step=global_step, value=precision_dev)
                            dev_writer.add_scalar(tag="recall", step=global_step, value=recall_dev)
                            dev_writer.add_scalar(tag="f1", step=global_step, value=f1_score_dev)
                            dev_writer.add_scalar(tag="loss", step=global_step, value=loss_dev)

                            # 保存当前模型参数等
                            if global_step >= 100:
                                save_dir = os.path.join(self.ckpt_dir, "model_%d" % global_step)
                                mkdir_if_not_exist(save_dir)
                                paddle.save(self.model.state_dict(), os.path.join(save_dir, "model.pdparams"))

    @staticmethod
    def convert_example(example, tokenizer, max_seq_len, label_vocab):
        tokens, labels = example["tokens"], example["labels"]
        encoded_inputs = tokenizer(text=tokens,
                                   max_seq_len=max_seq_len,
                                   return_length=True,
                                   is_split_into_words=True)

        # -2 for [CLS] and [SEP]
        if len(encoded_inputs['input_ids']) - 2 < len(labels):
            labels = labels[:len(encoded_inputs['input_ids']) - 2]
        labels = ["O"] + labels + ["O"]
        labels += ["O"] * (len(encoded_inputs['input_ids']) - len(labels))

        encoded_inputs["labels"] = [label_vocab[label] for label in labels]

        return tuple([np.array(x, dtype="int64") for x in [encoded_inputs["input_ids"],
                                                           encoded_inputs["token_type_ids"],
                                                           encoded_inputs["seq_len"],
                                                           encoded_inputs["labels"]]])

    @paddle.no_grad()
    def evaluate(self):
        self.model.eval()
        self.eval_metric.reset()

        avg_loss, precision, recall, f1_score = 0., 0., 0., 0.
        for batch in self.dev_data_loader:
            input_ids, token_type_ids, lens, labels = batch
            logits = self.model(input_ids, token_type_ids)
            preds = logits.argmax(axis=2)

            loss = self.eval_criterion(logits, labels)
            avg_loss = paddle.mean(loss)

            n_infer, n_label, n_correct = self.eval_metric.compute(lens, preds, labels)
            self.eval_metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
            precision, recall, f1_score = self.eval_metric.accumulate()

        self.logger.info("「%d/%d」eval loss: %.5f, precision: %.5f, recall: %.5f, f1: %.5f"
                         % (self.fold, self.total_fold, avg_loss, precision, recall, f1_score))
        self.model.train()
        self.eval_metric.reset()

        return avg_loss, precision, recall, f1_score
