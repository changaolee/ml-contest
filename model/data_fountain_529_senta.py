from base.model_base import ModelBase
from bunch import Bunch
from paddle import nn, unsqueeze, argmax
from paddlenlp.transformers import BertPretrainedModel
import paddle.nn.functional as F


class DataFountain529SentaBaselineModel(ModelBase):
    """
    df-529 情感分类模型 baseline: bert + fc
    """

    def __init__(self, config: Bunch):
        super().__init__(config)
        self.bert_pretrained = BertPretrainedModel.from_pretrained(self.config.pretrain_model)
        self.classifier = nn.layer.Linear(self.config.baseline.hidden_size, self.config.num_classes)
        self.loss_fct = nn.CrossEntropyLoss(soft_label=False, axis=-1)

    def forward(self, input_ids, attention_mask, label=None):
        # 将 attention_mask 进行维度变换，从 2 维变成 4 维
        # paddlenlp.transformers 的实现与 torch 或 tf 不一样，不会自动进行维度扩充
        attention_mask = unsqueeze(attention_mask, axis=[1, 2])

        # 获取 [CLS] 向量 pooled_output
        _, pooled_output = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)

        # 对 pooled_output 进行全连接，映射到 label 上
        logits = self.classifier(pooled_output)
        # 使用 softmax，获取每个标签类别的概率
        probs = F.softmax(logits, axis=1)
        # 获取标签类别概率最大的标签
        pred_label = argmax(logits, axis=-1)

        # 如果 label 不是 None，则使用 CrossEntropyLoss 计算 loss
        loss = self.loss_fct(logits, label) if label is not None else None

        return loss, pred_label, probs
