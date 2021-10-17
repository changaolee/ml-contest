from bunch import Bunch
from paddle import nn, unsqueeze
from paddlenlp.transformers import BertPretrainedModel, SkepPretrainedModel


class DataFountain529SentaBertBaselineModel(BertPretrainedModel):
    """
    df-529 情感分类模型 baseline: bert + fc
    """

    def __init__(self, bert, config: Bunch):
        super().__init__()
        self.config = config
        self.bert = bert
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.layer.Linear(self.config.hidden_size, self.config.num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        # 将 attention_mask 进行维度变换，从 2 维变成 4 维
        # paddlenlp.transformers 的实现与 torch 或 tf 不一样，不会自动进行维度扩充
        attention_mask = unsqueeze(attention_mask, axis=[1, 2])

        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DataFountain529SentaSkepBaselineModel(SkepPretrainedModel):
    """
    df-529 情感分类模型 baseline: skep + fc
    """

    def __init__(self, skep, config: Bunch):
        super().__init__()
        self.config = config
        self.skep = skep
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.layer.Linear(self.config.hidden_size, self.config.num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        # 将 attention_mask 进行维度变换，从 2 维变成 4 维
        # paddlenlp.transformers 的实现与 torch 或 tf 不一样，不会自动进行维度扩充
        attention_mask = unsqueeze(attention_mask, axis=[1, 2])

        _, pooled_output = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
