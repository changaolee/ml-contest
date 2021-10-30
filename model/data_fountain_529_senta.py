from bunch import Bunch
from paddle import nn
from paddlenlp.transformers import BertPretrainedModel


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

    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
