from paddlenlp.transformers import BertPretrainedModel, RobertaPretrainedModel
from bunch import Bunch
import paddle


class DataFountain529SentaBertHiddenFusionModel(BertPretrainedModel):
    """
    Bert 隐藏层向量动态融合
    """

    def __init__(self, bert, config: Bunch):
        super(DataFountain529SentaBertHiddenFusionModel, self).__init__()
        self.config = config
        self.bert = bert
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(self.config.hidden_size, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(12, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, input_ids, token_type_ids):
        encoder_outputs, _ = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=True)
        stacked_encoder_outputs = paddle.stack(encoder_outputs, axis=0)
        all_layer_cls_embedding = stacked_encoder_outputs[:, :, 0, :]
        weighted_average = (self.layer_weights * all_layer_cls_embedding).sum(axis=0) / self.layer_weights.sum()
        pooled_output = self.dropout(weighted_average)
        logits = self.classifier(pooled_output)
        return logits


class DataFountain529SentaRobertaHiddenFusionModel(RobertaPretrainedModel):
    """
    Roberta 隐藏层向量动态融合
    """

    def __init__(self, roberta, config: Bunch):
        super(DataFountain529SentaRobertaHiddenFusionModel, self).__init__()
        self.config = config
        self.roberta = roberta
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(self.config.hidden_size, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(12, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, input_ids, token_type_ids):
        encoder_outputs, _ = self.roberta(input_ids,
                                          token_type_ids=token_type_ids,
                                          output_hidden_states=True)
        stacked_encoder_outputs = paddle.stack(encoder_outputs, axis=0)
        all_layer_cls_embedding = stacked_encoder_outputs[:, :, 0, :]
        weighted_average = (self.layer_weights * all_layer_cls_embedding).sum(axis=0) / self.layer_weights.sum()
        pooled_output = self.dropout(weighted_average)
        logits = self.classifier(pooled_output)
        return logits
