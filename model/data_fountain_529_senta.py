import paddle
from bunch import Bunch
from paddlenlp.transformers import BertPretrainedModel, RobertaPretrainedModel, RobertaForSequenceClassification
from paddlenlp.transformers.roberta.modeling import RobertaModel


class DataFountain529SentaBertHiddenFusionModel(BertPretrainedModel):
    """
    Bert 隐藏层向量动态融合
    """

    def __init__(self, bert, config: Bunch):
        super().__init__()
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
        super().__init__()
        self.config = config
        self.roberta = roberta
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(self.config.hidden_size, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(12, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))

        # 重写 RobertaModel 的 forward 方法
        # 支持获取中间隐藏层向量
        RobertaModel.forward = self.roberta_forward

    def roberta_forward(self,
                        input_ids,
                        token_type_ids=None,
                        position_ids=None,
                        attention_mask=None,
                        output_hidden_states=False):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
            return encoder_outputs, pooled_output
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output

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
