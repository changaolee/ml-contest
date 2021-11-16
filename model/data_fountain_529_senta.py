from paddlenlp.transformers import BertPretrainedModel, SkepPretrainedModel
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
        self.classifier = paddle.nn.layer.Linear(768, self.config.num_classes)
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


class DataFountain529SentaBertClsSeqMeanMaxModel(BertPretrainedModel):
    """
    Bert：融合后的 CLS 拼接 Seq Mean、Max
    """

    def __init__(self, bert, config: Bunch):
        super(DataFountain529SentaBertClsSeqMeanMaxModel, self).__init__()
        self.config = config
        self.bert = bert
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(3 * 768, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(12, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, input_ids, token_type_ids):
        encoder_outputs, _ = self.bert(input_ids,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=True)
        stacked_encoder_outputs = paddle.stack(encoder_outputs, axis=0)
        all_layer_cls_embedding = stacked_encoder_outputs[:, :, 0, :]
        weighted_cls = (self.layer_weights * all_layer_cls_embedding).sum(axis=0) / self.layer_weights.sum()

        seq_embeddings = encoder_outputs[-1][:, 1:]
        mean_seq_embedding = seq_embeddings.mean(axis=1)
        max_seq_embedding = seq_embeddings.max(axis=1)

        concat_embedding = paddle.fluid.layers.concat([weighted_cls, mean_seq_embedding, max_seq_embedding], axis=-1)
        pooled_output = self.dropout(concat_embedding)
        logits = self.classifier(pooled_output)
        return logits


class DataFountain529SentaSkepHiddenFusionModel(SkepPretrainedModel):
    """
    Skep 隐藏层向量动态融合
    """

    def __init__(self, skep, config: Bunch):
        super(DataFountain529SentaSkepHiddenFusionModel, self).__init__()
        self.config = config
        self.skep = skep
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(1024, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(24, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, input_ids, token_type_ids):
        encoder_outputs, _ = self.skep(input_ids,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=True)
        stacked_encoder_outputs = paddle.stack(encoder_outputs, axis=0)
        all_layer_cls_embedding = stacked_encoder_outputs[:, :, 0, :]
        weighted_average = (self.layer_weights * all_layer_cls_embedding).sum(axis=0) / self.layer_weights.sum()
        pooled_output = self.dropout(weighted_average)
        logits = self.classifier(pooled_output)
        return logits
