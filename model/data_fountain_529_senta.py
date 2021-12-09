from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification, BertPretrainedModel
from dotmap import DotMap
import paddle


def get_model_and_tokenizer(config: DotMap):
    model_name = config.model_name
    if model_name == "bert_base":
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=config.num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "bert_hidden_fusion":
        model = DataFountain529SentaBertHiddenFusionModel.from_pretrained("bert-base-chinese", config=config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    else:
        raise RuntimeError("load model error: {}.".format(model_name))
    return model, tokenizer


class DataFountain529SentaBertHiddenFusionModel(BertPretrainedModel):
    """
    Bert 隐藏层向量动态融合
    """

    def __init__(self, bert, config: DotMap):
        super(DataFountain529SentaBertHiddenFusionModel, self).__init__()
        self.config = config
        self.bert = bert
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(768, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(12, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))
        self.apply(self.init_weights)

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
    Bert：最后一层 pooled_output 与 Seq Mean、Max 动态融合
    """

    def __init__(self, bert, config: DotMap):
        super(DataFountain529SentaBertClsSeqMeanMaxModel, self).__init__()
        self.config = config
        self.bert = bert
        self.dropout = paddle.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = paddle.nn.layer.Linear(768, self.config.num_classes)
        self.layer_weights = self.create_parameter(shape=(3, 1, 1),
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids):
        encoder_outputs, pooled_output = self.bert(input_ids,
                                                   token_type_ids=token_type_ids,
                                                   output_hidden_states=True)
        seq_embeddings = encoder_outputs[-1][:, 1:]
        mean_seq_embedding = seq_embeddings.mean(axis=1)
        max_seq_embedding = seq_embeddings.max(axis=1)

        stacked_outputs = paddle.stack([pooled_output, mean_seq_embedding, max_seq_embedding], axis=0)
        weighted_average = (self.layer_weights * stacked_outputs).sum(axis=0) / self.layer_weights.sum()

        pooled_output = self.dropout(weighted_average)
        logits = self.classifier(pooled_output)

        return logits
