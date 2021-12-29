from paddlenlp.transformers import BertTokenizer, BertForTokenClassification
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from dotmap import DotMap
import paddle


def get_model_and_tokenizer(model_name: str, config: DotMap):
    config = DotMap({**config.toDict(), **config.model_config[model_name].toDict()})
    if model_name == "bert_base":
        model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "bert_crf":
        bert = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(config.label_list))
        model = BertCrfForTokenClassification(bert)
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    elif model_name == "ernie_base":
        model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(config.label_list))
        tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    elif model_name == "ernie_crf":
        ernie = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(config.label_list))
        model = ErnieCrfForTokenClassification(ernie)
        tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    else:
        raise RuntimeError("load model error: {}.".format(model_name))
    return model, tokenizer, config


class BertCrfForTokenClassification(paddle.nn.Layer):
    def __init__(self, bert, crf_lr=100):
        super().__init__()
        self.num_classes = bert.num_classes
        self.bert = bert
        self.crf = LinearChainCrf(self.num_classes, crf_lr=crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                labels=None):
        logits = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)

        if labels is not None:
            loss = self.crf_loss(logits, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(logits, lengths)
            return prediction


class ErnieCrfForTokenClassification(paddle.nn.Layer):
    def __init__(self, ernie, crf_lr=100):
        super().__init__()
        self.num_classes = ernie.num_classes
        self.ernie = ernie
        self.crf = LinearChainCrf(
            self.num_classes, crf_lr=crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                labels=None):
        logits = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)

        if labels is not None:
            loss = self.crf_loss(logits, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(logits, lengths)
            return prediction
