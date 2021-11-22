from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
import paddle


class BertCrfForTokenClassification(paddle.nn.layer):
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
