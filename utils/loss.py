import paddle
import numpy as np


class FocalLoss(paddle.nn.Layer):
    def __init__(self, alpha=0.5, gamma=2, num_classes=3, weight=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = paddle.to_tensor(np.array(weight)) \
            if weight else paddle.to_tensor(np.array([1.] * num_classes))
        self.ce_fn = paddle.nn.CrossEntropyLoss(
            weight=self.weight, soft_label=False, ignore_index=ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = paddle.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
