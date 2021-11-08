import paddle
import paddle.nn.functional as F


class FocalLoss(paddle.nn.Layer):
    def __init__(self, num_classes, alpha=0.5, gamma=2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        probs = F.softmax(logits, axis=1)
        labels = F.one_hot(labels.flatten(), self.num_classes)

        log_pt = labels * paddle.fluid.layers.log(probs)
        log_pt = paddle.fluid.layers.reduce_sum(log_pt, dim=-1)

        weight = -1.0 * labels * paddle.fluid.layers.pow((1.0 - probs), self.gamma)
        weight = paddle.fluid.layers.reduce_sum(weight, dim=-1)

        alpha = self.alpha * labels
        alpha = paddle.fluid.layers.reduce_sum(alpha, dim=-1)

        loss = alpha * weight * log_pt

        return loss.mean()
