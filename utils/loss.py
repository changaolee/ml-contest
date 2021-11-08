import paddle
import paddle.nn.functional as F


class FocalLoss(paddle.nn.Layer):
    def __init__(self, num_classes, alpha=0.5, gamma=2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        pred = F.softmax(logits, axis=1)
        one_hot = F.one_hot(labels.flatten(), self.num_classes)
        cross_entropy = one_hot * paddle.fluid.layers.log(pred)
        cross_entropy = paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)
        weight = -1.0 * one_hot * paddle.fluid.layers.pow((1.0 - pred), self.gamma)
        weight = paddle.fluid.layers.reduce_sum(weight, dim=-1)
        ax = self.alpha * one_hot
        alpha = paddle.fluid.layers.reduce_sum(ax, dim=-1)
        loss = alpha * weight * cross_entropy
        return loss.sum()