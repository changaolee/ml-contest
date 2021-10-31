from paddle.metric import Metric
import paddle
import numpy as np


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


class Kappa(Metric):

    def __init__(self, num_classes, name="kappa", *args, **kwargs):
        super(Kappa, self).__init__(*args, **kwargs)
        self.n = 0  # 样本总数
        self.pred_each_n = [0] * num_classes  # 每种分类预测正确的个数
        self.label_each_n = [0] * num_classes  # 每种分类的真实数量
        self.num_classes = num_classes  # 类别数量
        self._name = name

    def update(self, preds, labels):
        """
        Update the states based on the current mini-batch prediction results.

        Args:
            preds(numpy.array): prediction results of current mini-batch,
                the output of two-class sigmoid function.
                Shape: [batch_size, 1]. Dtype: 'float64' or 'float32'.
            labels(numpy.array): ground truth (labels) of current mini-batch,
                the shape should keep the same as preds.
                Shape: [batch_size, 1], Dtype: 'int32' or 'int64'.
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")

        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

        sample_num = labels.shape[0]
        preds = np.rint(preds).astype("int32")

        for i in range(sample_num):
            pred = preds[i][0]
            label = labels[i][0]
            if pred == label:
                self.pred_each_n[label] += 1
            self.label_each_n[label] += 1
            self.n += 1

    def accumulate(self):
        """
        Calculate the final kappa.

        Returns:
            A scaler float: results of the calculated kappa.
        """
        po = float(np.sum(self.pred_each_n)) / self.n if self.n != 0 else .0
        pe = float(np.sum([self.pred_each_n[i] * self.label_each_n[i]
                           for i in range(self.num_classes)])) / (self.n * self.n) if self.n != 0 else .0

        return (po - pe) / (1 - pe)

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.n = 0
        self.pred_each_n = [0] * self.num_classes
        self.label_each_n = [0] * self.num_classes

    def name(self):
        """
        Returns metric name
        """
        return self._name
