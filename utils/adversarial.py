import paddle


class FGM(object):
    """
    Fast Gradient Method（FGM）
    针对 embedding 层梯度上升干扰的对抗训练方法
    """

    def __init__(self, model, epsilon=1., emb_name='emb'):
        # emb_name 这个参数要换成你模型中 embedding 的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and self.emb_name in name:  # 检验参数是否可训练及范围
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad 是个 numpy 对象
                norm = paddle.norm(grad_tensor)  # norm 化
                if norm != 0:
                    r_at = self.epsilon * grad_tensor / norm
                    param.add(r_at)  # 在原有 embed 值上添加向上梯度干扰

    def restore(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and self.emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有 embed 参数还原
        self.backup = {}
