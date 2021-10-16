from base.trainer_base import TrainerBase
from paddle.io import Dataset, DataLoader, BatchSampler
from paddle.optimizer import AdamW
from paddle import nn
from bunch import Bunch


class DataFountain529SentaTrainer(TrainerBase):
    def __init__(self, model: nn.Layer, train_data: Dataset, dev_data: Dataset, config: Bunch):
        super().__init__(model, train_data, dev_data, config)

    def train(self):
        train_sampler = BatchSampler(dataset=self.train_data, batch_size=self.config.train_batch_size, shuffle=True)
        train_data_loader = DataLoader(dataset=self.train_data, batch_sampler=train_sampler)

        optimizer = AdamW(learning_rate=self.config.learning_rate, parameters=self.model.parameters())
        criterion = nn.loss.CrossEntropyLoss()
