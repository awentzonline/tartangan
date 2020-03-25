import abc


class TrainerComponent(abc.ABC):
    """Interface for composable functionality in the Trainer."""

    def on_train_begin(self, step=0):
        pass

    def on_train_end(self, step):
        pass

    def on_batch_begin(self, step):
        pass

    def on_batch_end(self, step):
        pass

    def on_epoch_begin(self, step, epoch):
        pass

    def on_epoch_end(self, step, epoch):
        pass

    @property
    def trainer(self):
        if not hasattr(self, '_trainer'):
            raise AttributeError(f'trainer not set on `{self.__class__.__name__}`')
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer
