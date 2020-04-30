import abc


class TrainerComponent(abc.ABC):
    """Interface for composable functionality in the Trainer."""
    def __init__(self, args):
        self.args = args

    def on_train_begin(self, steps, logs):
        pass

    def on_train_end(self, steps, logs):
        pass

    def on_batch_begin(self, steps, logs):
        pass

    def on_batch_end(self, steps, logs):
        pass

    def on_epoch_begin(self, steps, epochs, logs):
        pass

    def on_epoch_end(self, steps, epochs, logs):
        pass

    @property
    def trainer(self):
        if not hasattr(self, '_trainer'):
            raise AttributeError(f'trainer not set on `{self.__class__.__name__}`')
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    @classmethod
    def add_args_to_parser(self, parser):
        pass
