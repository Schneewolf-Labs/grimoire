class TrainerCallback:
    """Base class for trainer callbacks. Override any method you need."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_step_end(self, trainer, step, loss, metrics):
        pass

    def on_log(self, trainer, metrics):
        pass

    def on_evaluate(self, trainer, metrics):
        pass

    def on_save(self, trainer, path):
        pass
