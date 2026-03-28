import numpy as np
import torch


class Stopper:
    def __init__(self, log_path, config, optimizer, model_path='./checkpoint.pth'):
        self.log_path = log_path
        self.config = config
        self.patience = self.config.patience
        self.lr_decay_weight = self.config.lr_decay_weight
        self.cooldown = self.config.cooldown
        self.lr_decay_max = self.config.lr_decay_max
        self.optimizer = optimizer
        self.model_path = model_path

        self.best_lun = 0
        self.valid_lun = 0
        self.best_decay = 0
        self.counter = 0
        self.lr_decay = 0
        self.cooldown_steps = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, lun):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.valid_lun = 1
            self.best_lun = 1
            self.save_checkpoint(val_loss, model)
            return self.valid_lun, self.best_lun, self.best_decay, False

        improved = val_loss <= self.best_loss

        if self.cooldown_steps > 0:
            if improved:
                self.valid_lun += 1
                self.best_lun = lun
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            else:
                msg = f'Cooldown period (remaining {self.cooldown_steps} epochs), no improvement in this epoch\n'
                self.log(msg)
            self.cooldown_steps -= 1
            return self.valid_lun, self.best_lun, self.best_decay, False
        else:
            if improved:
                self.valid_lun += 1
                self.best_lun = lun
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            else:
                self.counter += 1
                print_lr = f'Epochs without validation loss improvement: {self.counter}, Patience: {self.patience}\n'
                self.log(print_lr)
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.counter = 0

            if self.early_stop:
                stop_training = self.handle_early_stop()
                return self.valid_lun, self.best_lun, self.best_decay, stop_training
            else:
                return self.valid_lun, self.best_lun, self.best_decay, False

    def save_checkpoint(self, val_loss, model):
        print_lr = (f"Effective training epochs: {self.valid_lun}, validation loss decreased "
                    f"({self.val_loss_min:.8f} --> {val_loss:.8f}). Saving model ...\n")
        self.log(print_lr)
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss
        self.best_decay = self.lr_decay

    def handle_early_stop(self):
        if self.lr_decay < self.config.lr_decay_max:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.config.lr_decay_weight
            self.cooldown_steps = self.cooldown
            self.lr_decay += 1
            self.early_stop = False

            print_lr = (f"\n\n\n----------------------------------Learning rate decay count: {self.lr_decay}, Current "
                        f"learning rate: {self.optimizer.param_groups[0]['lr']}----------------------------------\n\n\n")
            self.log(print_lr)
            return False
        else:
            print_lr = (
                f"\n\n\n----------------------------------Maximum learning rate decay count "
                f"({self.config.lr_decay_max}) reached, stopping training----------------------------------\n\n\n")
            self.log(print_lr)
            return True

    def log(self, message):
        print(message, flush=True)
        with open(self.log_path, 'a', encoding="utf-8-sig") as f:
            f.write(message)


