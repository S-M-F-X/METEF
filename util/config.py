import argparse


class BaseConfig:

    def __init__(self):

        self.epoch = 1
        self.batch_size = 64

        self.train_ratio = 0.8
        self.valid_ratio = 0.1
        self.hist_len = 4
        self.pred_len = 2
        self.output_dim = 3

        self.patience = 3
        self.lr = 0.0001
        self.lr_decay_weight = 0.5
        self.cooldown = 1
        self.lr_decay_max = 4


class newConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64


def get_config(model_name: str):
    if model_name == 'Model':
        print('newConfig')
        return newConfig()
    if model_name == 1:
        print('new1Config')
        return newConfig()
    if model_name == 2:
        print('new2Config')
        return newConfig()
    if model_name == 3:
        print('new3Config')
        return newConfig()

    else:
        raise ValueError(f"Model not found:{model_name}")
