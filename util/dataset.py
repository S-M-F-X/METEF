import warnings

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, args, config, flag='train'):

        self.args = args
        self.output_dim = config.output_dim
        self.hist_len = config.hist_len
        self.pred_len = config.pred_len
        self.train_ratio = config.train_ratio
        self.valid_ratio = config.valid_ratio

        self.pi_fen = self.args.pi_fen
        self.time_fen = self.args.time_fen

        if self.train_ratio <= 0 or self.valid_ratio <= 0 or self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("Invalid dataset split ratio")

        train_end = 0
        valid_start = 0
        valid_end = 0
        test_start = 0

        read = pd.read_csv(args.data_path + args.dataset + '.csv')
        if self.args.print:
            print('Loaded data shape:', read.shape)

        self.has_date = 'date' in read.columns

        if self.has_date:
            time = self._get_time_feature(read[['date']], self.time_fen)

            data = read.drop(['date'], axis=1).values

        else:
            data = read.values

        if self.args.print:
            print('Total data shape:', data.shape)

        train_len = int(self.train_ratio * len(data))
        valid_len = int(self.valid_ratio * len(data))
        test_len = len(data) - train_len - valid_len
        need_len = self.hist_len + self.pred_len
        if min(train_len, valid_len, test_len) < need_len:
            raise ValueError(f"Insufficient dataset length:hist_len={self.hist_len};pred_len={self.pred_len}")

        if self.pi_fen == 1:
            train_end = int(self.train_ratio * len(data))
            valid_start = train_end
            valid_end = int((self.train_ratio + self.valid_ratio) * len(data))
            test_start = valid_end
        elif self.pi_fen == 2:
            train_end = int(self.train_ratio * len(data))
            valid_start = train_end - self.hist_len - self.pred_len + 1
            valid_end = int((self.train_ratio + self.valid_ratio) * len(data))
            test_start = valid_end - self.hist_len - self.pred_len + 1

        if self.args.print:
            print(f'train:{0}---{train_end}')
            print(f'valid:{valid_start}---{valid_end}')
            print(f'test:{test_start}---{len(data)}')

        self.scaler = StandardScaler()

        train_data = data[:train_end, :]
        self.scaler.fit(train_data)
        data = self.scaler.transform(data)

        if flag == 'train':
            self.data = data[:train_end, :]
            print('Training set data shape:', self.data.shape)
        elif flag == 'valid':
            self.data = data[valid_start:valid_end, :]
            print('Validation set data shape:', self.data.shape)
        elif flag == 'test':
            self.data = data[test_start:, :]
            print('Test set data shape:', self.data.shape)
        else:
            raise ValueError(f'Invalid flag:{flag}')

        if self.has_date:
            if flag == 'train':
                self.time = time[:train_end, :]
            elif flag == 'valid':
                self.time = time[valid_start:valid_end, :]
            elif flag == 'test':
                self.time = time[test_start:, :]
            else:
                raise ValueError(f'Invalid flag:{flag}')


    @staticmethod
    def _get_time_feature(read, time_fen):
        read['date'] = pd.to_datetime(read['date'])

        if time_fen == 1:
            read['month'] = (read['date'].dt.month - 1) / 11 - 0.5
            read['day'] = (read['date'].dt.day - 1) / 30 - 0.5
            read['weekday'] = read['date'].dt.weekday / 6 - 0.5
            read['hour'] = read['date'].dt.hour / 23 - 0.5
            read['minute'] = read['date'].dt.minute / 59 - 0.5
            read['second'] = read['date'].dt.second / 59 - 0.5
            return read[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values

        if time_fen == 2:
            read['month'] = np.sin(2 * np.pi * (read['month'] - 1) / 12)
            read['day'] = np.sin(2 * np.pi * (read['day'] - 1) / 31)
            read['weekday'] = np.sin(2 * np.pi * read['weekday'] / 7)
            read['hour'] = np.sin(2 * np.pi * read['hour'] / 24)
            read['minute'] = np.sin(2 * np.pi * read['minute'] / 60)
            read['second'] = np.sin(2 * np.pi * read['second'] / 60)
            return read[['month', 'day', 'weekday', 'hour', 'minute', 'second']].values

        if time_fen == 3:
            read['month_sin'] = np.sin(2 * np.pi * (read['month'] - 1) / 12)
            read['month_cos'] = np.cos(2 * np.pi * (read['month'] - 1) / 12)
            read['day_sin'] = np.sin(2 * np.pi * (read['day'] - 1) / 31)
            read['day_cos'] = np.cos(2 * np.pi * (read['day'] - 1) / 31)
            read['weekday_sin'] = np.sin(2 * np.pi * read['weekday'] / 7)
            read['weekday_cos'] = np.cos(2 * np.pi * read['weekday'] / 7)
            read['hour_sin'] = np.sin(2 * np.pi * read['hour'] / 24)
            read['hour_cos'] = np.cos(2 * np.pi * read['hour'] / 24)
            read['minute_sin'] = np.sin(2 * np.pi * read['minute'] / 60)
            read['minute_cos'] = np.cos(2 * np.pi * read['minute'] / 60)
            read['second_sin'] = np.sin(2 * np.pi * read['second'] / 60)
            read['second_cos'] = np.cos(2 * np.pi * read['second'] / 60)
            return read[['month_sin', 'month_cos','day_sin', 'day_cos','weekday_sin', 'weekday_cos',
                       'hour_sin', 'hour_cos','minute_sin', 'minute_cos','second_sin', 'second_cos']].values

    def __getitem__(self, index):

        if self.has_date:
            x_time = self.time[index:index + self.hist_len, :]
            y_time = self.time[index + self.hist_len:index + self.hist_len + self.pred_len, :]
        else:
            x_time = np.empty((0, 0))
            y_time = np.empty((0, 0))

        x_data = self.data[index:index + self.hist_len, :]
        y_data = self.data[index + self.hist_len:index + self.hist_len + self.pred_len, :]

        return x_data, y_data, x_time, y_time

    def inverse_transform(self, data):
        mean = self.scaler.mean_[:self.output_dim]
        std = self.scaler.scale_[:self.output_dim]
        if isinstance(data, torch.Tensor):
            params = (
                torch.tensor(mean, dtype=data.dtype, device=data.device),
                torch.tensor(std, dtype=data.dtype, device=data.device)
            )
        elif isinstance(data, np.ndarray):
            params = (mean, std)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Only NumPy arrays and PyTorch tensors are supported")

        mean_param, std_param = params

        expand_dims = data.ndim - 1

        shape = [1] * expand_dims + [-1]

        return data * std_param.reshape(*shape) + mean_param.reshape(*shape)

    def __len__(self):

        return len(self.data) - self.hist_len - self.pred_len + 1
