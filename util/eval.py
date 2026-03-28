import numpy as np
from einops import rearrange

import torch


def get_model_stats(model):

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size = (param_size + buffer_size) / 1024 / 1024
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_by_module = {}
    for name, param in model.named_parameters():
        top_level_module = name.split('.')[0]
        if top_level_module in params_by_module:
            params_by_module[top_level_module] += param.numel()
        else:
            params_by_module[top_level_module] = param.numel()
    return [model_size, total_params, trainable_params, params_by_module]


def evaluate(pred, true):

    pred = rearrange(pred, 'B q N -> (B q) N').astype(np.float64)
    true = rearrange(true, 'B q N -> (B q) N').astype(np.float64)

    T, N = pred.shape
    abs_error = np.abs(pred - true)
    squared_error = (pred - true) ** 2

    MAPE = np.mean(abs_error / (np.abs(true) + 1e-8), axis=0) * 100
    MAE = np.mean(abs_error, axis=0)
    MSE = np.mean(squared_error, axis=0)
    RMSE = np.sqrt(MSE)

    sigma_p = pred.std(axis=0)
    sigma_t = true.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_t = true.mean(axis=0)
    numerator = ((pred - mean_p) * (true - mean_t)).mean(axis=0)
    denominator = sigma_p * sigma_t

    index = denominator > 1e-8
    pearson = np.zeros(N)
    pearson[index] = numerator[index] / denominator[index]

    ss_res = np.sum(squared_error, axis=0)
    ss_tot = np.sum((true - mean_t) ** 2, axis=0)
    R2 = np.full(N, np.nan, dtype=np.float64)
    tot_mask = ss_tot > 1e-8
    R2[tot_mask] = 1.0 - (ss_res[tot_mask] / ss_tot[tot_mask])
    R2_adj = np.full(N, np.nan, dtype=np.float64)
    denominator_R2_adj = T - N - 1.0
    if denominator_R2_adj > 1e-8:
        factor = (T - 1.0) / denominator_R2_adj
        R2_adj[tot_mask] = 1.0 - (1.0 - R2[tot_mask]) * factor

    test_losses = {
        'MAPE': MAPE,
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'pearson': pearson,
        'R2_adj': R2_adj,
        'mean-MAPE': np.mean(MAPE),
        'mean-MAE': np.mean(MAE),
        'mean-MSE': np.mean(MSE),
        'mean-RMSE_feature': np.mean(RMSE),
        'mean-RMSE_all': np.sqrt(np.mean(MSE)),
        'mean-pearson': np.mean(pearson),
        'mean-R2_adj': np.nanmean(R2_adj),
    }

    return test_losses


class Evaluate_np:
    def __init__(self, N):
        self.N = N
        self.reset()

    def reset(self):

        self.T = 0

        self.sum_abs_error = np.zeros(self.N, dtype=np.float64)
        self.sum_squared_error = np.zeros(self.N, dtype=np.float64)
        self.sum_abs_percent_error = np.zeros(self.N, dtype=np.float64)

        self.sum_x = np.zeros(self.N, dtype=np.float64)
        self.sum_y = np.zeros(self.N, dtype=np.float64)
        self.sum_xx = np.zeros(self.N, dtype=np.float64)
        self.sum_yy = np.zeros(self.N, dtype=np.float64)
        self.sum_xy = np.zeros(self.N, dtype=np.float64)

    def update(self, true, pred):
        true = rearrange(true, 'B q N -> (B q) N')
        pred = rearrange(pred, 'B q N -> (B q) N')

        self.T += pred.shape[0]
        abs_error = np.abs(pred - true)

        self.sum_abs_error += np.sum(abs_error, axis=0)
        self.sum_squared_error += np.sum(abs_error ** 2, axis=0)
        self.sum_abs_percent_error += np.sum(abs_error / (np.abs(true) + 1e-8), axis=0)

        self.sum_x += np.sum(true, axis=0)
        self.sum_y += np.sum(pred, axis=0)
        self.sum_xx += np.sum(true ** 2, axis=0)
        self.sum_yy += np.sum(pred ** 2, axis=0)
        self.sum_xy += np.sum(true * pred, axis=0)

    def result(self):
        if self.T == 0:
            return {
                'MAPE': 0.0, 'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'pearson': 0.0
            }

        numerator = self.sum_xy - (self.sum_x * self.sum_y / self.T)
        var_x = self.sum_xx - (self.sum_x ** 2 / self.T)
        var_y = self.sum_yy - (self.sum_y ** 2 / self.T)
        denominator = np.sqrt(var_x * var_y)

        index = denominator > 1e-8
        pearson = np.zeros(self.N)
        pearson[index] = numerator[index] / denominator[index]

        R2 = np.full(self.N, np.nan, dtype=np.float64)
        var_x_mask = var_x > 1e-8
        R2[var_x_mask] = 1.0 - (self.sum_squared_error[var_x_mask] / var_x[var_x_mask])
        R2_adj = np.full(self.N, np.nan, dtype=np.float64)
        denominator_R2_adj = self.T - self.N - 1.0
        if denominator_R2_adj > 1e-8:
            factor = (self.T - 1.0) / denominator_R2_adj
            R2_adj[var_x_mask] = 1.0 - (1.0 - R2[var_x_mask]) * factor

        test_losses = {
            'MAPE': (self.sum_abs_percent_error / self.T) * 100,
            'MAE': self.sum_abs_error / self.T,
            'MSE': self.sum_squared_error / self.T,
            'RMSE': np.sqrt(self.sum_squared_error / self.T),
            'pearson': pearson,
            'R2_adj': R2_adj,
            'mean-MAPE': np.mean((self.sum_abs_percent_error / self.T) * 100),
            'mean-MAE': np.mean(self.sum_abs_error / self.T),
            'mean-MSE': np.mean(self.sum_squared_error / self.T),
            'mean-RMSE_feature': np.mean(np.sqrt(self.sum_squared_error / self.T)),
            'mean-RMSE_all': np.sqrt(np.mean(self.sum_squared_error / self.T)),
            'mean-pearson': np.mean(pearson),
            'mean-R2_adj': np.nanmean(R2_adj),
        }
        return test_losses


class Evaluate_tensor:
    def __init__(self, N, print_train_loss, device):
        self.N = N
        self.print_train_loss = print_train_loss
        self.device = device
        self.reset()

    def reset(self):
        self.T = 0

        self.sum_abs_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_squared_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_abs_percent_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_x = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_y = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_xx = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_yy = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_xy = torch.zeros(self.N, dtype=torch.float64, device=self.device)

    def update(self, true, pred):
        assert true.device == self.device and pred.device == self.device, \
            f"Input tensor device does not match evaluator device (Input: {true.device}, Evaluator: {self.device})"

        true = true.reshape(-1, self.N)
        pred = pred.reshape(-1, self.N)

        with torch.no_grad():
            abs_error = torch.abs(pred - true)

            self.sum_abs_percent_error += torch.sum(abs_error / (torch.abs(true) + 1e-8), dim=0)
            self.T += true.shape[0]

            if self.print_train_loss:
                self.sum_abs_error += torch.sum(abs_error, dim=0)
                self.sum_squared_error += torch.sum(abs_error ** 2, dim=0)

                self.sum_x += torch.sum(true, dim=0)
                self.sum_y += torch.sum(pred, dim=0)
                self.sum_xx += torch.sum(true ** 2, dim=0)
                self.sum_yy += torch.sum(pred ** 2, dim=0)
                self.sum_xy += torch.sum(true * pred, dim=0)

    def result(self):

        with torch.no_grad():
            MAPE = (self.sum_abs_percent_error / self.T) * 100
            mean_MAPE = torch.mean(MAPE)

            if self.print_train_loss:
                MAE = self.sum_abs_error / self.T
                MSE = self.sum_squared_error / self.T
                RMSE = torch.sqrt(MSE)

                numerator = self.sum_xy - (self.sum_x * self.sum_y) / self.T
                var_x = self.sum_xx - (self.sum_x ** 2) / self.T
                var_y = self.sum_yy - (self.sum_y ** 2) / self.T
                denominator = torch.sqrt(var_x * var_y)

                pearson = torch.zeros(self.N, dtype=torch.float64, device=self.device)
                index = denominator > 1e-8
                pearson[index] = numerator[index] / denominator[index]

                R2 = torch.full((self.N,), float('nan'), dtype=torch.float64, device=self.device)
                var_x_mask = var_x > 1e-8
                R2[var_x_mask] = 1.0 - (self.sum_squared_error[var_x_mask] / var_x[var_x_mask])
                R2_adj = torch.full((self.N,), float('nan'), dtype=torch.float64, device=self.device)
                denominator_R2_adj = self.T - self.N - 1.0
                if denominator_R2_adj > 1e-8:
                    factor = (self.T - 1.0) / denominator_R2_adj
                    R2_adj[var_x_mask] = 1.0 - (1.0 - R2[var_x_mask]) * factor

                mean_MAE = torch.mean(MAE)
                mean_MSE = torch.mean(MSE)
                mean_RMSE_feature = torch.mean(RMSE)
                mean_RMSE_all = torch.sqrt(mean_MSE)
                mean_pearson = torch.mean(pearson)
                valid_mask = ~torch.isnan(R2_adj)
                if valid_mask.sum() > 0:
                    mean_R2_adj = torch.mean(R2_adj[valid_mask])
                else:
                    mean_R2_adj = torch.tensor(0.0, device=self.device)

                return {
                    'MAPE': MAPE,
                    'MAE': MAE,
                    'MSE': MSE,
                    'RMSE': RMSE,
                    'pearson': pearson,
                    'R2_adj': R2_adj,
                    'mean-MAPE': mean_MAPE,
                    'mean-MAE': mean_MAE,
                    'mean-MSE': mean_MSE,
                    'mean-RMSE_feature': mean_RMSE_feature,
                    'mean-RMSE_all': mean_RMSE_all,
                    'mean-pearson': mean_pearson,
                    'mean-R2_adj': mean_R2_adj,
                }
            else:
                return {
                    'MAPE': MAPE,
                    'mean-MAPE': mean_MAPE
                }
