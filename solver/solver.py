from time import time

import numpy as np
import torch
import os
import gc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.dataset import MyDataset
from util.eval import get_model_stats, Evaluate_np, Evaluate_tensor
from util.save_result import SaveResult
from util.plotter import Plotter
from util.stoper import Stopper

from models.model import Model

model_dict = {
    'Model': Model,
}


class Solver:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = self._acquire_device()
        self.log_path = os.path.join(self.args.output_path, 'data.txt')
        self.batch = True

        self.train_loader, self.valid_loader, self.test_loader = self._get_loader()

        if self.args.print:
            print('Solver.init:Dataset loaded')

        self.channel = self.train_loader.dataset.data.shape[-1]

        self.model = model_dict[self.args.model](self.config, self.channel).to(self.device)
        if self.args.print:
            print('Solver.init:channel:', self.channel)
            print('Solver.init:Model initialized')

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.lr_decay = 0

        if self.args.print:
            print('Solver.init:Adam optimizer initialized')

        self.stopper = Stopper(log_path=self.log_path, config=self.config, optimizer=self.optimizer,
                               model_path=os.path.join(self.args.model_path, 'checkpoint.pth'))
        self.valid_lun = 0
        self.best_lun = 0
        self.best_decay = 0

        self.Evaluate_train = Evaluate_tensor(self.config.output_dim, self.args.print_train_loss, self.device)
        self.Evaluate_test = Evaluate_np(self.config.output_dim)
        self.SaveResult = SaveResult(args, self.config.pred_len, self.config.output_dim)
        self.Plotter = Plotter(self.args, self.config.output_dim)

        if self.args.print:
            print('Solver.init:Evaluation module initialized')

    @staticmethod
    def MAPE(y_true, y_pred):

        mape = torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + 1e-8)) * 100

        return mape

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f'cuda:{self.args.device}')
            print(f'Use GPU:cuda:{self.args.device}')
        else:
            device = torch.device('cpu')
            print('Use CPU')

        return device

    def _log(self, message):
        print(message, flush=True)
        with open(self.log_path, 'a', encoding="utf-8-sig") as f:
            f.write(message)

    def _get_loader(self):
        train_set = MyDataset(self.args, self.config, flag='train')
        valid_set = MyDataset(self.args, self.config, flag='valid')
        test_set = MyDataset(self.args, self.config, flag='test')
        print('Training set shape:', train_set.data.shape)
        print('Validation set shape:', valid_set.data.shape)
        print('Test set shape:', test_set.data.shape)
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader

    def _process_one_batch(self, x_data, y_data, x_time, y_time):
        x_data = x_data.float().to(self.device)
        y_data = y_data.float().to(self.device)
        if x_time.shape[1] > 0:
            x_time = x_time.float().to(self.device)
            y_time = y_time.float().to(self.device)
            if self.batch:
                print(f'x_time shape:{x_time.shape}')
                print(f'y_time shape:{y_time.shape}')
        else:
            x_time = None
            y_time = None
            if self.batch:
                print(f'x_time set to None')
                print(f'y_time set to None')
        label_len = self.config.hist_len // 2
        x_data = x_data.to(self.device)
        xY0_data = torch.concat([x_data[:, -label_len:, :], torch.zeros_like(y_data)], dim=1).to(self.device)
        if x_time is not None:
            x_time = x_time.to(self.device)
            xY_time = torch.concat([x_time[:, -label_len:, :], y_time], dim=1).to(self.device)
        else:
            xY_time = None
        if self.batch or self.args.print:
            self.batch = False
            print('Solver.one_batch:x_data shape:', x_data.shape)
            print('Solver.one_batch:xY0_data shape:', xY0_data.shape)
            if x_time is not None:
                print('Solver.one_batch:x_time shape:', x_time.shape)
                print('Solver.one_batch:xY_time shape:', xY_time.shape)
            else:
                print('Solver.one_batch:x_time:None')
                print('Solver.one_batch:xY_time:None')
        pred = self.model(x_data, xY0_data, x_time, xY_time)
        y_data = self.train_loader.dataset.inverse_transform(y_data[:, :, :self.config.output_dim])
        pred = self.train_loader.dataset.inverse_transform(pred)
        if self.args.print:
            print('Solver.one_batch:y_data shape:', y_data.shape)
            print('Solver.one_batch:pred shape:', pred.shape)

        return y_data, pred

    def _process_one_epoch(self, data_loader):
        for pi, (x_data, y_data, x_time, y_time) in enumerate(data_loader):
            if self.args.print:
                mode = "Training" if self.model.training else "Validation"
                print(f"\n--------------------------------------------Starting {mode} - Batch {pi + 1}"
                      f"--------------------------------------------")
                print(f'{mode} set x_data shape: {x_data.shape}')
                print(f'{mode} set y_data shape: {y_data.shape}')
                print(f'{mode} set x_time shape: {x_time.shape}')
                print(f'{mode} set y_time shape: {y_time.shape}')

            if self.model.training:
                self.optimizer.zero_grad()

            true, pred = self._process_one_batch(x_data, y_data, x_time, y_time)

            self.Evaluate_train.update(true, pred)

            if self.model.training:
                self.MAPE(true, pred).backward()
                self.optimizer.step()

        losses = self.Evaluate_train.result()
        self.Evaluate_train.reset()

        return losses

    def train(self):
        lun = 0
        for e in range(self.config.epoch):
            lun += 1
            print(f"\n--------------------------------------------Starting training - Epoch {lun}"
                  "--------------------------------------------")
            start = time()

            self.model.train()
            train_loss = self._process_one_epoch(self.train_loader)

            with torch.no_grad():
                self.model.eval()
                valid_loss = self._process_one_epoch(self.valid_loader)
            end = time()

            print_str = (
                f"--------------------Epoch: {e + 1} || Time Elapsed: {end - start:.6f}s-------------------- \n"
                f"Average Training MAPE: {train_loss['mean-MAPE']:.6f}")
            if self.args.print_train_loss:
                print_str += (
                    f" || Average Training MAE: {train_loss['mean-MAE']:.6f} Average Training MSE: {train_loss['mean-MSE']:.6f} "
                    f"Average Training RMSE: {train_loss['mean-RMSE_all']:.6f} || Distribution Similarity: {train_loss['mean-pearson']:.6f}")
            print_str += f"\nAverage Validation MAPE: {valid_loss['mean-MAPE']:.6f}"
            if self.args.print_train_loss:
                print_str += (
                    f" || Average Validation MAE: {valid_loss['mean-MAE']:.6f} Average Validation MSE: {valid_loss['mean-MSE']:.6f} "
                    f"Average Validation RMSE: {valid_loss['mean-RMSE_all']:.6f} || Distribution Similarity: {valid_loss['mean-pearson']:.6f}")
            print_str += f"\n"

            self._log(print_str)

            self.valid_lun, self.best_lun, self.best_decay, stop_training = self.stopper(valid_loss['mean-MAE'], self.model, lun)
            if stop_training:
                break

        return lun

    def test(self):

        start = time()
        if self.args.use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'),
                                              map_location=self.device))

        model_stats = get_model_stats(self.model)

        with torch.no_grad():
            self.model.eval()
            pi = 0
            for (x_data, y_data, x_time, y_time) in self.test_loader:
                pi = pi + 1
                if self.args.print:
                    print(f"\n--------------------------------------------Starting Testing - Batch {pi}"
                          "--------------------------------------------")

                true, pred = self._process_one_batch(x_data, y_data, x_time, y_time)

                self.Evaluate_test.update(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
                self.SaveResult.save_result_to_csvs(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
            self.SaveResult.flush()
            test_time = time() - start

            if self.args.plot:
                self.Plotter.plot()
            if self.args.plot_loss:
                self.Plotter.plot_loss()

            test_res = self.Evaluate_test.result()
            test_res["size"] = model_stats[0]
            test_res["total_params"] = model_stats[1]
            test_res["trainable_params"] = model_stats[2]
            test_res["params_by_module"] = model_stats[3]

        if self.args.only_test:
            return test_res, test_time, 0, 0, 0
        else:
            return test_res, test_time, self.valid_lun, self.best_lun, self.best_decay
