import numpy as np
import os
import csv


class SaveResult:
    def __init__(self, args, pred_len, N):
        self.args = args
        self.N = N
        self.q = pred_len
        self.header = ['true', 'pred']
        self.sample_count = 0
        self.cache_count = 0
        self.max_cache = self.args.max_cache
        self.caches = {
            'all': [None for _ in range(self.N)],
            'every_batch': [None for _ in range(self.N)],
            'every_pred_len': [None for _ in range(self.N)]
        }
        if self.args.save_result_all:
            for i in range(self.N):
                self.create_csv(f'feature_{i+1}_all.csv')
        if self.args.save_result_every_batch:
            for i in range(self.N):
                self.create_csv(f'feature_{i+1}_every_batch.csv')
        if self.args.save_result_every_pred_len:
            for i in range(self.N):
                self.create_csv(f'feature_{i+1}_every_pred_len.csv')

    def create_csv(self, filename):
        with open(os.path.join(self.args.result_path, filename), 'w', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)

    def _save_to_csv(self, filename, data):
        with open(os.path.join(self.args.result_path, filename), 'a', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(data)

    def _update_cache(self, mode, n, new_data):
        if self.caches[mode][n] is None:
            self.caches[mode][n] = new_data
        else:
            self.caches[mode][n] = np.concatenate([self.caches[mode][n], new_data], axis=0)

        if self.caches[mode][n].shape[0] >= self.max_cache:
            self._save_to_csv(f'feature_{n+1}_{mode}.csv', self.caches[mode][n])
            self.caches[mode][n] = None

    def save_result_to_csvs(self, true, pred):
        if self.args.save_result_all:
            true_all = true.reshape(-1, self.N)
            pred_all = pred.reshape(-1, self.N)
            for n in range(self.N):
                res = np.concatenate((true_all[:, n:n+1], pred_all[:, n:n+1]), axis=1)
                self._update_cache('all', n, res)

        if self.args.save_result_every_batch:
            for n in range(self.N):
                res = np.concatenate((true[:, 0, n:n+1], pred[:, 0, n:n+1]), axis=1)
                self._update_cache('every_batch', n, res)

        if self.args.save_result_every_pred_len:
            batch_size = true.shape[0]

            batch_start = self.sample_count
            batch_end = batch_start + batch_size

            self.sample_count += batch_size

            k_start = max(0, (batch_start + self.q - 1) // self.q)
            k_end = batch_end // self.q

            for k in range(k_start, k_end + 1):
                sample_idx = k * self.q
                if batch_start <= sample_idx < batch_end:
                    batch_idx = sample_idx - batch_start
                    for n in range(self.N):
                        res = np.stack([true[batch_idx, :, n], pred[batch_idx, :, n]], axis=1)
                        self._update_cache('every_pred_len', n, res)
                    self.cache_count += 1

    def flush(self):
        if self.args.save_result_all:
            for n in range(self.N):
                cache = self.caches['all'][n]
                if cache is not None and cache.shape[0] > 0:
                    self._save_to_csv(f'feature_{n + 1}_all.csv', cache)
                    self.caches['all'][n] = None
        if self.args.save_result_every_batch:
            for n in range(self.N):
                cache = self.caches['every_batch'][n]
                if cache is not None and cache.shape[0] > 0:
                    self._save_to_csv(f'feature_{n + 1}_every_batch.csv', cache)
                    self.caches['every_batch'][n] = None
        if self.args.save_result_every_pred_len:
            for n in range(self.N):
                cache = self.caches['every_pred_len'][n]
                if cache is not None and cache.shape[0] > 0:
                    self._save_to_csv(f'feature_{n + 1}_every_pred_len.csv', cache)
                    self.caches['every_pred_len'][n] = None

    def save_result_without_chche(self, true, pred):
        true_all = true.reshape(-1, self.N)
        pred_all = pred.reshape(-1, self.N)
        for n in range(self.N):
            res = np.concatenate((true_all[:, n:n + 1], pred_all[:, n:n + 1]), axis=1)
            self._save_to_csv(f'feature_{n + 1}_all.csv', res)
