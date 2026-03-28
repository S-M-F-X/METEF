import numpy as np
import re
import os
import gc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from tqdm import tqdm


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = 'gray'


class Plotter:
    def __init__(self, args, N):

        self.args = args
        self.N = N

    def _read_csv_data(self, mode, n, rows):
        csv_path = os.path.join(
            self.args.result_path,
            f'feature_{n + 1}_{mode}.csv'
        )

        data = pd.read_csv(csv_path, nrows=rows)
        true = data['true'].values
        pred = data['pred'].values

        return true, pred

    def single_plot(self, mode, n):
        try:
            true, pred = self._read_csv_data(mode, n, rows=168)

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_title(f"feature_{n + 1}_{mode}_time")
            ax.set_xlabel("Time (hour)", fontsize=14, labelpad=10)
            ax.set_ylabel("Amplitude", fontsize=14, labelpad=10)

            ax.plot(true, color='#1f77b4', linewidth=1.2, label='true')
            ax.plot(pred, color='#ff7f0e', linewidth=1.2, label='pred')
            ax.fill_between(np.arange(len(true)), true, color='#1f77b4', alpha=0.1)
            ax.fill_between(np.arange(len(pred)), pred, color='#ff7f0e', alpha=0.1)
            # ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5, zorder=0)
            ax.grid(True, linestyle='--', alpha=0.6)

            ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none',
                      facecolor='white', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            save_path = os.path.join(self.args.figures_path, f'feature_{n+1}_{mode}_time.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            true_freq = np.abs(np.fft.rfft(true)) / len(true)
            pred_freq = np.abs(np.fft.rfft(pred)) / len(pred)

            sampling_rate = 100
            x_freq = np.fft.rfftfreq(len(true), d=1 / sampling_rate)

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_title(f"feature_{n + 1}_{mode}_freq")
            ax.set_xlabel("Frequency (Hz)", fontsize=14, labelpad=10)
            ax.set_ylabel("Amplitude", fontsize=14, labelpad=10)

            ax.plot(x_freq, true_freq, color='#1f77b4', linewidth=2.0, label='true')
            ax.plot(x_freq, pred_freq, color='#ff7f0e', linewidth=2.0, label='pred')
            ax.fill_between(x_freq, true_freq, color='#1f77b4', alpha=0.1)
            ax.fill_between(x_freq, pred_freq, color='#ff7f0e', alpha=0.1)
            ax.grid(True, linestyle='--', alpha=0.6, which='both')

            ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none',
                      facecolor='white', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

            # max_true_idx = np.argmax(true_freq)
            # max_true_x = x_freq[max_true_idx]
            # max_true_y = true_freq[max_true_idx]
            # ax.plot(max_true_x, max_true_y, 'o', color='#1f77b4', markersize=8, zorder=5)
            # ax.annotate(
            #     f'Peak: ({max_true_x:.2f} Hz, {max_true_y:.2f})',
            #     xy=(max_true_x, max_true_y),
            #     xytext=(max_true_x + 1, max_true_y * 1.1),
            #     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', shrink=0.05, width=1),
            #     fontsize=10,
            #     color='#1f77b4'
            # )
            #
            # max_pred_idx = np.argmax(pred_freq)
            # max_pred_x = x_freq[max_pred_idx]
            # max_pred_y = pred_freq[max_pred_idx]
            # ax.plot(max_pred_x, max_pred_y, 'o', color='#ff7f0e', markersize=8, zorder=5)
            # ax.annotate(
            #     f'Peak: ({max_pred_x:.2f} Hz, {max_pred_y:.2f})',
            #     xy=(max_pred_x, max_pred_y),
            #     xytext=(max_pred_x + 1, max_pred_y * 1.02),
            #     arrowprops=dict(facecolor='#ff7f0e', edgecolor='#ff7f0e', shrink=0.05, width=1),
            #     fontsize=10,
            #     color='#ff7f0e'
            # )

            save_path = os.path.join(self.args.figures_path, f'feature_{n+1}_{mode}_freq.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            del true, pred, true_freq, pred_freq
            gc.collect()

        except Exception as e:
            print(f"Failed to plot feature_{n + 1}_{mode}: {str(e)}")
            for var in ['true', 'pred', 'true_freq', 'pred_freq']:
                if var in locals():
                    del locals()[var]
            gc.collect()

    def plot(self):
        if self.args.save_result_all:
            for n in tqdm(range(self.N), desc="Plotting in all mode"):
                self.single_plot('all', n)
        if self.args.save_result_every_batch:
            for n in tqdm(range(self.N), desc="Plotting in every_batch mode"):
                self.single_plot('every_batch', n)
        if self.args.save_result_every_pred_len:
            for n in tqdm(range(self.N), desc="Plotting in every_pred_len mode"):
                self.single_plot('every_pred_len', n)
        print(f"All feature plots completed, images saved to: {self.args.figures_path}")

    def plot_loss(self):
        metrics_to_plot = ['MAE', 'MAPE', 'MSE', 'Distribution Similarity']

        patterns = {
            'Train': {
                'MAPE': r'Average Training MAPE:\s*([-\d.]+)',
                'MAE': r'Average Training MAE:\s*([-\d.]+)',
                'MSE': r'Average Training MSE:\s*([-\d.]+)',
                'Distribution Similarity': r'Distribution Similarity:\s*([-\d.]+)'
            },
            'Val': {
                'MAPE': r'Average Validation MAPE:\s*([-\d.]+)',
                'MAE': r'Average Validation MAE:\s*([-\d.]+)',
                'MSE': r'Average Validation MSE:\s*([-\d.]+)',
                'Distribution Similarity': r'Distribution Similarity:\s*([-\d.]+)'
            }
        }

        data = {
            'Train': {m: [] for m in metrics_to_plot},
            'Val': {m: [] for m in metrics_to_plot}
        }

        try:
            with open(os.path.join(self.args.output_path, f'data.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to read file: {e}")
            return

        for line in lines:
            if 'Average Training' in line:
                for metric, pattern in patterns['Train'].items():
                    match = re.search(pattern, line)
                    if match:
                        data['Train'][metric].append(float(match.group(1)))

            elif 'Average Validation' in line:
                for metric, pattern in patterns['Val'].items():
                    match = re.search(pattern, line)
                    if match:
                        data['Val'][metric].append(float(match.group(1)))

        valid_metrics = [m for m in metrics_to_plot if len(data['Train'][m]) > 0 or len(data['Val'][m]) > 0]

        if not valid_metrics:
            print("The specified loss metric data was not found in the log!")
            return

        num_metrics = len(valid_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, valid_metrics):
            epochs_train = range(1, len(data['Train'][metric]) + 1)
            epochs_val = range(1, len(data['Val'][metric]) + 1)

            if data['Train'][metric]:
                ax.plot(epochs_train, data['Train'][metric], marker='o', linestyle='-', color='tab:blue',
                        label=f'Train {metric}')

            if data['Val'][metric]:
                ax.plot(epochs_val, data['Val'][metric], marker='s', linestyle='--', color='tab:orange',
                        label=f'Validation {metric}')

            ax.set_title(f'{metric} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()

        save_path = os.path.join(self.args.figures_path, 'loss.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image successfully saved to: {save_path}")

        plt.show()
