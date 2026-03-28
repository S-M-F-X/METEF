import argparse
from time import time
import gc
import os
import torch

from solver.solver import Solver
from util.seed import fixSeed
from util.config import get_config
from util.save_name import get_save_name
from util.new import plot_training_logs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Print """

    parser.add_argument('--print', type=bool, default=False,
                        help='Whether to print detailed information at each step. Default: False')
    parser.add_argument('--print_train_loss', type=bool, default=True,
                        help='Whether to print auxiliary losses for each epoch. Default: True')
    parser.add_argument('--save_result_all', type=bool, default=False,
                        help='Whether to save all test set data. Default: False')
    parser.add_argument('--save_result_every_batch', type=bool, default=True,
                        help='Whether to save test set data at the first time step of each batch. Default: False')
    parser.add_argument('--save_result_every_pred_len', type=bool, default=False,
                        help='Whether to save test set data sampled every pred_len time steps across all time steps. Default: False')
    parser.add_argument('--plot', type=bool, default=True,
                        help='Whether to generate plots based on the saved test set data. Default: False')
    parser.add_argument('--plot_loss', type=bool, default=True,
                        help='Whether to plot the loss trend curve')
    parser.add_argument('--max_cache', type=int, default=1000,
                        help='Cache threshold (length of concatenated test predictions and true values before writing to CSV). Default: 1000')

    """ Basic """

    parser.add_argument('--seed', type=int, default=1120,
                        help='Random seed. Default: 1120')
    parser.add_argument('--model', type=str, default='Model',
                        help='Backbone network model')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Dataset name.')
    parser.add_argument('--data_path', type=str, default='./dataset/',
                        help='Dataset path. Default: ./dataset/')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='Only test the model without training')
    parser.add_argument('--test_name', type=int, default=1,
                        help='When only_test is True, must specify the test folder index (n), corresponding to save_name(n)')

    """ Dataset """

    parser.add_argument('--pi_fen', type=int, default='2',
                        help='Dataset partition method')
    parser.add_argument('--time_fen', type=int, default='1',
                        help='Timestamp decomposition method')

    """ GPU """

    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Whether to use GPU. Default: True')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index to use. Default: 0')

    args = parser.parse_args()
    config = get_config(args.model)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    save_name = (f'{args.model}_{args.dataset}_e{config.epoch}_'
                 f'p{config.hist_len}_q{config.pred_len}_seed{args.seed}')

    if args.only_test:
        save_name = f'{save_name}({args.test_name})'
        model_path = f'./output/{save_name}/model/'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Test path does not exist: {model_path}")
    else:
        save_name = get_save_name(save_name)

    args.output_path = f'./output/{save_name}/'
    args.result_path = f'./output/{save_name}/result/'
    args.model_path = f'./output/{save_name}/model/'
    args.figures_path = f'./output/{save_name}/figures/'
    print(args)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.figures_path):
        os.makedirs(args.figures_path)

    print('\n=====================Args/config========================')
    print('args: ', args)
    print('config: ', config)
    print('=================================================\n')

    with open('./result.txt', 'a', encoding="utf-8-sig") as f:
        f.write(f"--------------------{save_name}--------------------\n")
    with open(args.output_path + 'data.txt', 'a', encoding="utf-8-sig") as f:
        f.write(f"--------------------{save_name}--------------------\n")

    fixSeed(args.seed)

    print(f'\n>>>>>>>>  Initializing : {save_name}  <<<<<<<<\n')
    solver = Solver(args, config)

    train_time = 0

    if not args.only_test:
        print(f'\n>>>>>>>>  Training : {save_name}  <<<<<<<<\n')
        try:
            start = time()
            epoch = solver.train()
            train_time = (time() - start) / epoch
            print('Average time per epoch: {:.4f}s'.format(train_time))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Training interrupted by user, saving current model state...')
            interrupt_model_path = f'./output/{save_name}/interrupt_model/'
            if not os.path.exists(interrupt_model_path):
                os.makedirs(interrupt_model_path)
            model_file_path = os.path.join(interrupt_model_path, 'interrupted_model.pth')
            torch.save(solver.model.state_dict(), model_file_path)
            print(f'Interrupted model saved to: {model_file_path}')

    print(f'\n>>>>>>>>  Testing : {save_name}  <<<<<<<<\n')
    test_res, test_time, valid_lun, best_lun, best_decay = solver.test()
    print(f'Test time: {test_time:.4f}s')

    if not args.only_test:
        print_str = (f"Average training time per epoch: {train_time:.4f} s, Test time: {test_time:.4f} s, "
                     f"Model size: {test_res['size']:.4f} MB, Total model parameters: {test_res['total_params']}, "
                     f"Trainable model parameters: {test_res['trainable_params']}\n"
                     f"Effective training epochs: {valid_lun}, Best model at epoch {best_lun}, "
                     f"Best model obtained after {best_decay} learning rate decays")
    else:
        print_str = (f"Test time: {test_time:.4f} s, Model size: {test_res['size']:.4f} MB, "
                     f"Total model parameters: {test_res['total_params']}, "
                     f"Trainable model parameters: {test_res['trainable_params']}")
    print_str += (f"\n--------------------Average Test Statistics--------------------"
                  f"\nMAPE: {test_res['mean-MAPE']:.6f} || MAE: {test_res['mean-MAE']:.6f} || "
                  f"MSE: {test_res['mean-MSE']:.6f} || Average feature-wise RMSE: {test_res['mean-RMSE_feature']:.6f}"
                  f" || Average overall RMSE: {test_res['mean-RMSE_all']:.6f} || "
                  f"Distribution similarity: {test_res['mean-pearson']:.6f} || "
                  f"Adjusted R² Score: {test_res['mean-R2_adj']:.6f}")
    for n in range(len(test_res['MAPE'])):
        print_str += (f"\n\n--------------------Test Statistics for Feature {n + 1}--------------------"
                      f"\nMAPE: {test_res['MAPE'][n]:.6f} || MAE: {test_res['MAE'][n]:.6f} || "
                      f"MSE: {test_res['MSE'][n]:.6f} || RMSE: {test_res['RMSE'][n]:.6f} || "
                      f"Distribution similarity: {test_res['pearson'][n]:.6f} || "
                      f"Adjusted R² Score: {test_res['R2_adj'][n]:.6f}")
    print(print_str)

    with open('./result.txt', 'a', encoding="utf-8-sig") as f:
        f.write(print_str + '\n\n\n\n')

    with open(os.path.join(args.output_path, 'data.txt'), 'a', encoding="utf-8-sig") as f:
        f.write(f'\n' + print_str + '\n\n\n\n')

    print(f"\n\nMAE={test_res['mean-MAE']:.6f}, Average training time per epoch: {train_time:.4f} s, "
          f"Effective training epochs: {valid_lun}, Best model at epoch {best_lun}, "
          f"Best model obtained after {best_decay} learning rate decays")

    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    print('\nExecution Completed')
