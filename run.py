import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    parser = argparse.ArgumentParser(description='MWMixer')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        choices=['long_term_forecast'], help='forecasting task')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, default='MWMixer', choices=['MWMixer'],
                        help='model name')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')

    # MWMixer configuration
    parser.add_argument('--modwt_level', type=int, default=1, help='levels of modwt')
    parser.add_argument('--wavelet', type=str, default='haar', help='wavelet of modwt, ' \
                'options:[haar, db2, db3, db5, sym2, sym3, sym5, bior3.1, bior3.5, coif4, coif5]')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--msloss_rate', type=float, default=0.1, help='multi-scale loss coefficient')

    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')


    # data loader
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='number of repeated experiments')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function:MASE, MSE, MAE, RMSE, MAPE, MSLE, Huber, SMAPE')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # model define
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # GPU
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=True,
                        help='enable GPU acceleration when the selected backend is available')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')



    args = parser.parse_args()
    if args.gpu_type == 'cuda' and torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    elif (args.gpu_type == 'mps' and args.use_gpu and hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available()):
        args.device = torch.device('mps')
        print('Using MPS')
    else:
        args.use_gpu = False
        args.device = torch.device('cpu')
        print('Using CPU')

    if args.use_amp and args.device.type != 'cuda':
        parser.error('--use_amp currently requires a CUDA device')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_pl{}_bs{}_wl{}_le{}_dsl{}_dsw{}_dm{}_dp{}_el{}_msr{}_eb{}_amp{}_{}'.format(
                args.model_id,
                args.des,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.batch_size,
                args.wavelet,
                args.modwt_level,
                args.down_sampling_layers,
                args.down_sampling_window,
                args.d_model,
                args.dropout,
                args.e_layers,
                args.msloss_rate,
                args.embed,
                args.use_amp, ii)

            print('>>>>>>>start training : {}<<<<<<<'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_pl{}_bs{}_wv{}_lv{}_dsl{}_dsw{}_dm{}_dp{}_el{}_msr{}_eb{}_amp{}_{}'.format(
                args.model_id,
                args.des,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.batch_size,
                args.wavelet,
                args.modwt_level,
                args.down_sampling_layers,
                args.down_sampling_window,
                args.d_model,
                args.dropout,
                args.e_layers,
                args.msloss_rate,
                args.embed,
                 args.use_amp, ii)

        print('>>>>>>>testing : {}<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
