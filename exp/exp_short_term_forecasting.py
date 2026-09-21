from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.scale_loss_reference import MultiScaleLoss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.multi_scale_loss = MultiScaleLoss(down_sampling_method='avg')
        self.criterion = self._select_criterion(args.loss)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='SMAPE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def backbone_loss(self, y_true, final_pred, scale_preds_dict):
        """
        计算主模型损失（包含多尺度损失）
        
        Args:
            y_true: 真实标签 [B, pred_len, C]
            scale_preds_dict: 多尺度预测字典，包含 "scale_preds" 键
            final_pred: 最终融合预测结果 [B, pred_len, C]
            
        Returns:
            total_loss: 总损失
            loss_info: 损失详细信息字典
        """
        # 计算final预测损失
        final_loss = self.criterion(final_pred, y_true)
        # 计算多尺度损失
        if scale_preds_dict is not None:
            # 使用MultiScaleLoss计算多尺度损失
            multi_scale_total_loss, global_loss, local_loss, weight_global, scaleloss_details = self.multi_scale_loss(
                y_true, scale_preds_dict)
            # weight_final = torch.sigmoid(2 * (final_loss /(final_loss + multi_scale_total_loss + 1e-8) - 0.5))
            weight_final = torch.sigmoid(final_loss /(multi_scale_total_loss + 1e-8))
            weight_multi_scale = 1 - weight_final
            # 加权组合两部分损失
            # total_loss = weight_final * final_loss + weight_multi_scale * multi_scale_total_loss
            total_loss = 0.8 * final_loss + 0.2 * multi_scale_total_loss
            
            # 详细损失信息
            loss_info = {
                "总损失": total_loss.item(),
                "最终损失": final_loss.item(),
                "多尺度损失": multi_scale_total_loss.item(),
                "全局损失": global_loss.item(),
                "局部损失": local_loss.item(),
                "尺度1": scaleloss_details[0] if len(scaleloss_details) > 0 else 0.0,
                "尺度2": scaleloss_details[1] if len(scaleloss_details) > 1 else 0.0,
                "尺度3": scaleloss_details[2] if len(scaleloss_details) > 2 else 0.0,
                "尺度4": scaleloss_details[3] if len(scaleloss_details) > 3 else 0.0,
                "全局权重": weight_global.item(),
                "局部权重": (1 - weight_global).item(),
                "最终损失权重": weight_final.item(),
                "尺度损失权重": weight_multi_scale.item()
            }
        else:
            # 如果不使用多尺度损失，只用final损失
            total_loss = final_loss
            loss_info = {
                "total_loss": total_loss.item(),
                "final_loss": final_loss.item()
            }
        
        return total_loss, loss_info

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        loss_history = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_loss_info = []

            print("Epoch: {}/{}".format(epoch + 1, self.args.train_epochs))
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.model in ['MWMixer', 'MWMixer_1', 'MWMixer_diagcomp']:
                    scale_preds = outputs["scale_preds"]
                    outputs = outputs["final"][:, -self.args.pred_len:, f_dim:]#融合结果
                    loss_value, loss_info = self.backbone_loss(batch_y, outputs, scale_preds)
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)

                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if epoch_loss_info:
                avg_loss_info = {}
                for key in epoch_loss_info[0].keys():
                    avg_loss_info[key] = np.mean([info[key] for info in epoch_loss_info])
                loss_history.append(avg_loss_info)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                model_outputs = self.model(x[id_list[i]:id_list[i + 1]], None,
                                        dec_inp[id_list[i]:id_list[i + 1]], None)
                
                if self.args.model in ['MWMixer', 'MWMixer_1', 'MWMixer_diagcomp']:
                    if isinstance(model_outputs, dict):
                        # 提取最终融合预测结果
                        final_outputs = model_outputs["final"]
                    else:
                        final_outputs = model_outputs
                else:
                    final_outputs = model_outputs
                    
                outputs[id_list[i]:id_list[i + 1], :, :] = final_outputs.detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                model_outputs = self.model(x[id_list[i]:id_list[i + 1]], None,
                                     dec_inp[id_list[i]:id_list[i + 1]], None)
            
                if self.args.model in ['MWMixer', 'MWMixer_1', 'MWMixer_diagcomp']:
                    if isinstance(model_outputs, dict):
                        final_outputs = model_outputs["final"]
                    else:
                        final_outputs = model_outputs
                else:
                    final_outputs = model_outputs
                outputs[id_list[i]:id_list[i + 1], :, :] = final_outputs
                
                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
