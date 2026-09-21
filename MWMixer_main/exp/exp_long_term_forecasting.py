from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.scale_loss_reference import MultiScaleLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.multi_scale_loss = MultiScaleLoss(
            down_sampling_method=getattr(args, 'down_sampling_method', 'avg'))
        self.criterion = self._select_criterion()

    def _build_model(self):
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

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 
    def backbone_loss(self, args, y_true, final_pred, scale_preds_dict):

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
            total_loss =  final_loss +  args.msloss_rate * multi_scale_total_loss
            
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                output = outputs["final"][:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = output.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_time = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()

        loss_history = []
        # ========================================
        global_step_time_sum = 0.0
        global_step_time_cnt = 0    
        # ========================================    
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_loss_info = []

            # ==== [新增] 统计 training time (s/iter) ====
            warmup_iters = 10
            step_time_sum = 0.0
            step_time_cnt = 0
            # ========================================

            # ==== [新增] 统计显存峰值（per-epoch）====
            mem_peak_alloc_mb = None
            mem_peak_reserved_mb = None
            if self.args.use_gpu:
                torch.cuda.reset_peak_memory_stats()
                # 可选：清理缓存（不影响已分配tensor，但会影响reserved的观感）
                # torch.cuda.empty_cache()
            # =======================================

            print("Epoch: {}/{}".format(epoch + 1, self.args.train_epochs))
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ==== [新增] iter 计时起点（只统计计算部分） ====
                if self.args.use_gpu:
                    torch.cuda.synchronize()
                iter_t0 = time.perf_counter()
                # =================================================

                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        scale_preds = outputs["scale_preds"]
                        output = outputs["final"][:, -self.args.pred_len:, f_dim:]#融合结果
                        loss, loss_info = self.backbone_loss(self.args, batch_y, output, scale_preds)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    scale_preds = outputs["scale_preds"]
                    output = outputs["final"][:, -self.args.pred_len:, f_dim:]#融合结果
                    loss, loss_info = self.backbone_loss(self.args, batch_y, output, scale_preds)
                    train_loss.append(loss.item())

                epoch_loss_info.append(loss_info)

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # ==== [新增] iter 计时终点（GPU 同步） ====
                if self.args.use_gpu:
                    torch.cuda.synchronize()
                iter_t1 = time.perf_counter()
                if iter_count > warmup_iters:
                    step_time_sum += (iter_t1 - iter_t0)
                    step_time_cnt += 1
                # =================================================

            # ==== [新增] epoch 结束时读取显存峰值 ====
            if self.args.use_gpu:
                torch.cuda.synchronize()
                mem_peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
                mem_peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                print(f"Peak GPU memory: allocated={mem_peak_alloc_mb:.2f} MB, reserved={mem_peak_reserved_mb:.2f} MB")
            # =====================================

            if epoch_loss_info:
                avg_loss_info = {}
                for key in epoch_loss_info[0].keys():
                    avg_loss_info[key] = np.mean([info[key] for info in epoch_loss_info])
                loss_history.append(avg_loss_info)

            # ==== [新增] 打印本 epoch 平均 s/iter ====
            if step_time_cnt > 0:
                avg_s_per_iter = step_time_sum / step_time_cnt
                print(f"Avg training time: {avg_s_per_iter:.6f} s/iter (warmup {warmup_iters} iters)")
            # ========================================
                # [新增] 2. 将本 epoch 的有效数据累加到全局
                global_step_time_sum += step_time_sum
                global_step_time_cnt += step_time_cnt
            # ========================================

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # if loss_history:
            #     latest_loss = loss_history[-1]
            #     loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in latest_loss.items()])
            #     print(f"loss_str: {loss_str}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print("train cost time: {}".format(time.time() - train_time))

        # ========================================
        # [新增] 3. 训练结束后计算并打印全局平均值 (这是顶会通常汇报的数值)
        if global_step_time_cnt > 0:
            global_avg_speed = global_step_time_sum / global_step_time_cnt
            print("=" * 40)
            print(f"Final Statistics (Top-tier Standard):")
            print(f"Overall Avg Speed: {global_avg_speed:.6f} s/iter")
            print(f"Total Measured Steps: {global_step_time_cnt}")
            print("=" * 40)
        # ========================================
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                output = outputs["final"][:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                output = output.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if output.shape[-1] != batch_y.shape[-1]:
                        output = np.tile(output, [1, 1, int(batch_y.shape[-1] / output.shape[-1])])
                    output = test_data.inverse_transform(output.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                output = output[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = output
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('test_mse:{}, test_mae:{}'.format(mse, mae))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('test_mse:{}, test_mae:{}\n'.format(mse, mae))
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
