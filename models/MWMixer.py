import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.StandardNorm import Normalize
from layers.modwt_pytorch import WaveletTransform


class IndependentScaleConfidenceFusion(nn.Module):
    def __init__(self, configs):
        super(IndependentScaleConfidenceFusion, self).__init__()
        self.configs = configs
        self.num_scales = configs.down_sampling_layers + 1
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        self.scale_input_lens = [
            configs.seq_len // (configs.down_sampling_window ** i) 
            for i in range(self.num_scales)
        ]

        self.feature_extractors = nn.ModuleList()
        self.confidence_mlps = nn.ModuleList()
        for i in range(self.num_scales):
            
            history_len = self.scale_input_lens[i]
            total_len = history_len + self.pred_len
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.AvgPool1d(kernel_size=2, stride=2) # 下采样
                )
            )
            
            # 计算卷积后的维度
            conv_out_len = total_len // 2
            input_dim = conv_out_len * 8
            confidence_mlp = nn.Sequential(
                nn.Linear(input_dim, self.configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.configs.d_model, self.configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.configs.d_model, self.pred_len),
            )
            self.confidence_mlps.append(confidence_mlp)
    
    def _build_full_sequence(self, x_enc, scale_pred):
        full_seq = torch.cat([x_enc, scale_pred], dim=1)
        return full_seq
    
    def _calculate_scale_confidence(self, scale_idx, x_enc, scale_pred):

        full_seq = self._build_full_sequence(x_enc, scale_pred)  # [B, T_full, C]
        _ , pred_len, _ = scale_pred.shape
        B, T_full, C = full_seq.shape
        full_seq_flat = full_seq.transpose(1, 2).reshape(B * C, T_full)  # [B * C, T_full]
        
        x_in = full_seq.permute(0, 2, 1).reshape(B * C, 1, T_full)

        features = self.feature_extractors[scale_idx](x_in) # [B*C, 8, T/2]
        features_flat = features.reshape(B * C, -1)
        

        confidence_logits = self.confidence_mlps[scale_idx](features_flat)  # [B*C, pred_len]
        confidence_logits = confidence_logits.reshape(B, C, pred_len)
        confidence = confidence_logits 
        confidence = torch.tanh(confidence)
        return confidence

    def forward(self, x_enc_list, scale_preds):

        batch_size, pred_len, feature_dim = scale_preds[0].shape
        device = scale_preds[0].device
        
        # 1. 计算各尺度的置信度
        scale_confidences = []
        for i in range(self.num_scales):
            x_enc = x_enc_list[i]  # [B, T_scale, C]
            scale_pred = scale_preds[i]  # [B, pred_len, C]
            
            # 计算该尺度的置信度
            confidence = self._calculate_scale_confidence(i, x_enc, scale_pred)  # [B, pred_len]
            scale_confidences.append(confidence)
        

        stacked_confidences = torch.stack(scale_confidences, dim=-1)
        
        confidence_weights = F.softmax(stacked_confidences, dim=-1)  # [B, pred_len, num_scales]

        stacked_preds = torch.stack(scale_preds, dim=-1).permute(0, 2, 1, 3)
        fused_prediction = torch.sum(stacked_preds * confidence_weights, dim=-1)

        fused_prediction = fused_prediction.permute(0, 2, 1)

        return fused_prediction, confidence_weights, scale_confidences

class MultiScaleFreqMix(nn.Module):
    def __init__(self, configs):
        super(MultiScaleFreqMix, self).__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.num_scales = configs.down_sampling_layers + 1
        self.num_freqs = configs.modwt_level + 1  
        
        # detail_mlps[scale_idx][freq_idx] 对应尺度scale_idx的第freq_idx个细节系数的MLP
        self.detail_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model *2),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Sigmoid()  
                ) for _ in range(configs.modwt_level) 
            ]) for _ in range(self.num_scales - 1)  
        ])
        

        self.approx_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Sigmoid()  
            ) for _ in range(self.num_scales - 1)  
        ])
        
        
    def forward(self, mixed_out_list):
        num_scales = len(mixed_out_list)
        B, C, freq_bands, d_model = mixed_out_list[0].shape
        

        interacted_features = [feat.clone() for feat in mixed_out_list]
        

        for freq_idx in range(self.configs.modwt_level):  
            for scale_idx in range(num_scales - 1):  
                big_detail = interacted_features[scale_idx][:, :, freq_idx, :]  # [B, C, d_model]

                big_detail_weights = self.detail_mlps[scale_idx][freq_idx](big_detail)  # [B, C, d_model]

                next_detail = interacted_features[scale_idx + 1][:, :, freq_idx, :]  # [B, C, d_model]

                small_detail_res = big_detail * big_detail_weights + next_detail
                new_features = interacted_features[scale_idx + 1].clone()
                new_features[:, :, freq_idx, :] = small_detail_res
                interacted_features[scale_idx + 1] = new_features
        

        approx_idx = self.configs.modwt_level
        for scale_idx in range(num_scales - 1, 0, -1):  # 从最粗糙尺度到最精细尺度(倒序)

            small_approx = interacted_features[scale_idx][:, :, approx_idx, :]  # [B, C, d_model]

            small_approx_weights = self.approx_mlps[scale_idx - 1](small_approx)  # [B, C, d_model]     
            big_approx = interacted_features[scale_idx - 1][:, :, approx_idx, :]  # [B, C, d_model]         
            big_approx_res = small_approx * small_approx_weights + big_approx
            new_features = interacted_features[scale_idx - 1].clone()
            new_features[:, :, approx_idx, :] = big_approx_res
            interacted_features[scale_idx - 1] = new_features
        
        return interacted_features
  
 
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.enc_in = configs.enc_in
        self.layer = configs.e_layers
        
        self.enc_embedding_layers = nn.ModuleList([
            DataEmbedding_inverted(
                configs.seq_len // (configs.down_sampling_window ** i), configs.d_model, configs.embed, configs.freq, configs.dropout
            ) for i in range(configs.down_sampling_layers + 1)])
        
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # 为每个尺度创建小波分解、重构、映射层
        self.wavedecomp_layers = nn.ModuleList([
            WaveletTransform(
                L=configs.seq_len // (configs.down_sampling_window ** i),
                level=configs.modwt_level,
                method=configs.wavelet
            ) for i in range(configs.down_sampling_layers + 1)
        ])
        self.iwavedecomp_layers = nn.ModuleList([
            WaveletTransform(
                L=configs.pred_len // (configs.down_sampling_window ** i),
                level=configs.modwt_level,
                method=configs.wavelet
            ) for i in range(configs.down_sampling_layers + 1)
        ])
        # self.wavelet_map_layers = nn.ModuleList([nn.ModuleList([
        #         nn.Linear(configs.d_model, configs.pred_len // (configs.down_sampling_window ** scale_idx)) 
        #         for _ in range(configs.modwt_level + 1)])
        #         for scale_idx in range(configs.down_sampling_layers + 1)])
        

        self.wavelet_map_layers = nn.ModuleList()
        for scale_idx in range(configs.down_sampling_layers + 1):
            scale_map_layers = nn.ModuleList()
            target_len = configs.pred_len // (configs.down_sampling_window ** scale_idx)
            
            for _ in range(configs.modwt_level + 1):
                map_layer = nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model*2),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model * 2, target_len)
                )
                scale_map_layers.append(map_layer)
            self.wavelet_map_layers.append(scale_map_layers)

        self.freq_scale_mix = MultiScaleFreqMix(configs)
        self.freq_scale_mix_layers = nn.ModuleList([MultiScaleFreqMix(configs) for _ in range(configs.e_layers)])
        # 多尺度融合权重
        self.independent_scale_fusion = IndependentScaleConfidenceFusion(configs)
        self.upsample_mlps = nn.ModuleList()
        for i in range(configs.down_sampling_layers + 1):
            if i == 0:
                self.upsample_mlps.append(nn.Identity())
            else:
                curr_len = configs.pred_len // (configs.down_sampling_window ** i)
                self.upsample_mlps.append(nn.Sequential(
                    nn.Linear(curr_len, configs.d_model*2),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model*2, configs.pred_len)
                ))

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):  #[原始，下采样1，下采样2，...]

        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='replicate',
                                  bias=False,
                                  groups=self.configs.enc_in)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc
        
        return x_enc, x_mark_enc
    
    def modwt_decompose(self, x_enc_list):
        """[B, T, C] -> [B, C, level+1, T]"""
        x_modwt_list = []
        for scale_idx, x_scale in enumerate(x_enc_list):
            # x_scale: [B, T, C] -> x_scale_modwt: [B, level+1, T, C]
            x_scale_modwt = self.wavedecomp_layers[scale_idx].modwt(x_scale)
            x_scale_modwt = x_scale_modwt.permute(0, 2, 3, 1)
            x_modwt_list.append(x_scale_modwt)
             
        return x_modwt_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        _, T, enc_in = x_enc.shape  # [B, T, C]
        x_enc, x_mark_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)#[B, T, C]
        x_list = []
        x_enc_original = []  # 原始尺度的输入
        for i, x in zip(range(len(x_enc)), x_enc, ):
                x = self.normalize_layers[i](x, 'norm')
                x_list.append(x)
                x_enc_original.append(x[:, :, :enc_in])
        # MODWT分解 - 得到每个尺度的MODWT系数 [B, C, T, level+1]
        x_enc_modwt = self.modwt_decompose(x_list)
        enc_out_list = []
        for i, x_modwt in enumerate(x_enc_modwt):
            B, freq_bands, T, C = x_modwt.shape  # x_modwt: [B, level+1, T, C]
            if x_mark_list is not None:
                x_mark = x_mark_list[i]  # [B, T, mark_dim]
            else:
                x_mark = None
            
            # 逐频带进行嵌入
            x_modwt_reshaped = x_modwt.reshape(B * freq_bands, T, C)  # [B*freq_bands, T, C]
            if x_mark is not None:
                x_mark_expanded = x_mark.unsqueeze(1).expand(-1, freq_bands, -1, -1)  # [B, freq_bands, T, mark_dim]
                x_mark_reshaped = x_mark_expanded.reshape(B * freq_bands, T, -1)
                freq_emb_batch = self.enc_embedding_layers[i](x_modwt_reshaped, x_mark_reshaped)
            else:
                freq_emb_batch = self.enc_embedding_layers[i](x_modwt_reshaped, None)

            _, C_emb, d_model = freq_emb_batch.shape
            enc_out = freq_emb_batch.reshape(B, freq_bands, C_emb, d_model).transpose(1, 2)  # [B, C, freq_bands, d_model]
            
            enc_out_list.append(enc_out)



        # 频带尺度交互
        freq_scale_mixed_list = enc_out_list
        for layer in self.freq_scale_mix_layers:
            freq_scale_mixed_list = layer(freq_scale_mixed_list)

        # 反变换
        map_wavelet_coeffs = []
        for i, mixed_out in enumerate(freq_scale_mixed_list):
            B, C, freq_bands, d_model = mixed_out.shape
            freq_coeffs_list = []
            for freq_idx in range(freq_bands):
                # 取出单个频带: [B, C, d_model]
                mixed_out_freq = mixed_out[:, :, freq_idx, :]
                wavelet_coeffs = self.wavelet_map_layers[i][freq_idx](mixed_out_freq)
                freq_coeffs_list.append(wavelet_coeffs.unsqueeze(-1))
            scale_wavelet_coeffs = torch.cat(freq_coeffs_list, dim=-1)
            map_wavelet_coeffs.append(scale_wavelet_coeffs)
        reconstructed_list = []
        for i, wavelet_coeffs in enumerate(map_wavelet_coeffs):
            B, C, T, freq_bands = wavelet_coeffs.shape
            reconstructed = self.iwavedecomp_layers[i].imodwt(wavelet_coeffs)  # [B, C, T]
            reconstructed = reconstructed.permute(0, 2, 1)  # [B, T, C]
            reconstructed_list.append(reconstructed)

        max_pred_len = max([pred.shape[1] for pred in reconstructed_list])
        adjusted_predictions = []
        for pred in reconstructed_list:
            pred = pred[:, :, :enc_in]
            if pred.shape[1] < max_pred_len:
                # 线性插值扩展预测长度
                pred = pred.transpose(1, 2)  # [B, C, pred_len]
                pred = F.interpolate(pred, size=max_pred_len, mode='linear', align_corners=False)
                pred = pred.transpose(1, 2)  # [B, max_pred_len, C]
            adjusted_predictions.append(pred)
    

        # adjusted_predictions = []
        # for i, pred in enumerate(reconstructed_list):
        #     pred = pred[:, :, :enc_in]
        #     if i > 0:
        #         # 使用MLP映射扩展预测长度，原始尺度(i=0)不需要映射
        #         pred = pred.transpose(1, 2)  # [B, C, curr_len]
        #         pred = self.upsample_mlps[i](pred)  # [B, C, pred_len]
        #         pred = pred.transpose(1, 2)  # [B, pred_len, C]
        #     adjusted_predictions.append(pred)

        _, confidence_weights, raw_confidences = self.independent_scale_fusion(
            x_enc_original,
            adjusted_predictions
        )


        scale_predictions_denorm_adjusted = []
        for i, pred in enumerate(adjusted_predictions):
            pred_denorm = self.normalize_layers[i](pred, 'denorm')
            scale_predictions_denorm_adjusted.append(pred_denorm)

        stacked_preds_denorm = torch.stack(scale_predictions_denorm_adjusted, dim=-1).permute(0, 2, 1, 3)
        
        final_prediction_denorm = torch.sum(stacked_preds_denorm * confidence_weights, dim=-1)
        final_prediction_denorm = final_prediction_denorm.permute(0, 2, 1) # [B, pred_len, C]

        raw_scale_preds_denorm = []
        for i, pred in enumerate(reconstructed_list):  
            pred_denorm = self.normalize_layers[i](pred[:, :, :enc_in], 'denorm')
            raw_scale_preds_denorm.append(pred_denorm)


        scale_preds_dict = {
            "final": final_prediction_denorm,
            "scale_preds": raw_scale_preds_denorm
        }

        
        return scale_preds_dict

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            outputs = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return outputs
        else:
            raise ValueError('Other tasks implemented yet')
