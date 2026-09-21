import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleLoss(nn.Module):
    def __init__(self, down_sampling_method='avg'):
        super(MultiScaleLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.down_sampling_method = down_sampling_method
    
    def forward(self, y_true, scale_preds_dict, global_ratio=0.5):
        """
        计算多尺度损失
        
        Args:
            y_true (torch.Tensor): 原始尺度的真实值，形状 [batch_size, seq_len, feature_dim]
            scale_preds_dict (dict): 包含'final'和'scale_preds'的字典:
                - final: 最终融合预测 [batch_size, seq_len, feature_dim]
                - scale_preds: 多尺度预测列表，按分辨率从高到低排列
            global_ratio (float): 全局损失权重比例因子 (0-1)
        
        Returns:
            total_loss (torch.Tensor): 加权后的总损失
            global_loss (torch.Tensor): 全局损失
            local_loss (torch.Tensor): 局部损失
            weight_global (torch.Tensor): 全局损失权重
        """
        # 提取多尺度预测列表
        scale_preds = scale_preds_dict
        num_scales = len(scale_preds)
        
        # 处理只有一个尺度的情况
        if num_scales <= 1:
            # 如果只有一个尺度，则只返回MSE损失
            pred = scale_preds[0]
            # 确保预测与真实值的长度匹配
            min_len = min(pred.shape[1], y_true.shape[1])
            loss = self.mse(pred[:, :min_len], y_true[:, :min_len])
            return loss, loss, torch.tensor(0.0, device=y_true.device), torch.tensor(1.0, device=y_true.device)
        
        # ====== 1. 全局损失计算 ======
        global_loss = 0.0
        
        # 为每个尺度下采样真实值并计算损失
        y_true_multiscale = self._multi_scale_decompose_targets(y_true, num_scales)
        
        scaleloss_details = []
        for scale_idx, pred in enumerate(scale_preds):
            # 获取当前尺度下采样后的真实值
            downsampled_true = y_true_multiscale[scale_idx]
            
            # 确保预测值和真实值形状匹配
            min_len = min(pred.size(1), downsampled_true.size(1))
            truncated_pred = pred[:, :min_len]
            truncated_true = downsampled_true[:, :min_len]
            
            # 计算当前尺度MSE
            scale_loss = self.mse(truncated_pred, truncated_true)
            scaleloss_details.append(scale_loss.item())
            global_loss += scale_loss
        
        # 平均所有尺度的全局损失
        global_loss /= num_scales
        
        # ====== 2. 局部损失计算 ======
        local_loss = 0.0
        valid_pairs = 0
        
        for i in range(num_scales):      # 高分辨率尺度
            for j in range(i+1, num_scales):  # 低分辨率尺度 (i < j)
                # 高分辨率和低分辨率预测
                high_res_pred = scale_preds[i]
                low_res_pred = scale_preds[j]
                
                # 下采样高分辨率预测到低分辨率
                downsampled_high_res = self._adjust_tensor_length(
                    high_res_pred, low_res_pred.size(1))
                
                # 计算一致性损失
                consistency_loss = self.mse(downsampled_high_res, low_res_pred)
                local_loss += consistency_loss
                valid_pairs += 1
        
        # 计算平均局部损失 (避免除零)
        if valid_pairs > 0:
            local_loss /= valid_pairs
        else:
            local_loss = torch.tensor(0.0, device=y_true.device)
        
        # ====== 3. 动态权重计算 ======
        # 根据两种损失的相对大小动态调整权重
        # 当global_ratio > 0.5时，全局损失权重更大
        # 当global_ratio < 0.5时，局部损失权重更大
        eps = 1e-8
        weight_global = torch.sigmoid(2 * (global_loss /(global_loss + local_loss + eps) - 0.5))
        weight_local = 1 - weight_global
        
        # ====== 4. 总损失计算 ======
        total_loss = weight_global * global_loss + weight_local * local_loss
        # for abl supervison loss
        # total_loss = local_loss
        # for abl consistency loss
        # total_loss = global_loss

        return total_loss, global_loss, local_loss, weight_global, scaleloss_details
    
    def _multi_scale_decompose_targets(self, y_true, num_scales):
        """
        将目标标签分解为多个尺度
        
        Args:
            y_true (torch.Tensor): 原始真实标签 [B, T, C]
            num_scales (int): 尺度数量
            
        Returns:
            y_true_list (list): 多尺度真实值列表
        """
        # 初始尺度是原始尺度
        y_true_list = [y_true]
        
        # 逐步下采样创建较低分辨率的真实值
        for i in range(1, num_scales):
            # 每一级缩小为上一级的一半
            factor = 2
            downsampled = self._adjust_tensor_length(
                y_true_list[-1], y_true_list[-1].size(1) // factor)
            y_true_list.append(downsampled)
        
        return y_true_list
    
    def _adjust_tensor_length(self, tensor, target_length):
            """
            Args:
                tensor (torch.Tensor): 输入张量 [B, src_len, D]
                target_length (int): 目标长度
                
            Returns:
                adjusted (torch.Tensor): 调整后的张量 [B, target_length, D]
            """
            B, src_len, D = tensor.shape
            
            # 如果长度已经匹配，直接返回
            if src_len == target_length:
                return tensor
            
            # 转置为通道优先格式 [B, D, src_len]
            tensor = tensor.transpose(1, 2)
            if src_len > target_length:
                if self.down_sampling_method == 'avg':
                    adjusted = F.adaptive_avg_pool1d(tensor, target_length)
                elif self.down_sampling_method == 'max':
                    adjusted = F.adaptive_max_pool1d(tensor, target_length)[0]
                elif self.down_sampling_method == 'conv':
                    # 简化的卷积下采样
                    padding = 1 if torch.__version__ >= '1.5.0' else 2
                    # 创建临时卷积层
                    down_conv = nn.Conv1d(in_channels=D, out_channels=D,
                                        kernel_size=3, padding=padding,
                                        stride=src_len//target_length,
                                        padding_mode='circular',
                                        bias=False).to(tensor.device)
                    adjusted = down_conv(tensor)
                    # 如果长度不精确匹配，使用自适应池化调整
                    if adjusted.size(2) != target_length:
                        adjusted = F.adaptive_avg_pool1d(adjusted, target_length)
                else:
                    # 默认使用线性插值
                    adjusted = F.interpolate(tensor, size=target_length, mode='linear', align_corners=False)
            else:
                # 上采样 (使用线性插值)
                adjusted = F.interpolate(tensor, size=target_length, mode='linear', align_corners=False)
            
            # 转回原始格式 [B, target_length, D]
            return adjusted.transpose(1, 2)