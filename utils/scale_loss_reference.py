import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    def __init__(self, down_sampling_method='avg'):
        super(MultiScaleLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.down_sampling_method = down_sampling_method

    def forward(self, y_true, scale_preds_dict, global_ratio=0.5):
        scale_preds = scale_preds_dict
        num_scales = len(scale_preds)

        if num_scales <= 1:
            pred = scale_preds[0]
            min_len = min(pred.shape[1], y_true.shape[1])
            loss = self.mse(pred[:, :min_len], y_true[:, :min_len])
            return (
                loss,
                loss,
                torch.tensor(0.0, device=y_true.device),
                torch.tensor(1.0, device=y_true.device),
                [loss.item()],
            )

        if self._use_full_resolution_global_only(y_true, scale_preds):
            return self._full_resolution_global_only_loss(y_true, scale_preds)

        return self._legacy_multiscale_loss(y_true, scale_preds)

    def _full_resolution_global_only_loss(self, y_true, scale_preds):
        num_scales = len(scale_preds)
        global_loss = 0.0
        scaleloss_details = []

        for pred in scale_preds:
            min_len = min(pred.size(1), y_true.size(1))
            truncated_pred = pred[:, :min_len]
            truncated_true = y_true[:, :min_len]
            scale_loss = self.mse(truncated_pred, truncated_true)
            scaleloss_details.append(scale_loss.item())
            global_loss += scale_loss

        global_loss /= num_scales
        local_loss = torch.tensor(0.0, device=y_true.device)
        weight_global = torch.tensor(1.0, device=y_true.device)
        total_loss = global_loss
        return total_loss, global_loss, local_loss, weight_global, scaleloss_details

    def _legacy_multiscale_loss(self, y_true, scale_preds):
        num_scales = len(scale_preds)
        global_loss = 0.0
        y_true_multiscale = self._multi_scale_decompose_targets(y_true, num_scales)

        scaleloss_details = []
        for scale_idx, pred in enumerate(scale_preds):
            downsampled_true = y_true_multiscale[scale_idx]
            min_len = min(pred.size(1), downsampled_true.size(1))
            truncated_pred = pred[:, :min_len]
            truncated_true = downsampled_true[:, :min_len]
            scale_loss = self.mse(truncated_pred, truncated_true)
            scaleloss_details.append(scale_loss.item())
            global_loss += scale_loss

        global_loss /= num_scales

        local_loss = 0.0
        valid_pairs = 0
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                high_res_pred = scale_preds[i]
                low_res_pred = scale_preds[j]
                downsampled_high_res = self._adjust_tensor_length(high_res_pred, low_res_pred.size(1))
                consistency_loss = self.mse(downsampled_high_res, low_res_pred)
                local_loss += consistency_loss
                valid_pairs += 1

        if valid_pairs > 0:
            local_loss /= valid_pairs
        else:
            local_loss = torch.tensor(0.0, device=y_true.device)

        eps = 1e-8
        weight_global = torch.sigmoid(2 * (global_loss / (global_loss + local_loss + eps) - 0.5))
        weight_local = 1 - weight_global
        total_loss = weight_global * global_loss + weight_local * local_loss

        return total_loss, global_loss, local_loss, weight_global, scaleloss_details

    def _use_full_resolution_global_only(self, y_true, scale_preds):
        target_len = y_true.size(1)
        return all(pred.size(1) == target_len for pred in scale_preds)

    def _multi_scale_decompose_targets(self, y_true, num_scales):
        y_true_list = [y_true]
        for _ in range(1, num_scales):
            downsampled = self._adjust_tensor_length(
                y_true_list[-1], y_true_list[-1].size(1) // 2
            )
            y_true_list.append(downsampled)
        return y_true_list

    def _adjust_tensor_length(self, tensor, target_length):
        _, src_len, channels = tensor.shape

        if src_len == target_length:
            return tensor

        tensor = tensor.transpose(1, 2)
        if src_len > target_length:
            if self.down_sampling_method == 'avg':
                adjusted = F.adaptive_avg_pool1d(tensor, target_length)
            elif self.down_sampling_method == 'max':
                adjusted = F.adaptive_max_pool1d(tensor, target_length)
            elif self.down_sampling_method == 'conv':
                padding = 1 if torch.__version__ >= '1.5.0' else 2
                stride = max(1, src_len // max(1, target_length))
                down_conv = nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    padding_mode='circular',
                    bias=False,
                ).to(tensor.device)
                adjusted = down_conv(tensor)
                if adjusted.size(2) != target_length:
                    adjusted = F.adaptive_avg_pool1d(adjusted, target_length)
            else:
                adjusted = F.interpolate(tensor, size=target_length, mode='linear', align_corners=False)
        else:
            adjusted = F.interpolate(tensor, size=target_length, mode='linear', align_corners=False)

        return adjusted.transpose(1, 2)
