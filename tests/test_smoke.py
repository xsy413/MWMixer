import unittest
from types import SimpleNamespace

import torch

from models.MWMixer import Model


class MWMixerSmokeTest(unittest.TestCase):
    def test_cpu_forward_shapes(self):
        args = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=16,
            label_len=0,
            pred_len=8,
            down_sampling_window=2,
            enc_in=3,
            e_layers=1,
            d_model=8,
            embed='timeF',
            freq='h',
            dropout=0.0,
            down_sampling_layers=1,
            use_norm=1,
            modwt_level=1,
            wavelet='haar',
            down_sampling_method='avg',
        )
        model = Model(args).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 16, 3), None, None, None)

        self.assertEqual(outputs['final'].shape, (2, 8, 3))
        self.assertEqual([x.shape for x in outputs['scale_preds']], [(2, 8, 3), (2, 4, 3)])


if __name__ == '__main__':
    unittest.main()
