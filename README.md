# MWMixer

Official PyTorch implementation of **Recovering Lost Details: Multi-Scale Frequency Compensation for Long-Term Time Series Forecasting** (CIKM 2026 submission cfp0285).

MWMixer combines multi-scale temporal inputs with MODWT frequency-band decomposition, bidirectional frequency-band mixing, and dynamic scale-adaptive fusion. The repository currently supports the long-term forecasting experiments reported in the paper.

## Repository layout

```text
MWMixer/
├── data_provider/       # Dataset loaders and train/validation/test splits
├── exp/                 # Training and evaluation loops
├── layers/              # Embeddings, normalization, and MODWT implementation
├── models/MWMixer.py    # MWMixer architecture
├── scripts/             # Reproduction commands
├── tests/               # Lightweight import and forward-pass tests
├── run.py               # Main CLI entry point
└── CIKM2026-cfp0285.pdf # Anonymous paper manuscript
```

## Installation

Python 3.10 or newer is recommended.

```bash
conda create -n mwmixer python=3.10 -y
conda activate mwmixer
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA runtime if GPU acceleration is required. CPU execution is supported through `--no-use_gpu`.

## Data

Download the standard ETT, Weather, Electricity, and Exchange benchmark CSV files and arrange them as follows:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/weather.csv
├── electricity/electricity.csv
└── exchange_rate/exchange_rate.csv
```

Each CSV must contain a `date` column. The ETT loaders use the standard fixed splits; the other long-term forecasting datasets use chronological 70%/10%/20% train/validation/test splits, matching the paper.

## Quick start

Train and evaluate ETTh1 with an input length of 96 and prediction length of 96:

```bash
python run.py \
  --is_training 1 \
  --model MWMixer \
  --model_id ETTh1_96_96 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7
```

The CLI defaults follow the common paper setup: Adam with learning rate `1e-4`, at most 30 epochs, patience 10, and three repeated runs. Dataset-specific hyperparameters selected during validation must be supplied explicitly. Use `python run.py --help` for the complete option list.

To reproduce a benchmark command for any reported horizon:

```bash
bash scripts/run_long_term.sh ETTh1 96
bash scripts/run_long_term.sh Weather 336
```

Supported dataset names are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `Electricity`, and `Exchange`. Supported horizons are 96, 192, 336, and 720.

Outputs are written to:

- `checkpoints/`: best validation checkpoints
- `results/`: metrics and predictions
- `test_results/`: qualitative forecast plots
- `result_long_term_forecast.txt`: aggregate test log

These generated paths are ignored by Git.

## Verification

Run the lightweight checks without downloading a dataset:

```bash
python -m unittest discover -s tests -v
python run.py --help
```

The test suite verifies that the public entry point imports and that MWMixer produces correctly shaped full-resolution and scale-wise forecasts on CPU.

## Reproducibility notes

- Random seeds are fixed to 2021 in `run.py`.
- The paper searches model dimension, number of downsampling layers, MODWT level, BFBM depth, wavelet, dropout, and multi-scale loss coefficient. The checked-in defaults are a valid baseline configuration, not a claim that one setting is optimal for every dataset and horizon.
- Reported paper numbers average three runs. Set `--itr 1` for a quick development run.
- Training data and checkpoints are intentionally excluded from the repository.

## Citation

The citation record will be updated after publication. Until then, please cite the anonymous manuscript by title:

```bibtex
@inproceedings{mwmixer2026,
  title     = {Recovering Lost Details: Multi-Scale Frequency Compensation for Long-Term Time Series Forecasting},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
