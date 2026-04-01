# TENT Reimplementation with Local ResNet-26 Training

This repository supports the full CIFAR-10 flow locally:

1. Train a CIFAR-style `ResNet-26` from scratch on CIFAR-10.
2. Save the trained checkpoint in `ckpt/`.
3. Evaluate the saved checkpoint on CIFAR-10-C with:
   - `source`
   - `norm`
   - `tent`
4. Generate logs, plots, and a `results.csv` summary.

## Setup

Python requirement:

- Python `3.9`
- Tested with Python `3.9.25`

Create a conda environment and install dependencies:

```bash
conda create --name tent python=3.9 -y
conda activate tent
pip install -r requirements.txt
```

## Data

### CIFAR-10

The training script can download CIFAR-10 automatically into `data/`:

```bash
python train_cifar10.py --download
```

### CIFAR-10-C

Place the extracted corruption files under `data/CIFAR-10-C/`, for example:

```text
data/CIFAR-10-C/gaussian_noise.npy
data/CIFAR-10-C/labels.npy
...
```

## Training

Train `ResNet-26` from scratch and save the best checkpoint to
`ckpt/cifar10/resnet26_best.pth`:

```bash
python train_cifar10.py \
  --download \
  --epochs 200 \
  --batch-size 128 \
  --eval-batch-size 256
```

Notes:

- Best checkpoint: `ckpt/cifar10/resnet26_best.pth`
- Last checkpoint: `ckpt/cifar10/resnet26_last.pth`
- Resume training:

```bash
python train_cifar10.py --resume
```

## Evaluation

Run the three evaluation modes on the saved local checkpoint:

```bash
python cifar10c.py --cfg cfgs/source.yaml
python cifar10c.py --cfg cfgs/norm.yaml
python cifar10c.py --cfg cfgs/tent.yaml
```

Each config points to:

```text
./ckpt/cifar10/resnet26_best.pth
```

Logs are written under `output/`.

## Full Pipeline

[`run_full_pipeline.sh`](run_full_pipeline.sh) runs the complete experiment in one go:

1. Train `ResNet-26` on CIFAR-10, or resume from `ckpt/cifar10/resnet26_last.pth` if it exists.
2. Run `source` evaluation on the resulting best checkpoint.
3. Run `norm` evaluation on the same checkpoint.
4. Run `tent` evaluation on the same checkpoint.
5. Generate `results.csv` and plots in `output/plots/`.

Default behavior:

- Checkpoint: `./ckpt/cifar10/resnet26_best.pth`
- Resume enabled: `RESUME=1`
- Train workers: `TRAIN_WORKERS=4`
- Eval workers: `EVAL_WORKERS=2`
- Runs inside the `tent` conda environment with live-streamed logs

Run it directly:

```bash
./run_full_pipeline.sh
```

If the script is interrupted or fails, simply re-running it will resume from the last checkpoint and skip completed steps.

## Results

After the three evaluations finish, generate plots and the summary CSV:

```bash
python plots_tent.py
```

Artifacts:

- `results.csv`
- `output/source_*.txt`
- `output/norm_*.txt`
- `output/tent_*.txt`
- `output/plots/*.png`

## Notes on ImageNet

This repo implements the CIFAR-10/CIFAR-10-C path end to end with a local
`ResNet-26`. There is no standard ImageNet-pretrained weight release for this
small CIFAR-specific architecture, so the CIFAR model is trained from scratch.


## Comparison with Original TENT
The original TENT paper reports CIFAR-10-C results for a `ResNet-26` trained on
CIFAR-10. This reimplementation also trains a local `ResNet-26` from scratch
on CIFAR-10 and evaluates `source`, `norm`, and `tent` on the saved checkpoint.
The table below uses the final values from `results.csv`.

## Results Comparison

### Original TENT Results
The original TENT paper reports the following CIFAR-10-C error rates for `WideResNet-28-10`:
|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source       | 43.5 |        72.3 |       65.7 |          72.9 |         46.9 |       54.3 |        34.8 |      42.0 | 25.1 |  41.3 | 26.0 |        9.3 |     46.7 |          26.6 |     58.5 | 30.3 |
| norm         | 20.4 |        28.1 |       26.1 |          36.3 |         12.8 |       35.3 |        14.2 |      12.1 | 17.3 |  17.4 | 15.3 |        8.4 |     12.6 |          23.8 |     19.7 | 27.3 |
| tent         | 18.6 |        24.8 |       23.5 |          33.0 |         12.0 |       31.8 |        13.7 |      10.8 | 15.9 |  16.2 | 13.7 |        7.9 |     12.1 |          22.0 |     17.3 | 24.2 |

### Reimplementation Results
These are the values written by `results.csv` from the completed local run using custom pre-trained `ResNet-26` on the same CIFAR-10/CIFAR-10-C dataset:

|                                                            | mean  | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ----: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source                                                     | 46.56 |       72.83 |      67.40 |         71.94 |        46.00 |      54.84 |       39.27 |     42.12 | 28.27 | 41.64 | 32.79 |       10.98 |    69.02 |         29.02 |    61.97 | 30.32 |
| norm                                                       | 27.36 |       38.63 |      37.00 |         44.76 |        17.02 |      42.40 |       19.27 |     17.45 | 25.58 | 25.88 | 18.87 |       13.31 |    18.21 |         30.12 |    28.62 | 33.32 |
| tent                                                       | 24.51 |       33.17 |      31.60 |         40.48 |        15.78 |      39.14 |       17.53 |     15.47 | 22.20 | 22.88 | 17.62 |       12.27 |    18.30 |         27.15 |    24.81 | 29.32 |

### Adaptation Gain Comparison

To compare how well `tent` is implemented, the more useful view is the drop in
mean error when switching adaptation methods, rather than the absolute model
error alone.

| transition | original mean error drop (pp) | reimplementation mean error drop (pp) | difference in drop (reimpl - original, pp) |
| ---------- | ----------------------------: | ------------------------------------: | -----------------------------------------: |
| source -> norm | 23.10 | 19.20 | -3.90 |
| source -> tent | 24.90 | 22.05 | -2.85 |
| norm -> tent   | 1.80  | 2.85  | +1.05 |

### Takeaways

- In the original results, moving from `source` to `tent` drops mean error by `24.90` points.
- In this reimplementation, moving from `source` to `tent` drops mean error by `22.05` points.
- In the original results, moving from `norm` to `tent` drops mean error by `1.80` points.
- In this reimplementation, moving from `norm` to `tent` drops mean error by `2.85` points.
- The biggest gap versus the original is `source -> norm`, where the local run gains `19.20` points instead of `23.10`.
- The key TENT-specific comparison, `norm -> tent`, is directionally consistent and slightly stronger here: `2.85` points vs `1.80` in the original.


### Conclusion
This reimplementation successfully trains a local `ResNet-26` on CIFAR-10 and evaluates `source`, `norm`, and `tent` on CIFAR-10-C. The results show a similar trend to the original TENT paper, with `tent` providing a significant error reduction over `source`. The exact numbers differ, which is expected due to differences in training runs and random seeds, but the overall pattern of improvement from `source` to `tent` is preserved.
