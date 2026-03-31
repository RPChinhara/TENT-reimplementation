# Tent: Fully Test-Time Adaptation by Entropy Minimization

This is a minimal reimplementation of the [Tent](https://openreview.net/forum?id=uXl3bZLkr3c) method for fully test-time adaptation by entropy minimization.

**Installation**:

```
pip install -r requirements.txt
```

Install Cifar 10C from this [link](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1) ; 
Unzip CIFAR-10-C.tar and place under data/

```bash
cd data/
tar -xvf CIFAR-10-C.tar
```

TENT depends on

- Python 3.9

and the example depends on

- [RobustBench](https://github.com/RobustBench/robustbench) v0.1 for the dataset and pre-trained model
- [yacs](https://github.com/rbgirshick/yacs) for experiment configuration

## Example: Adapting to Image Corruptions on CIFAR-10-C

The official repository contains examples that adapts a CIFAR-10 classifier to image corruptions on CIFAR-10-C.

This example compares a baseline without adaptation (source), test-time normalization for updating feature statistics during testing (norm), and our method for entropy minimization during testing (tent).
The dataset is [CIFAR-10-C](https://github.com/hendrycks/robustness/), with 15 types and 5 levels of corruption.

### WRN-28-10
The default model for [RobustBench](https://github.com/RobustBench/robustbench).

**Usage**:

```python
python cifar10c.py --cfg cfgs/source.yaml
python cifar10c.py --cfg cfgs/norm.yaml
python cifar10c.py --cfg cfgs/tent.yaml
python plots_tent.py
```

**Result**: TENT reduces the error (%) across corruption types at the most severe level of corruption (level 5).


## Comparison of Results with the Official Repository
|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source       | 43.5 |        72.3 |       65.7 |          72.9 |         46.9 |       54.3 |        34.8 |      42.0 | 25.1 |  41.3 | 26.0 |        9.3 |     46.7 |          26.6 |     58.5 | 30.3 |
| norm         | 20.4 |        28.1 |       26.1 |          36.3 |         12.8 |       35.3 |        14.2 |      12.1 | 17.3 |  17.4 | 15.3 |        8.4 |     12.6 |          23.8 |     19.7 | 27.3 |
| tent         | 18.6 |        24.8 |       23.5 |          33.0 |         12.0 |       31.8 |        13.7 |      10.8 | 15.9 |  16.2 | 13.7 |        7.9 |     12.1 |          22.0 |     17.3 | 24.2 |


## Results from My Reimplementation
|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source                                                     | 43.5 |        72.3 |       65.8 |          72.9 |         47.0 |       54.3 |        34.8 |      42.0 | 25.1 |  41.3 | 26.0 |        9.3 |     46.6 |          26.6 |     58.4 | 30.3 |
| norm                                                       | 20.4 |        28.1 |       26.1 |          36.3 |         12.8 |       35.3 |        14.2 |      12.1 | 17.3 |  17.4 | 15.2 |        8.4 |     12.7 |          23.8 |     19.6 | 27.3 |
| tent                                                       | 18.6 |        24.8 |       23.5 |          33.0 |         12.0 |       31.9 |        13.6 |      10.8 | 15.9 |  16.2 | 13.7 |        7.9 |     12.1 |          22.0 |     17.3 | 24.2 |