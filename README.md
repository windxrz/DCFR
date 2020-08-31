# Derivable Conditional Fairness Regularizer
Source code for KDD 2020 paper [Algorithmic Decision Making with Conditional Fairness](https://arxiv.org/abs/2006.10483).

DCFR is an adversarial learning method to deal with fairness issues in supervised machine learning tasks. More details can be found in the paper.

## Installation
### Requirements
- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Quick Start
### Run for single fair coefficient and random seed
Run DCFR model on Adult income dataset and conditional fairness task with fair coefficient 20.
```bash
python main.py --model DCFR --task CF --dataset adult --seed 0 --fair-coeff 20
```
You can see more options from
```bash
python main.py -h
```
Result files will be saved in `results/`. Saved models will be saved in `saved/`. Tensorboard logs will be saved in `tensorboard/`.

### Run for multiple fair coefficients and random seeds
Run DCFR model on Adult income dataset and conditional fairness task.
```bash
bash scripts/dcfr.bash
```
Then type in `adult` and `CF`.
More bash files are in `scripts/`.

### Plot the accuracy-fairness trade-off curve
Plot the curve for existing models.
```bash
python plot.py
```
The results are shown in `results/`. `pareto.png` shows the pareto front while `scatter.png` shows the scatter diagram.


## Citing DCFR
If you find this repo useful for your research, please consider citing the paper.

```
@inproceedings{xu2020algorithmic,
  title={Algorithmic Decision Making with Conditional Fairness},
  author={Xu, Renzhe and Cui, Peng and Kuang, Kun and Li, Bo and Zhou, Linjun and Shen, Zheyan and Cui, Wei},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2125--2135},
  year={2020}
}
```

## Acknowledgements
Part of this code is inspired by David Madras et al.'s [LAFTR: Learning Adversarially Fair and Transferable Representations](https://github.com/VectorInstitute/laftr).
