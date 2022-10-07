## meta-XB

This repository contains code for "[Few-Shot Calibration of Set Predictors via Meta-Learned Cross-Validation-Based Conformal Prediction](https://arxiv.org/abs/2210.03067)" - 
Sangwoo Park, Kfir M. Cohen, and Osvaldo Simeone.

### Dependencies

This program is written in python 3.9 and uses PyTorch 1.10.2.

### Essential Codes
- meta-XB can be found at `meta_train/meta_training.py`
- meta-VB can be found at `meta_train/meta_tr_benchmark.py`
- XB-CP can be found at `funcs/jk_plus.py` (for number of folds = number of examples) and `funcs/cv_plus.py` (for number of folds <= number of examples)
- VB-CP can be found at `funcs/split_conformal.py`
- soft inefficiency function is written in `funcs/utils_for_set_prediction.py`
- soft quantile via pinball loss (proposed way) and also via optimal transport (OT) can be found at `funcs/soft_quantile.py`
- further details can be found at the beginning of the main code `main.py`

### Experiment on Multinomial Model and Inhomogeneous Features (Sec. V-A)
- `runs_toy` directory contains all the running shell script files
### Experiment on Modulation Classification (Sec. V-B)
- `runs_modulation_classification` directory contains all the running shell script files
### Experiment on miniImagenet Classification (Sec. V-C)
- `runs_miniimagenet` directory contains all the running shell script files
### Experiment on Demodulation for Golden Angle Modulation (GAM) (Appendix C)
- `runs_toy_vis_gam` directory contains all the running shell script files
