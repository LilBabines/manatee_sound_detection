# 🐋 Manatee Sound Detection (Deep Learning)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.5-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-orange.svg)](https://hydra.cc/)

Implementation of a deep learning pipeline for detecting **manatee
vocalizations** from audio recordings, based on the paper:

> **[Few annotations, high accuracy: transfer learning and data augmentation improve passive acoustic monitoring of the vulnerable African manatee]** --- : Dubus et al., _in prep_


This repository provides: 
- model training : `train.py` 
- inference/prediction : `predict.py`

------------------------------------------------------------------------

## ⚙️ Installation (venv)

### 1) Clone the repo 

```bash
git clone https://github.com/LilBabines/manatee_sound_detection
cd manatee_sound_detection/
```

### 2) Create and activate a virtual environment (Python 3.13 used)

``` bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Download weights and data

```
Go to Zenodo repository : https://doi.org/10.5281/zenodo.20555275
And download data (training set and annotation, and testing set), runs (weights), test_set_annotation (test annotation), and if needed kaleidoscope_classification_trained_with_full_dataset
```
If necessary, modify the paths in YAML config files

------------------------------------------------------------------------

## 🚀 Usage

### 🔥 Train a model

To reproduce paper result :

``` bash
python train.py --config-name=resnet
pytohn train.py --config-name=dasheng   
```

All hyperparameters are configurable via the YAML files located in `cfg/train/`.

------------------------------------------------------------------------

### 🎧 Run inference / prediction

``` bash
python predict.py --config-name=dasheng \
    hydra.run.dir=runs/predict/dasheng_predict_custom \
    model.checkpoint_path=runs/dasheng/checkpoints/best.ckpt \
    data.pred_dir=data/test_set/extracts_1min
```

------------------------------------------------------------------------

## 📌 Citation

If you use this code, please cite the original paper:

``` bibtex
TODO
@article{paperkey,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```

------------------------------------------------------------------------

## 📜 License

This project is released under the **MIT License**.
