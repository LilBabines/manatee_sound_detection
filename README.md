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
```
Download data (training set and annotation, and testing set), runs (weights), test_set_annotation (test annotation), and if needed kaleidoscope_classification_trained_with_full_dataset
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
## 📊 Model evaluation on the test set

After running inference with `predict.py`, a `predictions.csv` file is generated.  
To evaluate the model performance on the independent test set, run the following scripts in order:

1. `evaluation_prediction_step1.ipynb`
2. `evaluation_prediction_step2.R`

These scripts require the test set annotations available on Zenodo: `test_set_annotation.zip`

Make sure the following files are available and correctly linked in the scripts:

- `test_set_all_deep.csv`
- `predictions.csv`
- `predictions_results.csv`

### Kaleidoscope evaluation
To reproduce the results obtained with Kaleidoscope in Dubus et al. (in prep), download the advanced classifier output from the Zenodo repository : kaleidoscope_classification_trained_with_full_dataset.zip
Then run the following script: 'evaluation_prediction_kaleidoscope.R' 
Before running the script, update the file paths if necessary so that they correctly point to:
cluster.csv
predictions_results.csv

------------------------------------------------------------------------

## 📌 Citation

If you use this code, please cite the original paper:

``` bibtex
@article{dubus_inprep_manatee,
  title={Few annotations, high accuracy: transfer learning and data augmentation improve passive acoustic monitoring of the vulnerable African manatee},
  author={Dubus, Lucas and Verdier, Auguste and Giotto, Nina and Mbemba, Grace and Michelin, Gabriel and Mulot, Baptiste and Manel, Stéphanie and Mouillot, David},
  journal={Remote Sensing in Ecology and Conservation},
  year={in prep}
}
```

------------------------------------------------------------------------

## 📜 License

This project is released under the **MIT License**.

This license applies to the source code in this GitHub repository. You are free to use, modify, and redistribute the code, provided that the original copyright notice and license are included.

If you use this code, the associated models, or the datasets in academic work, publications, reports, or derivative research, you are required to cite the associated paper and the Zenodo dataset.

The annotated training and test datasets associated with this project are available on Zenodo and are distributed under the license specified there : https://doi.org/10.5281/zenodo.20555275

The code is provided without warranty.
