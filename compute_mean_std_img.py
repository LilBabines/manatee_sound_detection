import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

import lightning.pytorch as pl
from src.models import get_model
from src.datasets import AudioDataModule
from hydra.utils import instantiate
import torch

import pandas as pd
from src.utils import compare_csv
from tqdm import tqdm



#TODO : add metrics in config

@hydra.main(version_base="1.3", config_path="config", config_name="resnet")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')

    model = get_model(cfg)
    data_module = AudioDataModule(cfg)

    data_module.setup()
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset

    mean_normalization = []
    std_normalization = []

    for i in range(len(train_dataset)):
        audio, _ = train_dataset[i]

        mean_normalization.append(audio.mean(dim=[1,2]))
        std_normalization.append(audio.std(dim=[1,2]))

    for i in range(len(val_dataset)):
        audio, _ = val_dataset[i]
        mean_normalization.append(audio.mean(dim=[1,2]))
        std_normalization.append(audio.std(dim=[1,2]))

    mean = sum(mean_normalization) / len(mean_normalization)
    std = sum(std_normalization) / len(std_normalization)
    print(f"Mean: {mean}, Std: {std}")

    predict_dataset = data_module.predict_dataset
    mean_normalization = []
    std_normalization = []

    for i in tqdm(range(len(predict_dataset))):
        audio, _, _ = predict_dataset[i]

        mean_normalization.append(audio.mean().item())
        std_normalization.append(audio.std().item())

    mean = sum(mean_normalization) / len(mean_normalization)
    std = sum(std_normalization) / len(std_normalization)

    print(f"Predictict : Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    main()