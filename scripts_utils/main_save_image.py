import hydra
from omegaconf import DictConfig

import lightning.pytorch as pl
from src.models import get_model
from src.datasets import AudioDataModule
import torch


torch.set_float32_matmul_precision('medium')

@hydra.main(version_base="1.3", config_path="config/train", config_name="resnet")
def main(cfg: DictConfig) -> None:


    pl.seed_everything(cfg.seed)

    data_module = AudioDataModule(cfg)
    data_module.setup()
    
    dataset_train = data_module.train_dataset
    dataset_train.save_image(n_sample=10, out_dir="with_axes_png_gayscla")
    

if __name__ == "__main__":

    main()
