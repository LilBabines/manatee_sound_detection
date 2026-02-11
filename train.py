import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from src.models import get_model
from src.datasets import AudioDataModule
from hydra.utils import instantiate
import torch
torch.set_float32_matmul_precision('medium')

@hydra.main(version_base="1.3", config_path="config/train", config_name="resnet")
def main(cfg: DictConfig) -> None:


    pl.seed_everything(cfg.seed)

    model = get_model(cfg)
    data_module = AudioDataModule(cfg)
    trainer = instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=data_module)
    trainer.validate(model=model, datamodule=data_module, ckpt_path="best")

if __name__ == "__main__":

    main()
