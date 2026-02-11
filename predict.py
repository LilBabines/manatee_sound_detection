import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from src.models import get_model
from src.datasets import PredictAudioDataset

from hydra.utils import instantiate
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

import pandas as pd

@hydra.main(version_base="1.3", config_path="config/predict", config_name="dasheng")
def main(cfg: DictConfig) -> None:

    print("Initializing model and dataset...    ")

    pl.seed_everything(cfg.seed)
    
    model = get_model(cfg)
    model = model.__class__.load_from_checkpoint(cfg.model.checkpoint_path, cfg=cfg)

    model.eval()
    model.freeze()

    dataset = PredictAudioDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    trainer = instantiate(cfg.trainer)

    print("Running prediction...    ")
    

    predictions = trainer.predict(model=model, dataloaders=dataloader)

    # Aplatir: liste de dicts
    rows = [row for batch in predictions for row in batch]

    df = pd.DataFrame(rows)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    df.to_csv(run_dir/"predictions.csv", index=False)


if __name__ == "__main__":
    main()