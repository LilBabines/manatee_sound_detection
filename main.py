import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

import lightning.pytorch as pl
from src.models import get_model
from src.datasets import AudioDataModule, BenchMarkDataModule
from hydra.utils import instantiate
import torch

import pandas as pd
from src.utils import compare_csv

torch.set_float32_matmul_precision('medium')

@hydra.main(version_base="1.3", config_path="config", config_name="dasheng")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    model = get_model(cfg)
    data_module = AudioDataModule(cfg)
    trainer = instantiate(cfg.trainer)


    trainer.fit(model=model, datamodule=data_module)
    trainer.validate(model=model, datamodule=data_module, ckpt_path="best")


    # metrics = trainer.test(model=model, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)
    # probs, preds, targets = model.test_probs, model.test_preds, model.test_targets
    # probs_np   = probs.numpy()
    # preds_np   = preds.numpy()
    # targets_np = targets.numpy()
    # classes_name = ['pas_lamantin','lamantin']
    # # On construit un DataFrame
    # df = pd.DataFrame(probs_np, columns=[classes_name[i] for i in range(probs_np.shape[1])])
    # df["pred"]   = preds_np
    # df["target"] = targets_np
    
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # df.to_csv(run_dir/"val_predictions.csv", index=False)


    predictions = trainer.predict(model=model, datamodule=data_module, ckpt_path="best")

    # Aplatir: liste de dicts
    rows = [row for batch in predictions for row in batch]

    df = pd.DataFrame(rows)

    df.to_csv(run_dir/"test_predictions.csv", index=False)

if __name__ == "__main__":
    main()