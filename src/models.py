from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn
import torch
import torchvision.models as models
from hydra.utils import instantiate

from typing import Mapping, Any
import dasheng


class BaseModule(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.classes_name = cfg.data.classes

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False,logger=True)
        
        # Log accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        # Log accuracy
        # print(y_hat)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, on_epoch=True, prog_bar=False, logger=True)

        return loss
    
    def on_test_start(self):
        self._test_batches = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test_loss", loss, prog_bar=False, logger=True)

        probs = logits
        # probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=False, logger=True)

        self._test_batches.append({
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
        })
        return loss

    def on_test_epoch_end(self):
        if self._test_batches:
            self.test_probs   = torch.cat([b["probs"] for b in self._test_batches])
            self.test_preds   = torch.cat([b["preds"] for b in self._test_batches])
            self.test_targets = torch.cat([b["targets"] for b in self._test_batches])
        self._test_batches.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        x, filename, start_time = batch
        # x,y = batch
        probs = self.forward(x) # already sigmoid
        # print(probs)
        preds = torch.argmax(probs, dim=1)

        return [{
            'filename': filename[i],
            'start_time': start_time[i].item(),
            'label_preds': self.classes_name[preds[i].item()],
            'id_preds': preds[i].item(),
            'probs_'+self.classes_name[0]: probs[i, 0].item(),
            'probs_'+self.classes_name[1]: probs[i, 1].item()
        } for i in range(x.shape[0])]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.cfg.optimizer.params)
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        } 


class AudioResnet50(BaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if cfg.model.pre_trained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, cfg.model.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(x).sigmoid()

class Dasheng(BaseModule):
    def __init__(self, cfg: DictConfig,):
        super().__init__(cfg)
        if cfg.model.kwargs is None:
            
                kwargs = {}
        else :
            
            kwargs = cfg.model.kwargs
            
        if cfg.model.type =="base" :

            if cfg.model.pre_trained :
            
                self.dashengmodel = dasheng.dasheng_base(**kwargs)

            else :
                kwargs["embed_dim"] = 768
                kwargs["depth"] = 12
                kwargs["num_heads"] = 12
                self.dashengmodel = dasheng.pretrained.pretrained.Dasheng(**kwargs)
        else :
            raise ValueError(f"Unknown Dasheng model type: {cfg.model.type}, only 'base' is supported currently.")

        self.classifier = torch.nn.Sequential(torch.nn.LayerNorm(self.dashengmodel.embed_dim), torch.nn.Linear(self.dashengmodel.embed_dim, 2))

 
    def load_state_dict(self, state_dict: Mapping[str, Any], assign: bool = False, strict=True, from_dasheng = True):
       
        stripped_state_dict = {k.replace('dashengmodel.', ''): v for k, v in state_dict.items() if k.startswith('dashengmodel.')}

        
        self.dashengmodel.load_state_dict(stripped_state_dict, strict=strict)

        if not from_dasheng :
            
            # Prepare classifier weights (if any)
            for_classifier_dict = {}
            for k, v in state_dict.items():
                if 'outputlayer' in k or 'classifier' in k:
                    for_classifier_dict[k.replace('outputlayer.', '').replace('classifier.', '')] = v
            self.classifier.load_state_dict(for_classifier_dict, strict=False)

        return self

    def forward(self, x):
        x = self.dashengmodel(x).mean(1)
        return self.classifier(x).sigmoid()

def get_model(cfg: DictConfig) :
    """Factory function to get the model based on the configuration.
    Args:
        cfg (DictConfig): Configuration object containing model parameters.
    Returns:
        LightningModule: An instance of the model class.
    Raises:
        ValueError: If the model name in the configuration is not recognized.
    """

    if cfg.model.name == "AudioResnet50":
        return AudioResnet50(cfg)
    elif cfg.model.name == "Dasheng":
        return Dasheng(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")