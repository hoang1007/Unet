from typing import Optional
from omegaconf import DictConfig

from pytorch_lightning import LightningModule

from utils.config import instantiate_from_config


class BaseModel(LightningModule):
    def __init__(self, optimizer: DictConfig, scheduler: Optional[DictConfig] = None):
        super().__init__()
        self.optimizer = instantiate_from_config(optimizer)
        self.scheduler = instantiate_from_config(scheduler)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer
