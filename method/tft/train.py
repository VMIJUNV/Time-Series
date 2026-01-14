import sys
sys.path.append(".")

import torch
from model import TemporalFusionTransformer,Hparams
from utils.dataset import TimeSeriesDataset
from utils.trainer import BaseTrainer, TrainerArguments

class Trainer(BaseTrainer):
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def loss(self, y_hat, y):
        return self.model.train_criterion.apply(y_hat, y)

    def training_forward(self, batch):
        x=batch["x"]
        y=batch["label"]
        x = x.to(self.device)
        y = y.to(self.device)
        y=y[:,self.model.num_encoder_steps:,:]

        y_hat = self.model(x)
        loss = self.loss(y_hat,y)
        return loss,{'train_loss': loss.item()}
    
    def validation_forward(self, batch):
        x=batch["x"]
        y=batch["label"]
        x = x.to(self.device)
        y = y.to(self.device)
        y=y[:,self.model.num_encoder_steps:,:]

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {'val_loss': loss.item()}

if __name__ == "__main__":
    model_config_path = "method/tft/config/model.yaml"
    train_config_path = "method/tft/config/trainer.yaml"
    data_path = "data/sse50/data.pkl"

    train_dataset = TimeSeriesDataset(data_path, split="train")
    val_dataset = TimeSeriesDataset(data_path, split="val")

    hparams=Hparams.from_yaml(model_config_path)
    model = TemporalFusionTransformer(hparams=hparams)

    args = TrainerArguments.from_yaml(train_config_path)

    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args
        )
    trainer.train(resume_from_checkpoint=False)