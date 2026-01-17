import sys
sys.path.append(".")

import torch
import yaml
from model import TemporalFusionTransformer,Hparams
from utils.dataset import TimeSeriesSeqDataset
from trainer.trainer import BaseTrainer, TrainerArguments

class Trainer(BaseTrainer):
    def __init__(self,args):
        super().__init__(args)
        
        self.input_steps = config["datasets"]["input_steps"]
        self.label_steps = config["datasets"]["label_steps"]

        hparams = Hparams.from_dict(config["model"])
        self.model = TemporalFusionTransformer(hparams)
        
        self.train_dataset = TimeSeriesSeqDataset(config["datasets"]["data_path"], split="train")
        self.val_dataset = TimeSeriesSeqDataset(config["datasets"]["data_path"], split="val")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_rate)


    def loss(self, y_hat, y):
        # return self.model.train_criterion.apply(y_hat, y)
        return torch.nn.functional.mse_loss(y_hat, y)

    def training_forward(self, batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)

        x=x[:, self.input_steps[0]:self.input_steps[1], :]
        label=label[:,self.label_steps[0]:self.label_steps[1], :]

        pred = self.model(x)
        loss = self.loss(pred, label)
        self.log("train_loss",loss.item())
        return loss
    
    def validation_forward(self, batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)

        x=x[:, self.input_steps[0]:self.input_steps[1], :]
        label=label[:, self.label_steps[0]:self.label_steps[1], :]
        
        pred = self.model(x)
        loss = self.loss(pred, label)
        self.log("val_loss",loss.item())

if __name__ == "__main__":
    config_path = "method/tft/config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    args = TrainerArguments.from_dict(config["trainer"])
    trainer = Trainer(args)
    trainer.train()