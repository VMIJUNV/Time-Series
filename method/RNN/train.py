import sys
sys.path.append(".")


import torch

from utils.dataset import TimeSeriesDataset
from utils.trainer import BaseTrainer, TrainerArguments
from model import RNN, Hparams

class Trainer(BaseTrainer):
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training_forward(self, batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)
        label=label[:, -1, :]
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, label)
        return loss, {"train_loss":loss.item()}
    
    def validation_forward(self, batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)
        label=label[:, -1, :]
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, label)
        return {"val_loss":loss.item()}

if __name__ == "__main__":
    model_config_path = "method/RNN/config/model.yaml"
    train_config_path = "method/RNN/config/trainer.yaml"
    data_path = "data/sse50/data_AR.pkl"
    
    train_dataset = TimeSeriesDataset(data_path, split="train")
    val_dataset = TimeSeriesDataset(data_path, split="val")

    hparams = Hparams.from_yaml(model_config_path)
    model = RNN(hparams)

    args = TrainerArguments.from_yaml(train_config_path)

    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args
        )
    trainer.train(resume_from_checkpoint=False)