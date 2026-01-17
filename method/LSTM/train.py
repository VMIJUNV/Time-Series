import sys
sys.path.append(".")
import yaml
import torch

from utils.dataset import TimeSeriesSeqDataset
from trainer.trainer import BaseTrainer, TrainerArguments
from model import LSTM, Hparams

class Trainer(BaseTrainer):
    def __init__(self,args):
        super().__init__(args)

        self.input_steps = config["datasets"]["input_steps"]
        self.label_steps = config["datasets"]["label_steps"]

        # 定义模型
        hparams = Hparams.from_dict(config["model"])
        self.model = LSTM(hparams)
        
        # 加载数据集(训练集和验证集)
        self.train_dataset = TimeSeriesSeqDataset(config["datasets"]["data_path"], split="train")
        self.val_dataset = TimeSeriesSeqDataset(config["datasets"]["data_path"], split="val")

        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_rate)

    # 训练前向传播
    def training_forward(self, batch):
        # x:(batch_size,windows_size,intput_feature_size)
        # label:(batch_size,windows_size,output_feature_size)
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)

        # x范围取input_steps，这里是除最后一个时间步的前面所有时间步
        # label范围取label_steps，这里是最后一个时间步
        x=x[:, self.input_steps[0]:self.input_steps[1], :]
        label=label[:,self.label_steps[0]:self.label_steps[1], :]

        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, label)
        self.log("train_loss",loss.item())
        return loss
    
    # 验证前向传播
    def validation_forward(self, batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)

        x=x[:, self.input_steps[0]:self.input_steps[1], :]
        label=label[:, self.label_steps[0]:self.label_steps[1], :]
        
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, label)
        self.log("val_loss",loss.item())

if __name__ == "__main__":
    # 加载配置文件
    config_path = "method/LSTM/config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    args = TrainerArguments.from_dict(config["trainer"])
    trainer = Trainer(args)
    trainer.train()