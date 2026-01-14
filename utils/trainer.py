from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import List, Dict, Any,Literal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch
import json
import shutil
from statistics import mean

from utils.more_utils import BaseDataclass

@dataclass
class TrainerArguments(BaseDataclass):
    output_path: str = None
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    train_epochs: int = 1
    train_steps: int = None

    micro_batch_size: int = 32
    gradient_accumulate_steps: int = 1

    val_strategy: Literal["no","epoch","step"] = "epoch"
    val_interval: int = 1
    val_batch_size: int = 32

    save_best_model: bool = False
    max_best_model: int = 3
    best_monitor_target: str = "val_loss"
    best_monitor_mode: Literal["min","max"] = "min"

    checkpoint_strategy: Literal["no","epoch","step"] = val_strategy
    checkpoint_interval: int = val_interval
    max_checkpoints: int = 3


class BaseTrainer(ABC):
    def __init__(self, 
                model: torch.nn.Module,
                train_dataset,
                val_dataset,
                args: TrainerArguments):
        
        self.args = args
        self.device = self.args.device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.output_path = Path(self.args.output_path)
        self.checkpoint_path = self.output_path / "checkpoints"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_path / "best_model"
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.output_path / "runs"
        self.runs_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.runs_path)

        self.step = 0
        self.skip_steps = 0
        self.log_data = {}

        self.best_monitor_target_old = None

        self.train_step_cache = []
        self.val_step_cache = []

        self.scheduler = None

        self.set_seed(self.args.seed)
        self.configure_dataloaders()
        self.configure_optimizers()

    @abstractmethod
    def configure_optimizers(self):
        ...

    @abstractmethod
    def training_forward(self, batch):
        ...

    @abstractmethod
    def validation_forward(self, batch):
        ...

    def configure_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.micro_batch_size, shuffle=True,drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.val_batch_size, shuffle=False)

    def log(self,key:str,value:Any):
        self.log_data.setdefault(key, []).append(value)
        self.writer.add_scalar(key, value,self.step)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_checkpoint_list(checkpoint_path: Path):
        checkpoint_folder_list = list(checkpoint_path.glob("checkpoint-*"))
        checkpoint_list = [{"step": int(x.stem.split("-")[1]), "folder": x} for x in checkpoint_folder_list]
        checkpoint_list.sort(key=lambda x: x["step"])
        return checkpoint_list

    def load_checkpoint(self):
        checkpoint_list = self.get_checkpoint_list(self.checkpoint_path)
        if len(checkpoint_list) >= 1:

            last_checkpoint_step = checkpoint_list[-1]["step"]
            checkpoint_folder = checkpoint_list[-1]["folder"]

            checkpoint_model = torch.load(checkpoint_folder / "model.pth")
            self.model.load_state_dict(checkpoint_model)

            checkpoint_optimizer = torch.load(checkpoint_folder / "optimizer.pth")
            self.optimizer.load_state_dict(checkpoint_optimizer)
            if self.scheduler is not None:
                checkpoint_scheduler = torch.load(checkpoint_folder / "scheduler.pth")
                self.scheduler.load_state_dict(checkpoint_scheduler)

            self.log_data = json.load(open(checkpoint_folder / "log.json", "r"))
            self.step = last_checkpoint_step

    def save_checkpoint(self):
        checkpoint_list = self.get_checkpoint_list(self.checkpoint_path)
        if len(checkpoint_list) >= self.args.max_checkpoints:
            shutil.rmtree(checkpoint_list[0]["folder"])

        checkpoint_folder = self.checkpoint_path / f"checkpoint-{self.step}"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_model = self.model.state_dict()
        torch.save(checkpoint_model, checkpoint_folder / "model.pth")

        checkpoint_optimizer = self.optimizer.state_dict()
        torch.save(checkpoint_optimizer, checkpoint_folder / "optimizer.pth")

        if self.scheduler is not None:
            checkpoint_scheduler = self.scheduler.state_dict()
            torch.save(checkpoint_scheduler, checkpoint_folder / "scheduler.pth")

        with open(checkpoint_folder / "log.json", "w") as f:
            json.dump(self.log_data, f, indent=4)

    def checkpoint_task(self):
        if self.checkpoint_interval_step is not None and self.step % self.checkpoint_interval_step == 0:
            self.save_checkpoint()

    @staticmethod
    def get_best_model_list(best_model_path: Path):
        best_model_folder_list = list(best_model_path.glob("best-*"))
        best_model_list = [{"index": int(x.stem.split("-")[1]), "folder": x} for x in best_model_folder_list]
        best_model_list.sort(key=lambda x: x["index"])
        return best_model_list

    def save_best_model(self):
        def save_model(best_monitor_target_new):
            best_model_list = self.get_best_model_list(self.best_model_path)
            if len(best_model_list) >= self.args.max_best_model:
                shutil.rmtree(best_model_list[0]["folder"])
            
            if len(best_model_list) == 0:
                index = 0
            else:
                index = best_model_list[-1]["index"] + 1

            best_model_folder = self.best_model_path / f"best-{index}"
            best_model_folder.mkdir(parents=True, exist_ok=True)

            best_model = self.model.state_dict()
            torch.save(best_model, best_model_folder / f"model.pth")

            log={
                "step": self.step,
                "monitor_target": self.args.best_monitor_target,
                "best_monitor_target": best_monitor_target_new,
            }
            with open(best_model_folder / "log.json", "w") as f:
                json.dump(log, f, indent=4)

        best_monitor_target_new = self.log_data[self.args.best_monitor_target][-1]

        if self.best_monitor_target_old is None:
            best_model_list = self.get_best_model_list(self.best_model_path)
            if len(best_model_list) == 0:
                self.best_monitor_target_old = best_monitor_target_new
                save_model(best_monitor_target_new)
                return
            else:
                with open(best_model_list[-1]["folder"] / "log.json", "r") as f:
                    log=json.load(f)
                self.best_monitor_target_old = log["best_monitor_target"]
        
        if self.args.best_monitor_mode == "min":
            if best_monitor_target_new < self.best_monitor_target_old:
                self.best_monitor_target_old = best_monitor_target_new
                save_model(best_monitor_target_new)
        elif self.args.best_monitor_mode == "max":
            if best_monitor_target_new > self.best_monitor_target_old:
                self.best_monitor_target_old = best_monitor_target_new
                save_model(best_monitor_target_new)

    def validation_epoch_before(self):
        self.val_step_cache = []

    def validation_epoch_after(self):
        keys = self.val_step_cache[0].keys()
        temp={}
        for key in keys:
            key_mean = mean([x[key] for x in self.val_step_cache])
            temp[key]=key_mean
            self.log(key,key_mean)

        print(f"Validation success. Res {temp}")

    def validation_loop(self):
        self.validation_epoch_before()
        with torch.no_grad():
            for batch in self.val_loader:
                val_res = self.validation_forward(batch)
                self.val_step_cache.append(val_res)
        self.validation_epoch_after()


    def validation_task(self):
        if self.val_interval_step is not None and self.step % self.val_interval_step == 0:
            self.model.eval()
            self.validation_loop()

            if self.args.save_best_model:
                self.save_best_model()
            self.model.train()

    def training_step_before(self):
        self.train_step_cache = []
        self.optimizer.zero_grad()

    def accumulate_step(self,batch):
        loss,tr_res = self.training_forward(batch)
        loss /= self.args.gradient_accumulate_steps
        loss.backward()
        self.train_step_cache.append(tr_res)

    def training_step_optimize(self):
        self.optimizer.step()
        
    def training_step_after(self):
        self.step += 1

        keys = self.train_step_cache[0].keys()
        for key in keys:
            key_mean = mean([x[key] for x in self.train_step_cache])
            self.log(key,key_mean)

        self.pbar.n = self.step
        self.pbar.refresh()
        
        self.validation_task()
        self.checkpoint_task()

    def train_epoch_before(self):
        ...
    
    def train_epoch_after(self):
        epoch = self.step // self.per_epoch_steps
        self.log("epoch",epoch)

    def train_loop(self):
        self.model.train()
        while self.step < self.train_steps:
            self.train_epoch_before()
            
            data_iter = iter(self.train_loader)
            for accumulate_steps in self.per_epoch_accumulate_steps_list:
                if self.skip_steps > 0:
                    for _ in range(accumulate_steps):
                        batch = next(data_iter)
                    self.skip_steps -= 1
                    continue

                self.training_step_before()
                for _ in range(accumulate_steps):
                    batch = next(data_iter)
                    self.accumulate_step(batch)
                self.training_step_optimize()
                self.training_step_after()

                if self.step >= self.train_steps:
                    break

            self.train_epoch_after()

    def train(self,resume_from_checkpoint = False):
        self.global_batch_size = self.args.gradient_accumulate_steps * self.args.micro_batch_size

        train_loader_len = len(self.train_loader)
        self.per_epoch_accumulate_steps_list = [self.args.gradient_accumulate_steps] * (train_loader_len // self.args.gradient_accumulate_steps)

        if train_loader_len % self.args.gradient_accumulate_steps != 0:
            self.per_epoch_accumulate_steps_list += [train_loader_len % self.args.gradient_accumulate_steps]

        self.per_epoch_steps=len(self.per_epoch_accumulate_steps_list)

        self.train_steps = self.per_epoch_steps * self.args.train_epochs
        if self.args.train_steps:
            self.train_steps = self.args.train_steps

        if self.args.val_strategy == "step":
            self.val_interval_step = self.args.val_interval
        elif self.args.val_strategy == "epoch":
            self.val_interval_step = self.per_epoch_steps * self.args.val_interval
        else:
            self.val_interval_step = None

        if self.args.checkpoint_strategy  == "step":
            self.checkpoint_interval_step = self.args.checkpoint_interval
        elif self.args.checkpoint_strategy  == "epoch":
            self.checkpoint_interval_step = self.per_epoch_steps * self.args.checkpoint_interval
        else:
            self.checkpoint_interval_step = None

        self.pbar = tqdm(total=self.train_steps, desc="Training", unit="step")
        self.step = 0

        if resume_from_checkpoint:
            self.load_checkpoint()
            self.skip_steps = self.step // self.per_epoch_steps

        self.train_loop()
        self.writer.close()




