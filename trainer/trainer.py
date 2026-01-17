from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any,Literal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch

from statistics import mean
import numpy as np
import random

from .checkpoint_manager import CheckpointManager
from .monitor import Monitor


class BaseDataclass:
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))

@dataclass
class TrainerArguments(BaseDataclass):
    output_path: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 888

    train_epochs: int = 1
    train_steps: int = None

    lr_rate: float = 0.001
    micro_batch_size: int = 32
    gradient_accumulate_steps: int = 1
    gradient_clip_val: float = None
    gradient_clip_algorithm: Literal["norm","value"] = "norm"

    val_interval_monitor: str = "epoch"
    val_interval: int = 1
    val_batch_size: int = 1

    callback: List[Dict[str,Any]] = field(default_factory=list)


class BaseTrainer(ABC):
    def __init__(self, args: TrainerArguments):
        
        self.args = args
        self.device = self.args.device
        self.output_path = Path(self.args.output_path)
        self.best_model_path = self.output_path / "best_model"
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.output_path / "runs"
        self.runs_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.runs_path)

        self.step = 0
        self.skip_steps = 0

        self.state = {}
        self.log_cache = {}

        self.val_monitor = Monitor(self.args.val_interval_monitor,self.args.val_interval)

        self.callback_list = []
        if self.args.callback is not None:
            for callback_config in self.args.callback:
                callback_type = callback_config.pop("type")
                if callback_type == "CheckpointManager":
                    callback_config["path"] = self.output_path / callback_config["path"]
                    checkpoint_manager = CheckpointManager(**callback_config)
                self.callback_list.append(checkpoint_manager.task)

        self.scheduler = None
        BaseTrainer.set_seed(self.args.seed)

    @abstractmethod
    def training_forward(self, batch):
        ...

    @abstractmethod
    def validation_forward(self, batch):
        ...

    def configure_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(torch.initial_seed())

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        self.train_loader = DataLoader(self.train_dataset,
                                        batch_size=self.args.micro_batch_size,
                                        shuffle=True,
                                        generator=g,
                                        worker_init_fn=seed_worker,
                                        drop_last=True)
        self.val_loader = DataLoader(self.val_dataset,
                                        batch_size=self.args.val_batch_size,
                                        shuffle=False,
                                        worker_init_fn=seed_worker,
                                        generator=g)

    def log(self,key:str,value:Any):
        self.log_cache.setdefault(key, []).append(value)

    def log_task(self):
        keys = self.log_cache.keys()
        temp={}
        for key in keys:
            value = mean(self.log_cache[key])
            temp[key]=value

            self.state[key] = value
            self.writer.add_scalar(key, value,self.step)

        self.log_cache={}
        ...

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def validation_before(self):
        ...

    def validation_loop(self):
        with torch.no_grad():
            for batch in self.val_loader:
                self.validation_forward(batch)

    def validation_after(self):
        old_val_step = self.state.get("val_step",0)
        self.log("val_step",old_val_step + 1)

        keys = self.log_cache.keys()
        temp={}
        for key in keys:
            value = mean(self.log_cache[key])
            temp[key]=value

        print(f"Validation set results: {temp}")

    def validation_task(self):
        is_change = self.val_monitor.update(self.state,"max")
        is_divisible = self.val_monitor.divisible(self.state)
        if is_change and is_divisible:
            self.model.eval()
            self.validation_before()
            self.validation_loop()
            self.validation_after()
            self.model.train()

    def training_step_before(self):
        self.optimizer.zero_grad()

    def training_accumulate_step(self,batch):
        loss = self.training_forward(batch)
        loss /= self.args.gradient_accumulate_steps
        loss.backward()

    def training_step_optimize(self):
        if self.args.gradient_clip_val is not None:
            if self.args.gradient_clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_val)
            elif self.args.gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_val)
        self.optimizer.step()
        
    def training_step_after(self):
        self.step += 1
        self.state["step"] = self.step

        epoch = self.step // self.per_epoch_steps
        self.state["epoch"] = epoch
        self.log("epoch",epoch)

        self.pbar.n = self.step
        self.pbar.refresh()
        
        self.validation_task()
        self.log_task()
        for callback in self.callback_list:
            callback(self)

    def train_loop(self):
        while self.step < self.train_steps:
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
                    self.training_accumulate_step(batch)
                self.training_step_optimize()
                self.training_step_after()

                if self.step >= self.train_steps:
                    break

    def train_before(self,checkpoint_path: str):
        self.configure_dataloaders()

        train_loader_len = len(self.train_loader)
        self.per_epoch_accumulate_steps_list = [self.args.gradient_accumulate_steps] * (train_loader_len // self.args.gradient_accumulate_steps)

        if train_loader_len % self.args.gradient_accumulate_steps != 0:
            self.per_epoch_accumulate_steps_list += [train_loader_len % self.args.gradient_accumulate_steps]

        self.per_epoch_steps=len(self.per_epoch_accumulate_steps_list)

        self.train_steps = self.per_epoch_steps * self.args.train_epochs
        if self.args.train_steps:
            self.train_steps = self.args.train_steps

        self.pbar = tqdm(total=self.train_steps, desc="Training", unit="step")
        self.step = 0

        if checkpoint_path is not None:
            CheckpointManager.load_checkpoint(checkpoint_path,self)
            self.skip_steps = self.step % self.per_epoch_steps

        self.model.to(self.device)
        self.model.train()
        

    def train_after(self):
        self.writer.close()

    def train(self,checkpoint_path: str = None):
        self.train_before(checkpoint_path)
        self.train_loop()
        self.train_after()



