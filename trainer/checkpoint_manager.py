from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any,Literal,TYPE_CHECKING
from pathlib import Path
import torch
from .monitor import Monitor

if TYPE_CHECKING:
    from .trainer import BaseTrainer

class CheckpointManager:
    def __init__(self,
                path,
                monitor: str,
                monitor_mode: Literal["min","max","always"] = "min",
                interval_monitor: str = "step",
                interval: int = 1,
                max_checkpoints: int = 3,
                weights_only: bool = False):
    
        self.path = Path(path)
        self.monitor = Monitor(monitor)
        self.monitor_mode = monitor_mode
        self.interval_monitor = Monitor(interval_monitor,divisor=interval)
        self.max_checkpoints = max_checkpoints

        self.weights_only = weights_only
        
        self.read_old_monitor_value()

    def read_old_monitor_value(self):
        latest_checkpoint=CheckpointManager.get_latest_checkpoint(self.path)
        if latest_checkpoint is not None:
            state_path=latest_checkpoint["folder"] / "state.json"
            with open(state_path, "r") as f:
                state = json.load(f)
            self.monitor.update(state,mode="always")

    @staticmethod
    def get_checkpoint_list(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_folder_list = list(checkpoint_path.glob("checkpoint-*"))
        checkpoint_list = [{"index": int(x.stem.split("-")[1]), "folder": x} for x in checkpoint_folder_list]
        checkpoint_list.sort(key=lambda x: x["index"])
        return checkpoint_list

    @staticmethod
    def get_latest_checkpoint(checkpoint_path):
        checkpoint_list = CheckpointManager.get_checkpoint_list(checkpoint_path)
        if len(checkpoint_list) == 0:
            return None
        return checkpoint_list[-1]
    
    @staticmethod
    def get_latest_checkpoint_model_path(checkpoint_path):
        checkpoint = CheckpointManager.get_latest_checkpoint(checkpoint_path)
        if checkpoint is None:
            return None
        return checkpoint["folder"] / "model.pth"

    def save_checkpoint(self,trainer: BaseTrainer):
        checkpoint_list = CheckpointManager.get_checkpoint_list(self.path)

        excess = len(checkpoint_list) - (self.max_checkpoints - 1)
        if excess > 0:
            for checkpoint in checkpoint_list[:excess]:
                shutil.rmtree(checkpoint["folder"])
        
        if len(checkpoint_list) > 0:
            index = checkpoint_list[-1]["index"] + 1
        else:
            index = 0

        checkpoint_folder = self.path / f"checkpoint-{index}"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_model = trainer.model.state_dict()
        torch.save(checkpoint_model, checkpoint_folder / "model.pth")

        if not self.weights_only:
            checkpoint_optimizer = trainer.optimizer.state_dict()
            torch.save(checkpoint_optimizer, checkpoint_folder / "optimizer.pth")

        if trainer.scheduler is not None and not self.weights_only:
            checkpoint_scheduler = trainer.scheduler.state_dict()
            torch.save(checkpoint_scheduler, checkpoint_folder / "scheduler.pth")

        with open(checkpoint_folder / "state.json", "w") as f:
            json.dump(trainer.state, f, indent=4)
        

    @staticmethod
    def load_checkpoint(checkpoint_path: str, trainer: BaseTrainer):
        checkpoint = CheckpointManager.get_latest_checkpoint(checkpoint_path)
        if checkpoint is None:
            return
        
        checkpoint_folder = checkpoint["folder"]

        model_path = checkpoint_folder / "model.pth"
        if model_path.exists():
            checkpoint_model = torch.load(model_path)
            trainer.model.load_state_dict(checkpoint_model)

        optimizer_path = checkpoint_folder / "optimizer.pth"
        if optimizer_path.exists():
            checkpoint_optimizer = torch.load(optimizer_path)
            trainer.optimizer.load_state_dict(checkpoint_optimizer)

        scheduler_path = checkpoint_folder / "scheduler.pth"
        if scheduler_path.exists():
            checkpoint_scheduler = torch.load(scheduler_path)
            trainer.scheduler.load_state_dict(checkpoint_scheduler)

        state_path = checkpoint_folder / "state.json"
        with open(state_path, "r") as f:
            trainer.state = json.load(f)
        trainer.step = trainer.state["step"]

    def task(self,trainer: BaseTrainer):
        state = trainer.state

        is_change = self.interval_monitor.update(state,"unequal")
        is_divisible = self.interval_monitor.divisible(state)
        if not is_change or not is_divisible:
            return
        
        res=False
        if  self.monitor_mode == "always":
            res = self.monitor.update(state,"always")
        elif self.monitor_mode == "min":
            res = self.monitor.update(state,"min")
        elif self.monitor_mode == "max":
            res = self.monitor.update(state,"max")

        if res:
            self.save_checkpoint(trainer)
                

