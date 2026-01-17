import sys
sys.path.append(".")
from pathlib import Path
import torch
import yaml
from model import RNN, Hparams
from utils.dataset import TimeSeriesSeqIdDataset,TimeSeriesDataset
from trainer import CheckpointManager
from utils.evaluator import BaseEvaluator

class Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.args = config["eval"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_steps = config["datasets"]["input_steps"]
        self.label_steps = config["datasets"]["label_steps"]

        self.test_dataset_one_step = TimeSeriesSeqIdDataset(config["datasets"]["data_path"], split="test",batch_dim="id")
        self.test_dataset_multiple_steps = TimeSeriesDataset(config["datasets"]["data_path"], split="test")

        hparams = Hparams.from_dict(config["model"])
        self.model = RNN(hparams)
        checkpoint_model_path = CheckpointManager.get_latest_checkpoint_model_path(config["eval"]["checkpoint_path"])
        self.model.load_state_dict(torch.load(checkpoint_model_path))

    def test_forward(self,batch):
        x=batch["x"].to(self.device)
        label=batch["label"].to(self.device)

        x=x[:, self.input_steps[0]:self.input_steps[1], :]
        label=label[:, self.label_steps[0]:self.label_steps[1], :]

        pred = self.model(x)
        return pred, label

if __name__ == "__main__":
    config_path = "method/RNN/config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator()
    evaluator.eval(0)
    evaluator.eval_multiple_steps(0)

