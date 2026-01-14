import sys
sys.path.append(".")
from pathlib import Path
import torch
from model import TemporalFusionTransformer,Hparams
from utils.dataset import TimeSeriesDataset
from utils.trainer import BaseTrainer, TrainerArguments
from utils.evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):

    def test__forward(self,batch):
        x=batch["x"]
        label=batch["label"]
        x = x.to(self.device)
        label = label.to(self.device)
        label=label[:,self.model.num_encoder_steps:,:]

        pred = self.model(x)
        return pred, label

if __name__ == "__main__":
    output_path = "method/tft/output"
    model_config_path = "method/tft/config/model.yaml"
    data_path = "data/sse50/data.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = TimeSeriesDataset(data_path, split="test")

    output_path = Path(output_path)
    best_model_list = BaseTrainer.get_best_model_list(output_path / "best_model")
    best_model_path=best_model_list[-1]["folder"] / "model.pth"

    hparams = Hparams.from_yaml(model_config_path)
    model = TemporalFusionTransformer(hparams)
    model.load_state_dict(torch.load(best_model_path))

    evaluator = Evaluator(model, test_dataset, device)
    evaluator.eval()
