import numpy as np
import torch
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from .data_processing import DataProcessing



class BaseEvaluator(ABC):
    def __init__(self,model,dataset,device):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        self.test_dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        self.scalerY = self.dataset.scalerY

    @abstractmethod
    def test__forward(self,batch):
        ...

    def eval(self):
        self.model.eval()
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                pred,label = self.test__forward(batch)
                pred_list.append(pred.cpu().numpy())
                label_list.append(label.cpu().numpy())
        
        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        preds_denorm = DataProcessing.inverse_scale(preds, self.scalerY)
        labels_denorm = DataProcessing.inverse_scale(labels, self.scalerY)

        mse = np.mean((preds_denorm - labels_denorm) ** 2)
        print(f"mse: {mse}")