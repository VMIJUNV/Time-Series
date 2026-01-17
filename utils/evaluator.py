import numpy as np
import torch
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from .data_processing import DataProcessing
import matplotlib.pyplot as plt
from pathlib import Path


class BaseEvaluator(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def test_forward(self,batch):
        ...

    def eval(self,show_id=0):
        self.model = self.model.to(self.device)
        self.model.eval()
        pred_list = []
        label_list = []
        with torch.no_grad():
            with torch.no_grad():
                for idx in range(len(self.test_dataset_one_step)):
                    batch=self.test_dataset_one_step[idx]
                    pred,label = self.test_forward(batch)
                    pred_list.append(pred[:,0,:].cpu().numpy())
                    label_list.append(label[:,0,:].cpu().numpy())
        
        show_pred=pred_list[show_id]
        show_label=label_list[show_id]
        target_cols=self.test_dataset_one_step.target_cols
        if hasattr(self.test_dataset_one_step, "scaler"):
            show_pred = DataProcessing.inverse_scale(show_pred,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)
            show_label = DataProcessing.inverse_scale(show_label,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)
        self.show(show_pred,show_label,target_cols,f"one_step_sample_{show_id}.pdf")

        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        if hasattr(self.test_dataset_one_step, "scaler"):
            preds = DataProcessing.inverse_scale(preds,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)
            labels = DataProcessing.inverse_scale(labels,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)

        mse = np.mean((preds - labels) ** 2, axis=0)
        res={}
        
        for dim in range(len(target_cols)):
            res[target_cols[dim]]=float(mse[dim])
        print(f"one_step mse: {res}")


    def eval_multiple_steps(self,show_id=0):
        self.model = self.model.to(self.device)
        self.model.eval()

        input_steps=self.label_steps[1]-self.input_steps[0]
        skip_steps=self.label_steps[1]-self.label_steps[0]

        feature_cols=self.test_dataset_multiple_steps.feature_cols
        target_cols=self.test_dataset_multiple_steps.target_cols
        target_cols_idx=[feature_cols.index(col) for col in target_cols]

        pred_list=[]
        label_list=[]
        with torch.no_grad():
            for id in range(len(self.test_dataset_multiple_steps)):
                batch = self.test_dataset_multiple_steps[id]
                all_x=batch["x"].to(self.device).clone()
                all_label=batch["label"].to(self.device).clone()
                times=all_x.shape[0]
                steps=(times-input_steps) // skip_steps + 1

                pred_list_step=[]
                label_list_step=[]
                for step in range(steps):
                    start=step*skip_steps
                    end=start+input_steps
                    x = all_x[start:end]
                    label = all_label[start:end]
                    batch_={
                        "x":x.unsqueeze(0),
                        "label":label.unsqueeze(0)
                    }
                    pred,label = self.test_forward(batch_)
                    pred=pred.squeeze(0)
                    label=label.squeeze(0)
                    pred_list_step.append(pred.cpu().numpy())
                    label_list_step.append(label.cpu().numpy())
                    all_x[end-skip_steps:end,target_cols_idx]=pred

                pred=np.concatenate(pred_list_step, axis=0)
                label=np.concatenate(label_list_step, axis=0)
                pred_list.append(pred)
                label_list.append(label)


        show_pred=pred_list[show_id]
        show_label=label_list[show_id]
        if hasattr(self.test_dataset_multiple_steps, "scaler"):
            show_pred = DataProcessing.inverse_scale(show_pred,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)
            show_label = DataProcessing.inverse_scale(show_label,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)
        self.show(show_pred,show_label,target_cols,f"multiple_step_sample_{show_id}.pdf")
        
        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        if hasattr(self.test_dataset_multiple_steps, "scaler"):
            preds = DataProcessing.inverse_scale(preds,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)
            labels = DataProcessing.inverse_scale(labels,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)

        mse = np.mean((preds - labels) ** 2, axis=0)
        res={}
        for dim in range(len(target_cols)):
            res[target_cols[dim]]=float(mse[dim])
        print(f"multiple_steps mse: {res}")


    def show(self,pred,label,target_cols,name):
        res_path=Path(self.args["output_path"])
        res_path.mkdir(parents=True, exist_ok=True)
        dims=pred.shape[1]
        fig,axs=plt.subplots(dims,figsize=(12,2*dims))
        for dim in range(dims):
            pred_=pred[:,dim]
            label_=label[:,dim]
            x=np.arange(len(label_))
            axs[dim].plot(label_,label="label",color="green")
            axs[dim].plot(pred_,label="pred",color="blue")
            axs[dim].scatter(x, label_, s=20,marker="o",color="green")
            axs[dim].scatter(x, pred_, s=20,marker="*",color="blue")
            axs[dim].grid(True, linestyle='--', alpha=0.5)
            axs[dim].set_title(f"{target_cols[dim]}")
            axs[dim].legend()

        fig.tight_layout()
        fig.savefig( res_path/ name)