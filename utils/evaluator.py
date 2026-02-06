import numpy as np
import torch
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from .data_processing import DataProcessing
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json

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

        target_cols=self.test_dataset_one_step.target_cols

        preds=np.array(pred_list)
        labels=np.array(label_list)
        origin_preds=preds.copy()
        origin_labels=labels.copy()
        if hasattr(self.test_dataset_one_step, "scaler"):
            preds = DataProcessing.inverse_scale(preds,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)
            labels = DataProcessing.inverse_scale(labels,self.test_dataset_one_step.target_cols,self.test_dataset_one_step.scaler)

        self.save_data(preds,labels,origin_preds,origin_labels,target_cols,"one_step")
        
        show_pred=preds[show_id,...]
        show_label=labels[show_id,...]
        self.show(show_pred,show_label,target_cols,f"one_step_sample_{show_id}")

        self.statistics(preds,labels,target_cols,"one_step")


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

        preds=np.array(pred_list)
        labels=np.array(label_list)
        origin_preds=preds.copy()
        origin_labels=labels.copy()
        if hasattr(self.test_dataset_multiple_steps, "scaler"):
            preds = DataProcessing.inverse_scale(preds,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)
            labels = DataProcessing.inverse_scale(labels,self.test_dataset_multiple_steps.target_cols,self.test_dataset_multiple_steps.scaler)

        self.save_data(preds,labels,origin_preds,origin_labels,target_cols,"multiple_steps")

        show_pred=preds[show_id,...]
        show_label=labels[show_id,...]
        self.show(show_pred,show_label,target_cols,f"multiple_step_sample_{show_id}")

        self.statistics(preds,labels,target_cols,"multiple_steps")

    def save_data(self,preds,labels,origin_preds,origin_labels,target_cols,name):
        res_path=Path(self.args["output_path"])
        res_path.mkdir(parents=True, exist_ok=True)
        eval_data={
            "preds":preds,
            "labels":labels,
            "origin_preds":origin_preds,
            "origin_labels":origin_labels,
            "target_cols":target_cols,
        }
        with open(res_path/ f"{name}_data.pkl", "wb") as f:
            pickle.dump(eval_data, f)

    def statistics(self,preds,labels,target_cols,name):
        res_path=Path(self.args["output_path"])
        res_path.mkdir(parents=True, exist_ok=True)

        axis = (0, 1)
        # 1. MSE
        mse = np.mean((preds - labels) ** 2, axis=axis)

        # 2. RMSE
        rmse = np.sqrt(mse)

        # 3. MAE
        mae = np.mean(np.abs(preds - labels), axis=axis)

        # 4. MAPE
        eps = 1e-8  # 防止除零的小常数（也可用 mask 方式）
        mape = np.mean(
            np.abs((labels - preds) / np.where(np.abs(labels) < eps, eps, np.abs(labels))),
            axis=axis
        ) * 100  # 转换为百分比

        # 5. SMAPE
        smape = np.mean(
            2.0 * np.abs(preds - labels) / (np.abs(labels) + np.abs(preds) + eps),
            axis=axis
        ) * 100

        # 6. R² (coefficient of determination)
        ss_res = np.sum((labels - preds) ** 2, axis=axis)  # (F,)
        ss_tot = np.sum((labels - np.mean(labels, axis=axis, keepdims=True)) ** 2, axis=axis)
        r2 = np.where(ss_tot < eps, 1.0, 1 - ss_res / ss_tot)

        res = {
            "mse": {col: float(mse[dim]) for dim, col in enumerate(target_cols)},
            "rmse": {col: float(rmse[dim]) for dim, col in enumerate(target_cols)},
            "mae": {col: float(mae[dim]) for dim, col in enumerate(target_cols)},
            "mape (%)": {col: float(mape[dim]) for dim, col in enumerate(target_cols)},
            "smape (%)": {col: float(smape[dim]) for dim, col in enumerate(target_cols)},
            "r2": {col: float(r2[dim]) for dim, col in enumerate(target_cols)},
        }
        print(f"name: {name}")
        print(res)
        with open(res_path/ f"{name}_stats.json", "w") as f:
            json.dump(res, f, indent=4)

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
        fig.savefig( res_path/ f"{name}.pdf")