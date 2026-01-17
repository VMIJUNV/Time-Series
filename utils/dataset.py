from torch.utils.data import Dataset
import pickle
import numpy as np
import torch


class TimeSeriesSeqDataset(Dataset):
    def __init__(self,data_path,split="train"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.target_cols = data["target_cols"]

        self.data = data[split]

        X = self.data["Seq_X"]
        Y = self.data["Seq_Y"]

        self.X=X.astype(np.float32)
        self.Y=Y.astype(np.float32)

        self.X=self.X.reshape(-1, *X.shape[2:])
        self.Y=self.Y.reshape(-1, *Y.shape[2:])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        temp={
            "x": torch.from_numpy(self.X[idx]),
            "label": torch.from_numpy(self.Y[idx]),
        }
        return temp

class TimeSeriesSeqIdDataset(Dataset):
    def __init__(self,data_path,split="train",batch_dim="id"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.target_cols = data["target_cols"]

        self.data = data[split]

        X = self.data["Seq_X"]
        Y = self.data["Seq_Y"]

        self.X=X.astype(np.float32)
        self.Y=Y.astype(np.float32)

        if batch_dim=="seq":
            self.X=self.X
            self.Y=self.Y
        elif batch_dim=="id":
            self.X=np.transpose(self.X, axes=(1,0,2,3))
            self.Y=np.transpose(self.Y, axes=(1,0,2,3))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        temp={
            "x": torch.from_numpy(self.X[idx]),
            "label": torch.from_numpy(self.Y[idx]),
        }
        return temp


class TimeSeriesDataset(Dataset):
    def __init__(self,data_path,split="train",batch_dim="id"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.target_cols = data["target_cols"]

        self.data = data[split]
        X = self.data["X"]
        Y = self.data["Y"]

        X=X.astype(np.float32)
        Y=Y.astype(np.float32)

        if batch_dim=="id":
            self.X=X
            self.Y=Y
        elif batch_dim=="time":
            self.X=np.transpose(X, axes=(1,0,2))
            self.Y=np.transpose(Y, axes=(1,0,2))
        ...
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        temp={
            "x": torch.from_numpy(self.X[idx]),
            "label": torch.from_numpy(self.Y[idx]),
        }
        return temp


if __name__ == "__main__":
    data_path = "data/hgjj/data.pkl"
    dataset = TimeSeriesSeqDataset(data_path, split="train")
    aaa=dataset[1]["label"]
    dataset = TimeSeriesDataset(data_path, split="train")
    aaa=dataset[1]["label"]
