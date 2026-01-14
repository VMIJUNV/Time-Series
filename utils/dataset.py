from torch.utils.data import Dataset
import pickle
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self,data_path,split="train"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.scalerX = data["scalerX"]
        self.scalerY = data["scalerY"]

        self.data = data[split]
        X = self.data["X"]
        Y = self.data["Y"]

        X=X.astype(np.float32)
        Y=Y.astype(np.float32)

        self.X=X.reshape(-1, *X.shape[2:])
        self.Y=Y.reshape(-1, *Y.shape[2:])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        temp={
            "x": self.X[idx],
            "label": self.Y[idx],
        }
        return temp
    

if __name__ == "__main__":
    data_path = "data/sse50/data.pkl"
    dataset = TimeSeriesDataset(data_path, split="train")
    aaa=dataset[0]
    ...
