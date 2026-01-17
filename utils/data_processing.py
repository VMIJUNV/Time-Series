import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from utils.more_utils import BaseDataclass

@dataclass
class DataProcessingArgs(BaseDataclass):
    data_path: str = None
    output_path: str = None

    id_col: str = "code"
    time_col: str = "date"

    normalization_cols: list = field(default_factory=list)

    feature_cols: list = field(default_factory=list)
    target_cols: list = field(default_factory=list)
    
    window_size: int = 10

    train_range: list = field(default_factory=list)
    val_range: list = field(default_factory=list)
    test_range: list = field(default_factory=list)

class DataProcessing:
    def __init__(self,args:DataProcessingArgs):
        self.args = args
        self.feature_cols = self.args.feature_cols
        self.normalization_cols = self.args.normalization_cols
        self.target_cols = self.args.target_cols

        self.data_path = Path(args.data_path)
        self.data_origin = pd.read_csv(self.data_path)
        
        self.prepare()
        self.handle_missing()
        self.split_data()
        self.scale_data()
        self.build_data()

    # 准备
    def prepare(self):
        self.data = self.data_origin.copy()
        self.data["id_col"] = self.data[self.args.id_col]
        self.data["time_col"] = pd.to_datetime(self.data[self.args.time_col])

        self.data = self.data.sort_values(["id_col","time_col"])
        self.time_set = self.data["time_col"].unique()

    # 异常处理
    def handle_missing(self):
        all_cols = list(set(self.feature_cols + self.target_cols))
        expected_time_set = set(self.time_set)

        def is_valid_group(group):
            # 1. 检查时间点是否完整
            time_check = set(group["time_col"]) == expected_time_set
            # 2. 检查是否有缺失值 (any().any() 检查整个子表)
            null_check = not group[all_cols].isnull().any().any()
            return time_check and null_check

        # 过滤并更新
        self.data = self.data.groupby("id_col").filter(is_valid_group)
        self.id_set = self.data["id_col"].unique()

        print("id count:", len(self.id_set))
        print("time count:", len(self.time_set))
        print("start time:", self.time_set[0])
        print("end time:", self.time_set[-1])

    def split_data(self):
        self.time_set

        train_range = [pd.to_datetime(t) for t in self.args.train_range]
        val_range = [pd.to_datetime(t) for t in self.args.val_range]
        test_range = [pd.to_datetime(t) for t in self.args.test_range]

        train_times = self.time_set[(self.time_set >= train_range[0]) & (self.time_set <= train_range[1])]
        val_times = self.time_set[(self.time_set >= val_range[0]) & (self.time_set <= val_range[1])]
        test_times = self.time_set[(self.time_set >= test_range[0]) & (self.time_set <= test_range[1])]

        self.data.loc[self.data["time_col"].isin(train_times), "train_split"] = True
        self.data.loc[self.data["time_col"].isin(val_times), "val_split"] = True
        self.data.loc[self.data["time_col"].isin(test_times), "test_split"] = True
        ...

    def scale_data(self):
        train_data = self.data[self.data["train_split"]==True]
        scaler_mean = train_data[self.normalization_cols].mean()
        scaler_std = train_data[self.normalization_cols].std()

        self.data[self.normalization_cols] = (self.data[self.normalization_cols] - scaler_mean) / scaler_std

        self.scaler_mean =dict(scaler_mean)
        self.scaler_std =dict(scaler_std)

    # 构建数据序列
    def build_data(self):
        
        def build_data_(data):
            # X:(N,T,F) Y:(N,T,L)
            X = []
            Y = []

            for id_val in self.id_set:
                group = data[data["id_col"] == id_val]
                X.append(group[self.feature_cols].values)
                Y.append(group[self.target_cols].values)
            X = np.stack(X, axis=0)
            Y = np.stack(Y, axis=0)

            # Seq_X:(S,N,WS,F) Seq_Y:(S,N,WS,L)
            Seq_X = []
            Seq_Y = []

            window_size = self.args.window_size
            _, T, _ = X.shape
            S = T - window_size + 1

            for s in range(S):
                Seq_X.append(X[:, s:s+window_size, :])  # (N, WS, F)
                Seq_Y.append(Y[:, s:s+window_size, :])  # (N, WS, L)

            Seq_X = np.stack(Seq_X, axis=0)
            Seq_Y = np.stack(Seq_Y, axis=0)

            return X,Y,Seq_X,Seq_Y

        train_data = self.data[self.data["train_split"]==True]
        val_data = self.data[self.data["val_split"]==True]
        test_data = self.data[self.data["test_split"]==True]

        X_train,Y_train,Seq_X_train,Seq_Y_train = build_data_(train_data)
        X_val,Y_val,Seq_X_val,Seq_Y_val = build_data_(val_data)
        X_test,Y_test,Seq_X_test,Seq_Y_test = build_data_(test_data)

        save_data = {
            "train":{
                "X": X_train,
                "Y": Y_train,
                "Seq_X": Seq_X_train,
                "Seq_Y": Seq_Y_train,
            },
            "val":{
                "X": X_val,
                "Y": Y_val,
                "Seq_X": Seq_X_val,
                "Seq_Y": Seq_Y_val,
            },
            "test":{
                "X": X_test,
                "Y": Y_test,
                "Seq_X": Seq_X_test,
                "Seq_Y": Seq_Y_test,
            },
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "scaler": {
                "mean": self.scaler_mean,
                "std": self.scaler_std,
            },
        }

        with open(self.args.output_path, "wb") as f:
            pickle.dump(save_data, f)
        ...

    @staticmethod
    def inverse_scale(data:np.ndarray,data_cols:list,scaler:dict):
        data_denorm = data.copy()
        scaler_cols=list(scaler["mean"].keys())
        for index,col in enumerate(data_cols):
            if col in scaler_cols:
                data_denorm[...,index] = data[...,index] * scaler["std"][col] + scaler["mean"][col]
        return data_denorm



if __name__ == '__main__':
    config_path = "utils/data_processing_config/hgjj.yaml"

    args = DataProcessingArgs.from_yaml(config_path)
    DataProcessing(args)