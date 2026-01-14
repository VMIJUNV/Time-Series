import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from utils.more_utils import BaseDataclass

@dataclass
class DataProcessingArgs(BaseDataclass):
    data_path: str = None
    output_path: str = None

    id_col: str = "code"
    time_col: str = "date"

    feature_cols: list = field(default_factory=list)
    normalization_feature_cols: list = field(default_factory=list)

    target_cols: list = field(default_factory=list)
    normalization_target_cols: list = field(default_factory=list)
    #-1为预测下一期。0为预测当前期
    target_shift_step: int = 0 

    missing_mode: str = "drop"
    window_size: int = 10

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

class DataProcessing:
    def __init__(self,args:DataProcessingArgs):
        self.args = args
        self.feature_cols = self.args.feature_cols
        self.normalization_feature_cols = self.args.normalization_feature_cols
        self.target_cols = self.args.target_cols
        self.normalization_target_cols = self.args.normalization_target_cols
        self.label_cols = ["label_"+col for col in self.target_cols]

        self.data_path = Path(args.data_path)
        self.data_origin = pd.read_csv(self.data_path)
        
        self.prepare()
        self.handle_missing()
        self.build_labels()
        self.build_sequences()
        self.split_data()
        self.scale_data()
        self.save_data()

    # 准备
    def prepare(self):
        self.data = self.data_origin.copy()
        self.data["time_col"] = pd.to_datetime(self.data[self.args.time_col])
        self.data["time_col"] = pd.factorize(self.data["time_col"], sort=True)[0]

        self.data["id_col"] = self.data[self.args.id_col]
        
        self.data=self.data.sort_values(["id_col","time_col"])

        self.time_set = self.data["time_col"].unique()
        self.start_time = self.time_set[0]
        self.end_time = self.time_set[-1]

        self.id_set = self.data["id_col"].unique()

    # 异常处理
    def handle_missing(self):
        mode = self.args.missing_mode
        all_cols = list(set(self.feature_cols + self.target_cols))

        if mode == "drop":
            valid_ids = []
            grouped = self.data.groupby("id_col")
            for id_val, group in grouped:
                if set(group["time_col"]) != set(self.time_set):
                    continue
                if group[all_cols].isnull().values.any():
                    continue
                valid_ids.append(id_val)
            self.data = self.data[self.data["id_col"].isin(valid_ids)]

        self.data.loc[:, "id_col"], uniques = pd.factorize(self.data["id_col"])
        self.id_set = self.data["id_col"].unique()
        ...

    # 构建标签
    def build_labels(self):
        for i, col in enumerate(self.target_cols):
            self.data[self.label_cols[i]] = (
                self.data
                .groupby("id_col")[col]
                .shift(self.args.target_shift_step)
            )
        self.data = self.data.dropna(subset=self.label_cols)

    # 构建数据序列
    def build_sequences(self):
        # X:(N,T,F) Y:(N,T,L)
        X = []
        Y = []

        for id_val in self.id_set:
            group = self.data[self.data["id_col"] == id_val]
            X.append(group[self.feature_cols].values)
            Y.append(group[self.label_cols].values)
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

        self.Seq_X = np.stack(Seq_X, axis=0)
        self.Seq_Y = np.stack(Seq_Y, axis=0)
    
    # 数据划分
    def split_data(self):

        train_ratio = self.args.train_ratio
        val_ratio = self.args.val_ratio
        test_ratio = self.args.test_ratio

        S, N, WS, F = self.Seq_X.shape
        S_train = int(S * train_ratio)
        S_val = int(S * val_ratio)
        S_test = S - S_train - S_val

        self.train_Seq_X = self.Seq_X[:S_train]
        self.train_Seq_Y = self.Seq_Y[:S_train]

        self.val_Seq_X = self.Seq_X[S_train:S_train+S_val]
        self.val_Seq_Y = self.Seq_Y[S_train:S_train+S_val]

        self.test_Seq_X = self.Seq_X[S_train+S_val:]
        self.test_Seq_Y = self.Seq_Y[S_train+S_val:]

    # 数据标准化
    def scale_data(self):

        scalerX = StandardScaler()
        
        norm_indices = [self.feature_cols.index(col) for col in self.normalization_feature_cols]
        num_norm_cols = len(norm_indices)

        self.train_Seq_X_normalization = self.train_Seq_X.copy()
        train_to_scale = self.train_Seq_X[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_train = scalerX.fit_transform(train_to_scale).reshape(self.train_Seq_X.shape[0], self.train_Seq_X.shape[1],self.train_Seq_X.shape[2], num_norm_cols)
        self.train_Seq_X_normalization[..., norm_indices] = scaled_train

        self.val_Seq_X_normalization = self.val_Seq_X.copy()
        val_to_scale = self.val_Seq_X[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_val = scalerX.transform(val_to_scale).reshape(self.val_Seq_X.shape[0], self.val_Seq_X.shape[1],self.val_Seq_X.shape[2], num_norm_cols)
        self.val_Seq_X_normalization[..., norm_indices] = scaled_val

        self.test_Seq_X_normalization = self.test_Seq_X.copy()
        test_to_scale = self.test_Seq_X[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_test = scalerX.transform(test_to_scale).reshape(self.test_Seq_X.shape[0], self.test_Seq_X.shape[1],self.test_Seq_X.shape[2], num_norm_cols)
        self.test_Seq_X_normalization[..., norm_indices] = scaled_test

        self.scalerX = {
            "norm_indices": norm_indices,
            "mean": scalerX.mean_,
            "var": scalerX.var_,
        }

        scalerY = StandardScaler()
        norm_indices = [self.target_cols.index(col) for col in self.normalization_target_cols]
        num_norm_cols = len(norm_indices)

        self.train_Seq_Y_normalization = self.train_Seq_Y.copy()
        train_to_scale = self.train_Seq_Y[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_train = scalerY.fit_transform(train_to_scale).reshape(self.train_Seq_Y.shape[0], self.train_Seq_Y.shape[1],self.train_Seq_Y.shape[2], num_norm_cols)
        self.train_Seq_Y_normalization[..., norm_indices] = scaled_train

        self.val_Seq_Y_normalization = self.val_Seq_Y.copy()
        val_to_scale = self.val_Seq_Y[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_val = scalerY.transform(val_to_scale).reshape(self.val_Seq_Y.shape[0], self.val_Seq_Y.shape[1],self.val_Seq_Y.shape[2], num_norm_cols)
        self.val_Seq_Y_normalization[..., norm_indices] = scaled_val

        self.test_Seq_Y_normalization = self.test_Seq_Y.copy()
        test_to_scale = self.test_Seq_Y[..., norm_indices].reshape(-1, num_norm_cols)
        scaled_test = scalerY.transform(test_to_scale).reshape(self.test_Seq_Y.shape[0], self.test_Seq_Y.shape[1],self.test_Seq_Y.shape[2], num_norm_cols)
        self.test_Seq_Y_normalization[..., norm_indices] = scaled_test

        self.scalerY = {
            "norm_indices": norm_indices,
            "mean": scalerY.mean_,
            "var": scalerY.var_,
        }

    @staticmethod
    def inverse_scale(data, scaler):
        data_denorm = data.copy()
        
        norm_indices = scaler["norm_indices"]
        mean = scaler["mean"]
        var = scaler["var"]
        std = np.sqrt(var)
        
        for i, col_idx in enumerate(norm_indices):
            data_denorm[:, col_idx] = data[:, col_idx] * std[i] + mean[i]
        return data_denorm

    # 保存
    def save_data(self):
        save_data = {
            "train":{
                "X": self.train_Seq_X_normalization,
                "Y": self.train_Seq_Y_normalization,
            },
            "val":{
                "X": self.val_Seq_X_normalization,
                "Y": self.val_Seq_Y_normalization,
            },
            "test":{
                "X": self.test_Seq_X_normalization,
                "Y": self.test_Seq_Y_normalization,
            },
            "scalerX": self.scalerX,
            "scalerY": self.scalerY,
        }

        with open(self.args.output_path, "wb") as f:
            pickle.dump(save_data, f)
        ...


if __name__ == '__main__':
    config_path = "utils/data_processing_config/sse50.yaml"
    # config_path = "utils/data_processing_config/sse50_AR.yaml"

    args = DataProcessingArgs.from_yaml(config_path)
    DataProcessing(args)