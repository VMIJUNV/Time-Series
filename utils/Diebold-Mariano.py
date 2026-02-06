import pickle
import numpy as np
from scipy import stats
import json


def dm_test(method_data, output_path):
    sample=next(iter(method_data.values()))
    num_method=len(method_data)
    N=sample["preds"].shape[0]
    T=sample["preds"].shape[1]
    F=sample["preds"].shape[2]

    # 计算误差
    method_err={}
    method_names=[]
    mean_axis=(0,1) # 要平均的误差维度，第0维是个体数量，第1维是特征数
    err_shape = (T,) # 最终误差的形状
    for method, data in method_data.items():
        preds=data["origin_preds"]
        labels=data["origin_labels"]
        err = np.abs(preds - labels)
        err=err.transpose((0,2,1))
        err = np.mean(err, axis=mean_axis)
        method_err[method] = err
        method_names.append(method)
    
    # 计算d，代表之间的误差差异
    d=np.empty((num_method, num_method,*err_shape))
    for i, method_i in enumerate(method_names):
        for j, method_j in enumerate(method_names):
            err_i=method_err[method_i]
            err_j=method_err[method_j]
            d[i, j,...] = err_i - err_j
    
    # 计算dm统计量
    d_mean=np.mean(d,axis=(-1,))
    d_std=np.std(d, axis=(-1,),ddof=1)+1e-8
    dm = d_mean/d_std

    # 计算p值
    p_values = np.empty_like(dm)
    for i in range(num_method):
        for j in range(num_method):
            dm_=dm[i, j, ...]
            p=stats.norm.cdf(dm_)
            p_values[i, j] = p
    
    # 格式化结果
    p_res={}
    dm_res={}
    for i, method_i in enumerate(method_names):
        temp1={}
        temp2={}

        for j, method_j in enumerate(method_names):
            temp1[method_j]=float(p_values[i, j])
            temp2[method_j]=float(dm[i, j])
        p_res[method_i]=temp1
        dm_res[method_i]=temp2
    
    res={
        "p_values":p_res,
        "dm":dm_res,
    }

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    
    # one_step
    method_data_map={
        "RNN":"method/RNN/result/one_step_data.pkl",
        "LSTM":"method/LSTM/result/one_step_data.pkl",
        "Transformer":"method/Transformer/result/one_step_data.pkl",
        "tft":"method/tft/result/one_step_data.pkl",
    }

    method_data={}
    for method, data_path in method_data_map.items():
        with open(data_path, "rb") as f:
            method_data[method] = pickle.load(f)
    
    output_path="utils/Diebold-Mariano_res/one_step_dm.json"
    dm_test(method_data,output_path)

    # multiple_steps
    method_data_map={
        "RNN":"method/RNN/result/multiple_steps_data.pkl",
        "LSTM":"method/LSTM/result/multiple_steps_data.pkl",
        "Transformer":"method/Transformer/result/multiple_steps_data.pkl",
        "tft":"method/tft/result/multiple_steps_data.pkl",
    }

    method_data={}
    for method, data_path in method_data_map.items():
        with open(data_path, "rb") as f:
            method_data[method] = pickle.load(f)
    
    output_path="utils/Diebold-Mariano_res/multiple_steps_dm.json"
    dm_test(method_data,output_path)
