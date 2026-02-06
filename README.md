# 步骤1: 数据预处理

运行utils/data_processing.py

# 步骤2: 训练模型

运行method/LSTM/train.py

LSTM这个方法的代码带有一定的注释

其他模型方法运行同理

# 步骤3: 评估模型

运行method/LSTM/eval.py
结果在method/LSTM/result目录下
其他模型方法运行同理

# 步骤4: Diebold-Mariano检验

评估完所有模型后，运行utils/Diebold-Mariano.py
结果在utils/Diebold-Mariano_res目录下