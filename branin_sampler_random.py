import numpy as np 
import os  
import pickle as pkl
from bayeso_benchmarks.two_dim_branin import Branin as BraninFunction

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义采样数量
num_samples = 5000  

# Branin函数定义域
x0_min, x0_max = -5, 10
x1_min, x1_max = 0, 15

# 在定义域内均匀随机采样
X_random = np.zeros((num_samples, 2))
X_random[:, 0] = np.random.uniform(x0_min, x0_max, num_samples)
X_random[:, 1] = np.random.uniform(x1_min, x1_max, num_samples)
print(f"生成的随机点数: {X_random.shape}")

# 计算函数值
Ys = BraninFunction().output(X_random)
print(f"预测值形状: {Ys.shape}")

# 创建目标目录
output_dir = "design_baselines/diff_branin/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 pkl 文件
output_file = os.path.join(output_dir, "branin_gaussian_5k.p")
with open(output_file, "wb") as f:
    pkl.dump([X_random, Ys], f)
    print(f"数据已成功保存到: {output_file}")

# 保存 txt 文件
txt_file = os.path.join(output_dir, "branin_random_5k.txt")
with open(txt_file, "w") as f:
    f.write("x0\tx1\tf(x)\n")
    for i in range(len(X_random)):
        f_value = Ys[i] if isinstance(Ys[i], (float, int)) else Ys[i][0]
        f.write(f"{X_random[i, 0]:.6f}\t{X_random[i, 1]:.6f}\t{f_value:.6f}\n")
    print(f"所有点已保存到TXT文件: {txt_file}")
