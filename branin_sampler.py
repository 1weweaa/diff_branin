import numpy as np 
import os  # 导入 os 模块用于目录操作
import pickle as pkl
from trainer import Branin  # 假设 trainer.py 在同一目录下
from bayeso_benchmarks.two_dim_branin import Branin as BraninFunction
# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义高斯分布的参数
mean = np.array([-np.pi, 12.275])  # Branin函数的一个最小值点
cov = np.array([[1.0, 0.0], [0.0, 1.0]])  # 单位协方差矩阵

# 生成高斯分布采样点
X = np.random.multivariate_normal(mean=mean, cov=cov, size=5500)
print(f"生成的原始点数: {X.shape}")

# 筛选位于Branin函数定义域内的点
X_inside = X[
    (X[:, 0] > -5) & 
    (X[:, 0] < 10) & 
    (X[:, 1] > 0) & 
    (X[:, 1] < 15)
]
print(f"筛选后的有效点数: {X_inside.shape}")

# 确保有足够的点
if len(X_inside) < 5000:
    # 如果点数不足，补充采样
    additional_needed = 5000 - len(X_inside)
    while additional_needed > 0:
        additional_X = np.random.multivariate_normal(mean=mean, cov=cov, size=additional_needed)
        additional_inside = additional_X[
            (additional_X[:, 0] > -5) & 
            (additional_X[:, 0] < 10) & 
            (additional_X[:, 1] > 0) & 
            (additional_X[:, 1] < 15)
        ]
        X_inside = np.vstack((X_inside, additional_inside))
        additional_needed = 5000 - len(X_inside)
    
    # 如果补充后仍然超过5000点，截取前5000个
    X_inside = X_inside[:5000]

print(f"最终点数: {X_inside.shape}")

# 加载Branin函数预测器
# 确保路径正确，或者使用绝对路径
# branin = Branin(path="diff_branin/dataset/branin_unif_5000.p")

# 预测函数值

Ys = BraninFunction().output(X_inside)
print(f"预测值形状: {Ys.shape}")

# 创建目标目录（如果不存在）
output_dir = "design_baselines/diff_branin/dataset"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

# 定义输出文件路径
output_file = os.path.join(output_dir, "branin_gaussian_5k.p")

# 保存数据
with open(output_file, "wb") as f:
    pkl.dump([X_inside, Ys], f)
    print(f"数据已成功保存到: {output_file}")

# 验证保存的数据
try:
    with open(output_file, "rb") as f:
        loaded_data = pkl.load(f)
        loaded_X, loaded_Y = loaded_data
        print(f"加载的数据点形状: {loaded_X.shape}")
        print(f"加载的函数值形状: {loaded_Y.shape}")
        print("数据验证成功!")
        print(f"前5个数据点:\n{loaded_X[:5]}")
        print(f"前5个函数值:\n{loaded_Y[:5]}")
except Exception as e:
    print(f"验证数据时出错: {e}")
# 保存所有点到TXT文件 - 修复格式化错误
txt_file = os.path.join(output_dir, "branin_gaussian_5k.txt")
with open(txt_file, "w") as f:
    # 写入表头
    f.write("x0\tx1\tf(x)\n")
    
    # 写入所有点 - 修复格式化错误
    for i in range(len(X_inside)):
        # 确保Ys[i]是标量值
        f_value = Ys[i] if isinstance(Ys[i], (float, int)) else Ys[i][0]
        f.write(f"{X_inside[i, 0]:.6f}\t{X_inside[i, 1]:.6f}\t{f_value:.6f}\n")
    
    print(f"所有点已保存到TXT文件: {txt_file}")

# # a*f(x) + b
# al = [1,2,3,4,5]
# bl = [1,2,3,4,5]

# import pickle as pkl
# for i in range(len(al)):
#     with open(f"dataset/branin_gaussian_5k_{al[i]}_{bl[i]}.p", "wb") as f:
#         pkl.dump([X_inside, al[i]*Ys+bl[i]], f)
