import numpy as np
import os
import pickle as pkl
from bayeso_benchmarks.two_dim_branin import Branin as BraninFunction

# -----------------------------
# 设置随机种子
np.random.seed(42)

# -----------------------------
# 定义采样区域（每个区域以Branin函数全局最小值为中心）
regions = [
    {"mean": np.array([-np.pi, 12.275]), "cov": np.array([[1.0, 0.0], [0.0, 1.0]]), "num_samples": 2000},
    {"mean": np.array([np.pi, 2.275]), "cov": np.array([[1.0, 0.0], [0.0, 1.0]]), "num_samples": 2000},
    {"mean": np.array([9.42478, 2.475]), "cov": np.array([[1.0, 0.0], [0.0, 1.0]]), "num_samples": 2000}
]

# -----------------------------
# 生成采样点
X_all = []
for region in regions:
    X_region = np.random.multivariate_normal(
        mean=region["mean"],
        cov=region["cov"],
        size=region["num_samples"]
    )
    X_all.append(X_region)

# 合并所有区域
X_all = np.vstack(X_all)
print(f"合并后的原始点数: {X_all.shape}")

# -----------------------------
# 筛选在Branin函数定义域内的点
x_min, x_max = -5, 10
y_min, y_max = 0, 15

X_inside = X_all[
    (X_all[:, 0] >= x_min) & (X_all[:, 0] <= x_max) &
    (X_all[:, 1] >= y_min) & (X_all[:, 1] <= y_max)
]
print(f"筛选后的有效点数: {X_inside.shape}")

# -----------------------------
# 确保有足够的点（可选：如果想要精确数量）
desired_num = 5000
if X_inside.shape[0] < desired_num:
    additional_needed = desired_num - X_inside.shape[0]
    while additional_needed > 0:
        for region in regions:
            X_region = np.random.multivariate_normal(
                mean=region["mean"],
                cov=region["cov"],
                size=additional_needed
            )
            X_region_inside = X_region[
                (X_region[:, 0] >= x_min) & (X_region[:, 0] <= x_max) &
                (X_region[:, 1] >= y_min) & (X_region[:, 1] <= y_max)
            ]
            X_inside = np.vstack((X_inside, X_region_inside))
            additional_needed = desired_num - X_inside.shape[0]
            if additional_needed <= 0:
                break
    X_inside = X_inside[:desired_num]

print(f"最终点数: {X_inside.shape}")

# -----------------------------
# 计算Branin函数值
Ys = BraninFunction().output(X_inside)
print(f"函数值形状: {Ys.shape}")

# -----------------------------
# 保存数据
output_dir = "design_baselines/diff_branin/dataset"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "branin_gaussian_multi_region_5k.p")

with open(output_file, "wb") as f:
    pkl.dump([X_inside, Ys], f)

print(f"数据已成功保存到: {output_file}")

# -----------------------------
# 验证保存的数据
with open(output_file, "rb") as f:
    loaded_X, loaded_Y = pkl.load(f)
    print(f"加载的数据点形状: {loaded_X.shape}")
    print(f"加载的函数值形状: {loaded_Y.shape}")
    print("数据验证成功!")
    print(f"前5个数据点:\n{loaded_X[:5]}")
    print(f"前5个函数值:\n{loaded_Y[:5]}")
