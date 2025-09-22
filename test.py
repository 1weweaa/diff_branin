import numpy as np

def branin(x):
    """
    Branin 函数
    x: shape (N, 2)
    """
    x1, x2 = x[:, 0], x[:, 1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

def lagrangian(x, lam):
    """
    拉格朗日函数
    x: shape (N, 2)
    lam: float 或 shape (N,) 的数组
    """
    g = -x[:, 0] - 2 * x[:, 1] + 10
    return branin(x) + lam * g

# 多组点
X = np.array([
    # [np.pi, 2.275],      # Branin 最优点之一
    [-np.pi, 12.275],    # 另一个最优点
    [1.409847, 6.635965]     # 第三个最优点 (≈ 3π, 2.475)
])

lam = 5 # 单个 λ
print("L(x, λ) (单个 λ):", lagrangian(X, lam))

# lam_arr = np.array([1.0, 2.0, 3.0])   # 每个点一个 λ
# print("L(x, λ) (多个 λ):", lagrangian(X, lam_arr))
