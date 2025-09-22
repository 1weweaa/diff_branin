import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Branin函数定义
def branin(xy):
    x, y = xy
    a = 1.0
    b = 5.1 / (4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    return a*(y - b*x**2 + c*x - r)**2 + s*(1-t)*np.cos(x) + s-5*(x+2*y-10)

# 创建网格点
x = np.linspace(-5, 10, 200)
y = np.linspace(0, 15, 200)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = branin([X[i, j], Y[i, j]])

# 全局最小值点
global_minima = [
    (-np.pi, 12.275),
    (np.pi, 2.275), 
    (9.42478, 2.475)
]

# 在整个定义域中找到最小值
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
min_point = (X[min_idx], Y[min_idx])
min_value = Z[min_idx]

# 创建图形
fig = plt.figure(figsize=(18, 5))

# 1. 3D表面图
ax1 = fig.add_subplot(131, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Branin函数 - 3D表面图')

# 标记全局最小值点
for point in global_minima:
    if -5 <= point[0] <= 10 and 0 <= point[1] <= 15:
        z_val = branin(point)
        ax1.scatter([point[0]], [point[1]], [z_val], color='red', s=100, marker='*')

# 标记最小值点
ax1.scatter([min_point[0]], [min_point[1]], [min_value],
            color='blue', s=100, marker='o')

fig.colorbar(surf, ax=ax1, shrink=0.5)

# 2. 等高线图
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.6, linewidths=0.5)
contourf = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
ax2.clabel(contour, inline=True, fontsize=8)

# 标记全局最小值点
for point in global_minima:
    if -5 <= point[0] <= 10 and 0 <= point[1] <= 15:
        ax2.plot(point[0], point[1], 'r*', markersize=15)

# 标记最小值点
ax2.plot(min_point[0], min_point[1], 'bo', markersize=10)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Branin函数 - 等高线图')
ax2.grid(True, alpha=0.3)
fig.colorbar(contourf, ax=ax2)

# 3. 热力图
ax3 = fig.add_subplot(133)
heatmap = ax3.imshow(Z, extent=[-5, 10, 0, 15], origin='lower', cmap='viridis', aspect='auto')

# 标记全局最小值点
for point in global_minima:
    if -5 <= point[0] <= 10 and 0 <= point[1] <= 15:
        ax3.plot(point[0], point[1], 'r*', markersize=15)

# 标记最小值点
ax3.plot(min_point[0], min_point[1], 'bo', markersize=10)

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Branin函数 - 热力图')
fig.colorbar(heatmap, ax=ax3)

plt.tight_layout()
plt.show()

# 打印函数信息
print("Branin函数信息:")
print(f"定义域: x ∈ [-5, 10], y ∈ [0, 15]")
print(f"全局最小值: f(x*) ≈ 0.397887")
print("全局最小值点:")
for i, point in enumerate(global_minima, 1):
    print(f"  {i}. ({point[0]:.5f}, {point[1]:.3f}) = {branin(point):.6f}")

print("\n在整个定义域内:")
print(f"函数最小值: {min_value:.6f}")
print(f"函数最大值: {np.max(Z):.6f}")
print(f"函数平均值: {np.mean(Z):.6f}")
print(f"最小值点: ({min_point[0]:.5f}, {min_point[1]:.5f})")