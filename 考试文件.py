import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mtl

# 设置字体
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 1. 生成数据 y=3x+2 + 噪声
np.random.seed(42)
x = np.linspace(0, 10, 10)
print(x)
y = 3 * x + 2 + np.random.normal(0, 1, size=10)
print(y)
# 划分训练集(前8)和测试集(后2)
x_train, y_train = x[:8], y[:8]
x_test, y_test = x[8:], y[8:]

# 添加偏置列 x0=1
X_train = np.column_stack([np.ones(8), x_train])
X_test = np.column_stack([np.ones(2), x_test])

# 预测函数
def hypothesis(x, theta):
    return x @ theta

# 代价函数
def cost_function(x, y, theta):
    m = len(y)
    error = hypothesis(x, theta) - y
    return 1/(2*m) * np.sum(error**2)

# 梯度下降
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        theta = theta - (alpha/m) * (x.T @ (hypothesis(x, theta) - y))
        cost_history.append(cost_function(x, y, theta))
    return theta, cost_history

# 参数初始化
theta = np.zeros(2)
alpha = 0.01
iterations = 30

# 训练模型
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations)
print(f"最优参数: θ0={theta[0]:.2f}, θ1={theta[1]:.2f}")

# (1) 训练集散点图和拟合直线
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, c='b', label='训练数据')
plt.plot(x_train, hypothesis(X_train, theta), 'r-', label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('训练集散点图和拟合直线')
plt.legend()
plt.grid(True)
plt.show()

# (2) 代价函数三维曲面
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-10, 10, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i,j] = cost_function(X_train, y_train, t)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_mesh, theta1_mesh, J_vals.T, cmap='viridis')
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('J(θ0,θ1)')
ax.set_title('代价函数三维曲面')
plt.show()

# (3) 代价函数等高线图
plt.figure(figsize=(8, 6))
plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-1, 3, 20))
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title('代价函数等高线图')
plt.colorbar(label='J(θ0,θ1)')
plt.scatter(theta[0], theta[1], c='r', marker='x', s=100)
plt.grid(True)
plt.show()

# (4) 梯度下降过程
plt.figure(figsize=(8, 5))
plt.plot(range(iterations), cost_history)
plt.xlabel('迭代次数')
plt.ylabel('代价J(θ)')
plt.title('梯度下降过程')
plt.grid(True)
plt.show()