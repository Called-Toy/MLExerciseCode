import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mtl

# 设置字体
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
with open(r"C:\Users\唐涛\Desktop\machine learning database\ex1data1.txt") as f:
    populations = []
    profits = []
    for line in f:
        parts = line.strip().split(',')
        if line:
            populations.append(float(parts[0]))
            profits.append(float(parts[1]))

populations = np.array(populations)
profits = np.array(profits)


# 添加偏置列 x0 = 1
n = len(populations)
X = np.column_stack([np.ones(n), populations])

# 可视化数据集
plt.scatter(populations, profits, label='人口-收益')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.show()

# 预测函数
def hypothesis(x, theta):
    H = x @ theta
    return H

# 代价函数
def cost_function(x, y, theta):
    n = len(profits)
    prediction = hypothesis(x, theta)
    error = prediction - y
    J = 1 / (2 * n) * np.sum(error ** 2)
    return J

# 梯度下降函数
def gradient_descent(x, y, theta, alpha, iterations):
    n = len(profits)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = hypothesis(x, theta)
        errors = predictions - y
        theta = theta - (alpha / n) * (x.T @ errors)
        cost_history[i] = cost_function(x, y, theta)
    return theta, cost_history


# 参数初始化
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

# 寻最优参数
theta, cost_history = gradient_descent(X, profits, theta, alpha, iterations)
print(theta)

# 可视化决策边界
plt.plot(populations, hypothesis(X, theta),c='r')
plt.scatter(populations, profits, label='人口-收益')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.show()

# 可视化代价函数
plt.plot(range(iterations), cost_history)
plt.show()

# 预测结果
print(hypothesis(np.array([1,3.5]),theta))
print(hypothesis(np.array([1,7]),theta))