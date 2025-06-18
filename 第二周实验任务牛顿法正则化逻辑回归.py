import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
from scipy.special import expit

# 更改字体显示
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
with open(r"C:\Users\唐涛\Desktop\machine learning database\ex2data2.txt") as f:
    u, v, y = [], [], []
    for line in f:
        parts = line.strip().split(',')
        if line:
            u.append(float(parts[0]))
            v.append(float(parts[1]))
            y.append(float(parts[2]))


def data_scatter():  #数据散点图
    plt.scatter(u[:57], v[:57], marker='+', color='black', label='y=1')
    plt.scatter(u[58:], v[58:], marker='.', color='yellow', label='y=0')
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    plt.legend()


def mapfeature_x(u, v, max_degree=6):  #构建多项式特征向量
    """
    构建多项式特征向量 x = [1, u, v, u², uv, v², ..., v⁶]
    """
    features = [1.0]  # 偏置项
    for degree in range(1, max_degree + 1):
        for i in range(degree + 1):
            j = degree - i
            features.append((u ** i) * (v ** j))
    return np.array(features)


def sigmoid(z):  #sigmoid函数
    return expit(z)


def hypothesis(x, theta):  #预测函数
    return sigmoid(x @ theta)


def cost_function(x, y, theta, lambda_reg):  #代价函数
    m = len(y)
    h = hypothesis(x, theta)
    J = (-(y @ np.log(h) + (1 - y) @ np.log(1 - h))) / m
    reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
    return J + reg_term


def gradient( x, y,theta, lambda_reg):  #梯度向量
    m = len(y)
    h = hypothesis(x,theta)
    grad = (x.T @ (h - y)) / m
    reg_term = (lambda_reg / m) * theta
    reg_term[0] = 0  # 不惩罚偏置项
    return grad + reg_term


def hessian( x, y,theta, lambda_reg):  #海森矩阵
    n = x.shape[0]  # 样本数
    h = hypothesis(x,theta)

    # 计算对角权重矩阵
    D = np.diag(h * (1 - h))

    # 计算Hessian核心部分
    H = (1 / n) * np.dot(x.T, np.dot(D, x))

    # 添加正则化项（不惩罚偏置项，假设theta[0]是偏置）
    if lambda_reg > 0:
        reg_matrix = (lambda_reg / n) * np.eye(len(theta))
        reg_matrix[0, 0] = 0  # 不惩罚偏置项
        H += reg_matrix

    return H


def newton_optimization(x, y,theta, lambda_reg, iteration):  # 牛顿法优化
    cost_history = []
    for i in range(iteration):
        # 计算梯度和Hessian
        grad = gradient(x, y,theta, lambda_reg)
        H = hessian(x, y,theta, lambda_reg)

        # 计算当前代价
        current_cost = cost_function(x, y,theta, lambda_reg)
        cost_history.append(current_cost)

        # 计算牛顿步长
        delta = np.linalg.solve(H, -grad)

        # 更新参数
        theta += delta
    return theta, cost_history

def feature_mapping(u, v, degree=6):
    """
    将2D特征(u,v)映射到28D多项式特征
    六阶多项式展开应该有28项:
    1, u, v, u^2, uv, v^2, ..., u^6, u^5v, ..., uv^5, v^6
    """
    out = np.ones((u.size, 1))
    col = 0
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.hstack((out, (u ** (i - j) * v ** j).reshape(-1, 1)))
            col += 1
    return out

def plot_decision_boundary(theta, ax,lambda_reg):
    # 创建网格
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    U, V = np.meshgrid(u, v)

    # 向量化处理所有点
    uv = np.c_[U.ravel(), V.ravel()]  # 组合所有(u,v)对
    mapped = feature_mapping(uv[:, 0], uv[:, 1])  # 特征映射
    Z = sigmoid(np.dot(mapped, theta.T))  # 向量化计算概率
    Z = Z.reshape(U.shape)  # 重塑为网格形状

    # 绘制决策边界
    ax.contour(U, V, Z, levels=[0.5], colors='red')
    ax.set_xlabel('Feature u')
    ax.set_ylabel('Feature v')
    ax.set_title({'lambda':lambda_reg})

# 构建完整的特征矩阵
x = np.array([mapfeature_x(u[i], v[i]) for i in range(len(u))])
y = np.array(y)
# 初始化参数
theta = np.zeros(x.shape[1])
print(theta.shape,x.shape)
lambda_reg = 1
iteration = 10
tol = 1e-6

# 开始优化
theta, cost_history = newton_optimization( x, y,theta, lambda_reg, iteration)


# 输出结果
print("优化后的参数:", theta)
print("最终代价:", cost_history[-1])

# 结果可视化
fig,ax= plt.subplots(1,2,figsize=(16,8))
ax[0].plot(range(iteration),cost_history)

data_scatter()
plot_decision_boundary(theta,ax[1],lambda_reg)
plt.show()
