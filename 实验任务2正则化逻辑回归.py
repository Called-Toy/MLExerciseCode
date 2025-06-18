import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
from scipy.special import expit

# 设置字体
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
with open(r"C:\Users\唐涛\Desktop\machine learning database\ex2data2.txt") as f:
    score1 = []
    score2 = []
    y = []
    for line in f:
        if line:
            parts = line.strip().split(',')
            score1.append(float(parts[0]))
            score2.append(float(parts[1]))
            y.append(float(parts[2]))

score = np.column_stack((score1, score2))
y = np.array(y)

# 可视化数据集
plt.scatter(score[y == 1, 0], score[y == 1, 1], marker='+', color='black', label='Admitted')
plt.scatter(score[y == 0, 0], score[y == 0, 1], marker='o', color='yellow', label='Not admitted')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# 特征映射函数
def feature_mapping(u, v, degree=6):
    """
    将2D特征(u,v)映射到28D多项式特征
    六阶多项式展开应该有28项:
    1, u, v, u^2, uv, v^2, ..., u^6, u^5v, ..., uv^5, v^6
    """
    out = np.ones((u.size, 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.hstack((out, (u ** (i - j) * v ** j).reshape(-1, 1)))
    return out


# 逻辑函数
def sigmoid(z):
    return expit(z)


# 预测函数
def hypothesis(x, theta):
    return sigmoid(x @ theta)


# 代价函数
def reg_cost_function(x, y, theta, r=0):
    m = len(y)
    H = hypothesis(x, theta)
    J = (-1 / m) * np.sum(y * np.log(H) + (1 - y) * np.log(1 - H))
    reg_term = (r / (2 * m)) * np.sum(theta[1:] ** 2)  # 不惩罚theta0
    return J + reg_term


# 梯度下降
def reg_gradient_descent(x, y, theta, alpha, iteration, r=0):
    m = len(y)
    cost_history = []

    for i in range(iteration):
        H = hypothesis(x, theta)
        gradient = (1 / m) * x.T @ (H - y)
        # 正则化项（不惩罚theta0）
        reg_term = (r / m) * theta
        reg_term[0] = 0  # theta0不正则化
        theta = theta - alpha * (gradient + reg_term)
        cost_history.append(reg_cost_function(x, y, theta, r))

    return theta, cost_history

# 决策边界函数
def plot_decision_boundary(theta, ax, lambda_reg):
    # 创建网格
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U, V = np.meshgrid(u, v)

    # 向量化处理所有点
    uv = np.c_[U.ravel(), V.ravel()]
    mapped = feature_mapping(uv[:, 0], uv[:, 1])
    Z = sigmoid(mapped @ theta)
    Z = Z.reshape(U.shape)

    # 绘制决策边界
    ax.contour(U, V, Z, levels=[0.5], colors='red')
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    ax.set_title(f'Decision Boundary (λ={lambda_reg})')


# 实施特征映射
X = feature_mapping(np.array(score1), np.array(score2))

# 初始化参数
theta = np.zeros(X.shape[1])
iteration = 300
alpha = 0.5
r = 1

# 运行梯度下降
theta, cost_history = reg_gradient_descent(X, y, theta, alpha, iteration, r)

# 绘制决策边界
fig, ax = plt.subplots(1)
ax.scatter(score[y == 1, 0], score[y == 1, 1], marker='+', color='black', label='Admitted')
ax.scatter(score[y == 0, 0], score[y == 0, 1], marker='o', color='yellow', label='Not admitted')
plot_decision_boundary(theta, ax, r)
plt.legend()
plt.show()

# 绘制代价函数变化
plt.plot(range(iteration), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.tight_layout()
plt.show()

# 计算训练准确率
predictions = hypothesis(X, theta) >= 0.5
accuracy = np.mean(predictions == y) * 100
print(f'Training Accuracy: {accuracy:.2f}%')