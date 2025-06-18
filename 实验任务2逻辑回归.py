import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
from scipy.special import expit

# 设置字体
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
with open(r"C:\Users\唐涛\Desktop\machine learning database\ex2data1.txt") as f:
    score = []
    if_admitted = []
    for line in f:
        if line:
            parts = line.strip().split(',')
            score.append([float(parts[0]),float(parts[1])])
            if_admitted.append(int(parts[2]))

# 画出学生两次得分和录取率的散点图
for i in range(len(score)):
    if if_admitted[i] == 1:
        plt.scatter(score[i][0],score[i][1],marker='+',color='black')
    else:
        plt.scatter(score[i][0],score[i][1],marker='.',color='yellow')

plt.show()

# 添加偏置项theta 0
score = np.array(score)
if_admitted = np.array(if_admitted)
x = np.column_stack([np.ones(len(score)),score])
print(x)
def sigmoid(z): # sigmoid函数
    return expit(z)


def hypothesis(x,theta): # 预测函数
    H = sigmoid(x @ theta)
    return H

def cost_function(x,y,theta): # 代价函数
    n = len(y)
    H = hypothesis(x,theta)
    H = np.clip(H, 1e-15, 1 - 1e-15)
    J = (-1 / n) * np.sum(y * np.log(H) + (1 - y) * np.log(1 - H))
    return J

def gradient_vector(x,y,theta): # 梯度向量
    n = len(y)
    H = hypothesis(x,theta)
    grad = (1 / n) * x.T @ (H -y)
    return grad

def hessian_matrix(x,y,theta): # 海森矩阵
    n = len(y)
    H = hypothesis(x,theta)
    D = np.diag(H *(1 - H))
    hessian = (1 / n) * x.T @ D @ x
    return hessian

def newton_function(x,y,theta,iterations): # 牛顿法
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        grad = gradient_vector(x,y,theta)
        H = hessian_matrix(x,y,theta)
        theta = theta - np.linalg.solve(H, grad)
        cost_history[i] = cost_function(x,y,theta)
    return theta, cost_history

theta = np.zeros(3)
iterations = 10

theta,cost_history = newton_function(x,if_admitted,theta,iterations)



score_1 = 45
score_2 = 85
predict_score = np.array([1,score_1,score_2])
print(hypothesis(predict_score,theta))

# 生成决策边界
x1_values = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]

for i in range(len(score)):
    if if_admitted[i] == 1:
        plt.scatter(score[i][0],score[i][1],marker='+',color='black')
    else:
        plt.scatter(score[i][0],score[i][1],marker='.',color='yellow')
plt.plot(x1_values, x2_values, 'r-', label='Decision Boundary')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()

plt.plot(range(iterations),cost_history)
plt.show()

