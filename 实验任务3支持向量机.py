import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
from cvxopt import matrix, solvers

# 设置字体
mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# rbf核svm数据集读取，可视化
def rbf_kernel_plot_read():
    with open(r"C:\Users\唐涛\Desktop\machine learning database\ex5533.txt") as f:
        u, v = [], []
        for line in f:
            parts = line.strip().split('\t')
            if line:
                u.append(float(parts[0]))
                v.append(float(parts[1]))

    y = np.loadtxt(r"C:\Users\唐涛\Desktop\machine learning database\ex5534.txt")

    # 转换成numpy数组
    X = np.column_stack((u, v))  # X.shape = 863*2
    y = np.array(y)  # y.shape = 1*863
    # 将标签0转换为-1，便于SVM处理
    y[y == 0] = -1

    # 画出数据集
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.legend()
    plt.title("原始数据分布")
    plt.show()

    return X, y


# rbf核svm数据集2读取，可视化
def rbf2_kernel_plot_read():
    with open(r"C:\Users\唐涛\Desktop\code\ch5\ex5.3.3\ex533.txt") as f:
        u, v, y = [], [], []
        for line in f:
            parts = line.strip().split(' ')
            if line:
                y.append(float(parts[0]))
                u.append(float(parts[1].split(':')[1]))
                v.append(float(parts[2].split(':')[1]))

    # 转换成numpy数组
    X = np.column_stack((u, v))  # X.shape = 211*2
    y = np.array(y)  # y.shape = 1*211
    # 画出数据集
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.legend()
    plt.title("原始数据分布")
    plt.show()

    return X, y


# 高斯核svm数据集读取，可视化
def gaussian_kernel_svm_plot_read():
    # 设置字体
    mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 加载数据
    with open(r"C:\Users\唐涛\Desktop\machine learning database\ex5531.txt") as f:
        u, v = [], []
        for line in f:
            parts = line.strip().split('\t')
            if line:
                u.append(float(parts[0]))
                v.append(float(parts[1]))

    y = np.loadtxt(r"C:\Users\唐涛\Desktop\machine learning database\ex5532.txt")

    # 转换成numpy数组
    X = np.column_stack((u, v))  # x.shape = 51*2
    y = np.array(y)  # y.shape = 1*51
    # 将标签0转换为-1，便于SVM处理
    y[y == 0] = -1

    # 画出数据集
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.legend()
    plt.title("原始数据分布")
    plt.show()

    return X, y


# 线性svm数据集读取，可视化
def linear_svm_data_plot_read():
    # 加载数据
    with open(r"C:\Users\唐涛\Desktop\code\ch5\ex5.3.1\twofeature.txt") as datas:
        y = []
        u = []
        v = []
        for line in datas:
            line.strip()
            if line:
                parts = line.split(' ')
                y.append(int(parts[0]))  # Convert to int for labels
                block1 = parts[1].split(':')
                block2 = parts[2].split(':')
                u.append(float(block1[1]))
                v.append(float(block2[1]))

    # 转换为numpy数组
    X = np.column_stack((u, v))  # x.shape = 51*2
    y = np.array(y)  # y.shape = 1*51

    # 画出数据集
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.xlabel('Feature u')
    plt.ylabel('Feature v')
    plt.legend()
    plt.title('Data Visualization')
    plt.show()

    return X, y


# 预测函数，用于高斯核svm的决策边界可视化
def predict_gaussian(X, sv_alphas, sv_X, sv_y, b, gamma):
    """
    使用训练好的SVM模型进行预测
    参数:
        X: 待预测样本
        sv_alphas: 支持向量对应的拉格朗日乘子
        sv_X: 支持向量
        sv_y: 支持向量标签
        b: 偏置项
        gamma: 高斯核参数
    返回:
        预测标签(1或-1)
    """
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        s = 0
        for alpha, sv_y_j, sv_x_j in zip(sv_alphas, sv_y, sv_X):
            s += alpha * sv_y_j * np.exp(-gamma * np.linalg.norm(X[i] - sv_x_j) ** 2)
        y_pred[i] = np.sign(s + b)
    return y_pred


# 预测函数，用于rbf核svm决策边界可视化
def predict_rbf(X, sv_alphas, sv_X, sv_y, b, gamma):
    """
    使用训练好的RBF核SVM进行预测
    参数:
        X: 待预测样本
        sv_alphas: 支持向量对应的拉格朗日乘子
        sv_X: 支持向量
        sv_y: 支持向量标签
        b: 偏置项
        gamma: RBF核参数
    返回:
        预测标签(1或-1)
    """
    # 计算RBF核矩阵
    K = rbf_kernel(X, sv_X, gamma)
    # 计算决策函数值
    y_pred = np.dot(K, sv_alphas * sv_y) + b
    return np.sign(y_pred)


# 高斯核svm决策边界可视化
def plot_gaussian_decision_boundary(X, y, sv_X, sv_y, alphas, b, gamma):
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 预测网格点的类别
    Z = predict_gaussian(np.c_[xx.ravel(), yy.ravel()], alphas, sv_X, sv_y, b, gamma)
    Z = Z.reshape(xx.shape)

    # 绘制结果
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['white', 'lightblue'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.scatter(sv_X[:, 0], sv_X[:, 1], s=100, facecolors='none', edgecolors='red', label='支持向量')
    plt.title(f"高斯核SVM (γ={gamma}, C={C})")
    plt.legend()
    plt.show()


# rbf核2svm决策边界可视化
def plot_rbf2_decision_boundary(X, y, sv_X, sv_y, alphas, b, gamma):
    """
    绘制RBF核SVM的决策边界
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 预测网格点的类别
    Z = predict_rbf(np.c_[xx.ravel(), yy.ravel()], alphas, sv_X, sv_y, b, gamma)
    Z = Z.reshape(xx.shape)


    # 绘制结果
    plt.contourf(xx, yy, Z, levels=[-1,0, 1], colors=['white', 'lightblue'], alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.title(f"RBF Kernel SVM (γ={gamma})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# rbf核svm决策边界可视化
def plot_rbf_decision_boundary(X, y, sv_X, sv_y, alphas, b, gamma):
    """
    绘制RBF核SVM的决策边界
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.15, X[:, 0].max() + 0.15
    y_min, y_max = X[:, 1].min() - 0.15, X[:, 1].max() + 0.15
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 预测网格点的类别
    Z = predict_rbf(np.c_[xx.ravel(), yy.ravel()], alphas, sv_X, sv_y, b, gamma)
    Z = Z.reshape(xx.shape)

    # 绘制结果
    plt.contourf(xx, yy, Z, levels=[-1,0, 1], colors=['white', 'lightblue'], alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')
    plt.title(f"RBF Kernel SVM (γ={gamma}, C={C})")
    plt.legend()
    plt.show()


# 线性svm决策边界可视化
def plot_linear_decision_boundary(X, y, w, b, sv_X, C):
    # 绘制两类样本点
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='.', color='yellow', label='Class -1')

    # 标记支持向量
    plt.scatter(sv_X[:, 0], sv_X[:, 1], s=100, facecolors='none', edgecolors='red', label='Support Vectors')

    # 计算并绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    yy = (-w[0] * xx - b) / w[1]
    plt.plot(xx, yy, 'k-', label='Decision Boundary')

    # 绘制间隔边界
    yy_upper = (-w[0] * xx - b + 1) / w[1]  # 上边界
    yy_lower = (-w[0] * xx - b - 1) / w[1]  # 下边界
    plt.plot(xx, yy_upper, 'k--')
    plt.plot(xx, yy_lower, 'k--')

    plt.xlabel('Feature u')
    plt.ylabel('Feature v')
    plt.title(f'SVM Decision Boundary (C={C})')
    plt.legend()
    plt.show()


# 高斯核svm实现
def gaussian_kernel_svm(X, y, gamma=1):
    """
    高斯核SVM实现
    参数:
        X: 特征矩阵
        y: 标签向量(1或-1)

        gamma: 高斯核参数
    返回:
        alphas: 拉格朗日乘子
        sv_X: 支持向量
        sv_y: 支持向量对应的标签
        b: 偏置项
    """
    n_samples = X.shape[0]

    # 计算高斯核矩阵
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)

    # 构建QP问题的参数
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples))))
    A = matrix(y.reshape(1, -1).astype(np.double))
    b = matrix(0.0)

    # 求解QP问题
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()

    # 获取支持向量
    sv = alphas > 1e-5
    sv_X = X[sv]
    sv_y = y[sv]
    sv_alphas = alphas[sv]

    # 计算偏置项b
    b = 0
    for i in range(len(sv_alphas)):
        b += sv_y[i] - np.sum(sv_alphas * sv_y * K[sv][i, sv])
    b /= len(sv_alphas)

    return sv_alphas, sv_X, sv_y, b


# 线性核svm实现
def linear_svm(X, y, C=1):
    # 获取样本数量和特征维度
    n_samples, n_features = X.shape

    # 计算线性核矩阵
    K = np.dot(X, X.T)  # K.shape = matrix(51*51)

    # 构建二次规划问题的参数
    # P = y*y^T * K (Hadamard积)
    P = matrix(np.outer(y, y) * K)  # P <51*51 matrix,tc='d'>
    # q = -1向量
    q = matrix(-np.ones(n_samples))  # q <51*1 matrix, tc='d'>
    # G包含上下界约束: -I和I
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))  # 102*51
    # h对应约束: 0 <= alpha <= C
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))  # 1*102
    # 等式约束: sum(alpha*y) = 0
    A = matrix(y.reshape(1, -1).astype(np.double))  # 1*51
    b = matrix(0.0)  # 1*1

    # 使用二次规划求解器求解
    solution = solvers.qp(P, q, G, h, A,
                          b)  # 得到'x': <51x1 matrix, tc='d'>, 'y': <1x1 matrix, tc='d'>, 's': <102x1 matrix, tc='d'>, 'z': <102x1 matrix, tc='d'>,后面省略
    # 获取拉格朗日乘子并展平
    alphas = np.array(solution['x']).flatten()  # 1*51

    # 筛选支持向量(alpha > 1e-5)
    sv = alphas > 1e-5  # 布尔值列表 1*51
    alphas = alphas[sv]  # 1*11 array
    sv_X = X[sv]  # 11*2 array
    sv_y = y[sv]  # 1*11 array

    # 计算权重向量 w = sum(alpha_i * y_i * x_i)
    w = np.zeros(n_features)
    for i in range(len(alphas)):
        w += alphas[i] * sv_y[i] * sv_X[i]

    # 计算偏置项 b = average(y_i - w·x_i)
    bias = 0
    for i in range(len(alphas)):
        bias += sv_y[i] - np.dot(w, sv_X[i])
    bias /= len(alphas)

    return w, bias, alphas, sv_X, sv_y


# rbf核svm实现
def rbf_svm(X, y, C=1.0, gamma=1.0):
    """
    RBF核SVM实现
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,), 取值应为+1或-1
        C: 正则化参数
        gamma: RBF核参数
    返回:
        alphas: 支持向量对应的拉格朗日乘子
        sv_X: 支持向量
        sv_y: 支持向量对应的标签
        b: 偏置项
    """
    n_samples = X.shape[0]

    # 计算RBF核矩阵
    K = rbf_kernel(X, X, gamma)

    # 构建QP问题的参数
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = matrix(y.reshape(1, -1).astype(np.double))
    b = matrix(0.0)

    # 求解QP问题
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()

    # 获取支持向量
    sv_indices = alphas > 1e-5
    sv_alphas = alphas[sv_indices]
    sv_X = X[sv_indices]
    sv_y = y[sv_indices]

    # 计算偏置项b
    b = 0
    for i in range(len(sv_alphas)):
        b += sv_y[i] - np.sum(sv_alphas * sv_y * K[sv_indices][i, sv_indices])
    b /= len(sv_alphas)

    return sv_alphas, sv_X, sv_y, b


# rbf核函数
def rbf_kernel(X1, X2, gamma=1.0):
    """
    径向基函数(RBF)核计算
    参数:
        X1: 第一个样本集 (m x n)
        X2: 第二个样本集 (k x n)
        gamma: RBF核参数
    返回:
        核矩阵 (m x k)
    """
    sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

# --------------------------------------------------------------------------
# 线性核svm求解，决策边界可视化
X1, y1 = linear_svm_data_plot_read()
w, b, alphas, sv_X, sv_y = linear_svm(X1, y1, 1)
plot_linear_decision_boundary(X1, y1, w, b, sv_X, 1)
w, b, alphas, sv_X, sv_y = linear_svm(X1, y1, 10)
plot_linear_decision_boundary(X1, y1, w, b, sv_X, 10)

w, b, alphas, sv_X, sv_y = linear_svm(X1, y1, 100)
plot_linear_decision_boundary(X1, y1, w, b, sv_X, 100)
# ---------------------------------------------------------------------------
# 高斯核svm求解，决策边界可视化
X2, y2 = gaussian_kernel_svm_plot_read()
gamma = 0.5
C = 1
alphas, sv_X, sv_y, b = gaussian_kernel_svm(X2, y2, gamma=gamma)
plot_gaussian_decision_boundary(X2, y2, sv_X, sv_y, alphas, b, gamma)
# ---------------------------------------------------------------------------
# rbf核svm求解，决策边界可视化
X3, y3 = rbf_kernel_plot_read()
gamma = 100
sv_alphas, sv_X, sv_y, b = gaussian_kernel_svm(X3, y3, gamma=gamma)
plot_rbf_decision_boundary(X3, y3, sv_X, sv_y, sv_alphas, b, gamma)
# ----------------------------------------------------------------------------
# 第二个数据集，rbf核svm求解,决策边界可视化
X4,y4 = rbf2_kernel_plot_read()
gamma = 1
sv_alphas, sv_X, sv_y, b = rbf_svm(X4, y4, gamma=gamma)
plot_rbf2_decision_boundary(X4, y4, sv_X, sv_y, sv_alphas, b, gamma)

gamma = 10
sv_alphas, sv_X, sv_y, b = rbf_svm(X4, y4, gamma=gamma)
plot_rbf2_decision_boundary(X4, y4, sv_X, sv_y, sv_alphas, b, gamma)

gamma = 100
sv_alphas, sv_X, sv_y, b = rbf_svm(X4, y4, gamma=gamma)
plot_rbf2_decision_boundary(X4, y4, sv_X, sv_y, sv_alphas, b, gamma)
# ----------------------------------------------------------------------------