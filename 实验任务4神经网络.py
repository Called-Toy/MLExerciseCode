import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mtl
from scipy.io import loadmat
from scipy.special import expit
from scipy.optimize import minimize

mtl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 加载数据
data = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\ex4data1.mat")
weights = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\ex4weights.mat")

Theta1 = weights['Theta1']  # theta1.shape = 25*401
Theta2 = weights['Theta2']  # theta2.shape = 10*26

X = data['X']  # X.shape = 5000*400
y = data['y']  # y.shape = 5000*1


# 显示 10×10 的数字网格
def display_data(X):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            idx = np.random.randint(0, 5000)
            digit = X[idx].reshape(20, 20).T  # 20×20 并转置
            axes[i, j].imshow(digit, cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


display_data(X)


# 激活函数
def sigmoid(z):
    return expit(z)


# Sigmoid梯度函数
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


# 随机初始化权重
def rand_initialize_weights(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init


# 前向传播
def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]

    # 输入层 -> 隐藏层
    a1 = np.hstack([np.ones((m, 1)), X])  # 添加偏置单元 (5000x401)
    z2 = a1 @ Theta1.T  # (5000x401) @ (401x25).T = 5000x25
    a2 = sigmoid(z2)

    # 隐藏层 -> 输出层
    a2 = np.hstack([np.ones((m, 1)), a2])  # 添加偏置单元 (5000x26)
    z3 = a2 @ Theta2.T  # (5000x26) @ (26x10).T = 5000x10
    h = sigmoid(z3)  # hθ(x) = a3 (5000x10)

    return a1, z2, a2, z3, h


# 神经网络代价函数
def nn_cost_function(params, input_layer_size, hidden_layer_size, num_labels, X, y):
    # 解包参数
    Theta1 = params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        hidden_layer_size, input_layer_size + 1)
    Theta2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(
        num_labels, hidden_layer_size + 1)

    m = X.shape[0]

    # 前向传播
    a1, z2, a2, z3, h = forward_propagate(X, Theta1, Theta2)

    # 将y转换为one-hot编码 (5000x10)
    y_matrix = np.zeros((m, num_labels))
    y_matrix[np.arange(m), y.flatten() - 1] = 1  # y值1-10转为0-9索引

    # 计算代价函数
    J = (-1 / m) * np.sum(y_matrix * np.log(h + 1e-10) + (1 - y_matrix) * np.log(1 - h + 1e-10))


    return J


# 反向传播
def backpropagation(params, input_layer_size, hidden_layer_size, num_labels, X, y):
    # 解包参数
    Theta1 = params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        hidden_layer_size, input_layer_size + 1)
    Theta2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(
        num_labels, hidden_layer_size + 1)

    m = X.shape[0]

    # 初始化梯度
    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    # 前向传播
    a1, z2, a2, z3, h = forward_propagate(X, Theta1, Theta2)

    # 将y转换为one-hot编码
    y_matrix = np.zeros((m, num_labels)) # 5000*10
    y_matrix[np.arange(m), y.flatten() - 1] = 1

    # 反向传播
    for t in range(m):
        # 输出层误差
        delta3 = h[t] - y_matrix[t]  # (10,)

        # 隐藏层误差
        delta2 = (Theta2.T @ delta3)[1:] * sigmoid_gradient(z2[t])  # (25,)

        # 累积梯度
        Delta2 += np.outer(delta3, a2[t])  # (10,26)
        Delta1 += np.outer(delta2, a1[t])  # (25,401)

    # 计算正则化梯度
    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    # 展开梯度
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    # 计算代价函数（Cost Function）
    J = np.sum((y_matrix * np.log(h+1e-10)) + ((1 - y_matrix) * np.log((1 - h)+1e-10)))
    cost = (-1 / m) * J

    return cost, grad


# 梯度检查
def gradient_checking(params, input_layer_size, hidden_layer_size, num_labels, X, y):
    # 计算数值梯度
    def compute_numerical_gradient(J, params):
        numgrad = np.zeros_like(params)
        perturb = np.zeros_like(params)
        epsilon = 1e-4

        for p in range(len(params)):
            perturb[p] = epsilon
            loss1 = J(params - perturb)
            loss2 = J(params + perturb)
            numgrad[p] = (loss2 - loss1) / (2 * epsilon)
            perturb[p] = 0

        return numgrad

    # 包装代价函数
    def cost_func(p):
        return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y)

    # 计算数值梯度和解析梯度
    numgrad = compute_numerical_gradient(cost_func, params)
    cost,grad = backpropagation(params, input_layer_size, hidden_layer_size, num_labels, X, y)

    # 比较梯度
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(f"梯度检查结果: {diff} (应该小于1e-9)")

    return diff


# 训练神经网络
def train_neural_network(X, y, input_layer_size, hidden_layer_size, num_labels, maxiter=50):
    # 随机初始化权重
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

    # 梯度检查（仅在小数据集上运行）
    check_X = X[:5]
    check_y = y[:5]
    print("正在进行梯度检查...")
    gradient_checking(initial_params, input_layer_size, hidden_layer_size, num_labels, check_X, check_y)

    # 使用优化器训练
    print("开始训练神经网络...")
    print("initial_params形狀:", initial_params.shape)
    print("X形狀:", X.shape)
    print("y形狀:", y.shape)
    print("input_layer_size:", input_layer_size)
    print("hidden_layer_size:", hidden_layer_size)
    print("num_labels:", num_labels)

    options = {'maxiter': maxiter}
    res = minimize(fun=backpropagation,
                   x0=initial_params,
                   args=(input_layer_size, hidden_layer_size, num_labels, X, y),
                   method='L-BFGS-B',
                   jac=True,
                   options=options)

    # 提取训练好的参数
    Theta1 = res.x[:hidden_layer_size * (input_layer_size + 1)].reshape(
        hidden_layer_size, input_layer_size + 1)
    Theta2 = res.x[hidden_layer_size * (input_layer_size + 1):].reshape(
        num_labels, hidden_layer_size + 1)

    return Theta1, Theta2


# 预测函数
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.hstack([np.ones((m, 1)), X])
    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)
    return np.argmax(a3, axis=1) + 1  # 返回1-10的预测标签


# 可视化隐藏层
def visualize_hidden_layer(Theta1):
    # 移除偏置单元
    Theta1 = Theta1[:, 1:]

    # 创建25个20x20的图像
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < 25:  # 确保不超出范围
                digit = Theta1[idx].reshape(20, 20).T
                axes[i, j].imshow(digit, cmap='gray')
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


# 主程序
def main():
    # 参数设置
    input_layer_size = 400  # 400个输入单元
    hidden_layer_size = 25  # 25个隐藏单元
    num_labels = 10  # 10个输出类别
    maxiter = 50  # 训练次数
    # 训练神经网络
    Theta1, Theta2 = train_neural_network(X, y, input_layer_size, hidden_layer_size, num_labels,maxiter)

    # 计算训练集准确率
    pred = predict(Theta1, Theta2, X)
    accuracy = np.mean(pred == y.flatten()) * 100
    print(f"训练集准确率: {accuracy:.2f}%")

    # 可视化隐藏层
    print("正在可视化隐藏层...")
    visualize_hidden_layer(Theta1)


if __name__ == "__main__":
    main()