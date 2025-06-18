import numpy as np
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt


# ----- 辅助函数 -----
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 避免溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, labels):
    return -np.mean(np.sum(labels * np.log(probs + 1e-8), axis=1))


def relu_backward(dout, cache):
    return dout * (cache > 0)


def mean_pooling(feature_maps, pool_size=2):  # 平均池化操作
    batch_size, channels, h, w = feature_maps.shape
    out_h = h // pool_size
    out_w = w // pool_size
    pooled = np.zeros((batch_size, channels, out_h, out_w))
    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    patch = feature_maps[b, c, i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
                    pooled[b, c, i, j] = np.mean(patch)
    return pooled


# 输入：batch of 2D images (64, 28, 28)
# 卷积核：直接使用3D张量 (20, 9, 9) 表示20个9x9的滤波器

def conv_layer(images, kernels):
    batch_size, h, w = images.shape
    num_kernels, kh, kw = kernels.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    outputs = np.zeros((batch_size, num_kernels, out_h, out_w))
    caches = []  # 用于存储每个卷积操作的输入和核

    for b in range(batch_size):
        for k in range(num_kernels):
            for i in range(out_h):
                for j in range(out_w):
                    input_patch = images[b, i:i + kh, j:j + kw]
                    outputs[b, k, i, j] = np.sum(input_patch * kernels[k])
                    caches.append((input_patch, kernels[k]))  # 存储输入块和核
    return outputs, caches


def conv_layer_backward(dout, kernels, caches, input_shape):
    batch_size, num_kernels, out_h, out_w = dout.shape
    kh, kw = kernels.shape[1], kernels.shape[2]
    dinput_images = np.zeros((batch_size,) + input_shape)
    dkernels = np.zeros_like(kernels)

    cache_idx = 0
    for b in range(batch_size):
        for k in range(num_kernels):
            for i in range(out_h):
                for j in range(out_w):
                    input_patch, kernel = caches[cache_idx]
                    doutput = dout[b, k, i, j]

                    # 将梯度累加到对应的输入图像块
                    dinput_images[b, i:i + kh, j:j + kw] += kernel * doutput

                    # 累加卷积核的梯度
                    dkernels[k] += input_patch * doutput
                    cache_idx += 1

    return dinput_images / batch_size, dkernels / batch_size


def mean_pooling_backward(dpooled, original, pool_size=2):
    batch, channels, h, w = original.shape
    dout = np.zeros_like(original)
    for b in range(batch):
        for c in range(channels):
            for i in range(dpooled.shape[2]):
                for j in range(dpooled.shape[3]):
                    h_start, w_start = i * pool_size, j * pool_size
                    dout[b, c, h_start:h_start + pool_size, w_start:w_start + pool_size] = \
                        dpooled[b, c, i, j] / (pool_size ** 2)  # 均值池化的梯度均分
    return dout


# ----- 预测函数 -----
def predict(X):  # 添加了一个预测函数
    # 前向传播
    conv_out, _ = conv_layer(X, W1)

    relu_out = relu(conv_out)
    pooled = mean_pooling(relu_out, pool_size=2)
    flattened = pooled.reshape(len(X), -1)

    hidden_input = np.dot(flattened, W5) + b_fc1
    hidden_output = relu(hidden_input)

    logits = np.dot(hidden_output, Wo) + b_fc2
    probs = softmax(logits)

    return np.argmax(probs, axis=1)


# One-Hot 编码函数
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def plot_datas(x):
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for i in range(5):
        for j in range(5):
            idx = np.random.randint(0, 1000)
            images_idx = x[idx]
            axes[i, j].imshow(images_idx, cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


# 加载数据
images_data = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\Images.mat")
labels_data = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\Labels.mat")
images = images_data['Images']  # 28*28*10000
labels = labels_data['Labels']  # 10000x1


# 训练测试划分
train_images = images[:, :, :8000]  # 28*28*8000
train_labels = labels[:8000]  # 8000x1
test_images = images[:, :, 8000:]  # 28*28*2000
test_labels = labels[8000:]  # 2000x1

# 数据预处理与形状调整
train_images = np.transpose(train_images, (2, 0, 1))  # 变为 (8000, 28, 28)
test_images = np.transpose(test_images, (2, 0, 1))  # 变为 (2000, 28, 28)

plot_datas(train_images)
#调整标签形状并转换为 integers
train_labels = train_labels.flatten().astype(int) - 1
test_labels = test_labels.flatten().astype(int) - 1

# 归一化图像数据
train_images = train_images / 255.0
test_images = test_images / 255.0

# ----- 初始化权重 -----
num_filters = 20  # 卷积后输出的图像维度20*20
filter_size = 9  # 卷积核尺寸9*9
hidden_units = 100  # 隐藏层单元数
num_classes = 10  # 输出类别数

# 卷积层权重
W1 = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2. / (filter_size * filter_size))
# 全连接层权重
W5 = np.random.randn(num_filters * 10 * 10, hidden_units) * np.sqrt(2.0 / (num_filters * 10 * 10))  #
Wo = np.random.randn(hidden_units, num_classes) * np.sqrt(
    2. / hidden_units)  # He初始化  # num_filters通道,10x10 feature map
b_fc1 = np.zeros(hidden_units)
b_fc2 = np.zeros(num_classes)

# ----- 超参数 -----
learning_rate = 0.1
batch_size = 64
epochs = 3

# # ------ 训练循环 ------
for epoch in range(epochs):
    start_time = time.time()
    print(f"Epoch {epoch + 1}")
    for batch_i in range(0, len(train_images), batch_size):

        X_batch = train_images[batch_i:batch_i + batch_size]
        y_batch = train_labels[batch_i:batch_i + batch_size]
        y_batch_onehot = one_hot(y_batch, num_classes)  # 将标签转换为 one-hot 编码
        m = len(X_batch)

        # --- 前向传播 ---
        # 1. 卷积层
        conv_out, conv_cache = conv_layer(X_batch,
                                          W1)  # conv_out will batch x filter_num x feature_map_h x feature_map_w
        # 2. 激活层
        relu_out = relu(conv_out)
        relu_cache = conv_out
        # 3. 池化层
        pooled = mean_pooling(relu_out, pool_size=2)
        pool_shape = relu_out.shape
        # 4. 展平
        flattened = pooled.reshape(m, -1)

        # 5. 全连接层
        hidden_input = np.dot(flattened, W5) + b_fc1
        hidden_output = relu(hidden_input)
        hidden_cache = hidden_input

        # 6. 输出层
        logits = np.dot(hidden_output, Wo) + b_fc2
        probs = softmax(logits)

        # 7. 计算损失
        loss = cross_entropy_loss(probs, y_batch_onehot)
        if batch_i % 1024 == 0:
            print(f"Batch {batch_i // batch_size}, Loss: {loss:.4f}")

        # --- 反向传播 ---
        # 1. 输出层
        dlogits = probs - y_batch_onehot  # Softmax + Cross-entropy 的简化形式
        dWo = np.dot(hidden_output.T, dlogits) / m  # 除以批大小
        db_fc2 = np.sum(dlogits, axis=0) / m
        dhidden_output = np.dot(dlogits, Wo.T)

        # 2. 全连接隐藏层
        dhidden_input = relu_backward(dhidden_output, hidden_cache)
        dW5 = np.dot(flattened.T, dhidden_input) / m
        db_fc1 = np.sum(dhidden_input, axis=0) / m
        dflattened = np.dot(dhidden_input, W5.T)

        # 3. 反池化层
        dpooled = dflattened.reshape(pooled.shape)  # 恢复 pooled 张量的形状
        drelu_out = mean_pooling_backward(dpooled, relu_out, pool_size=2)

        # 4. ReLU 激活层
        dconv_out = relu_backward(drelu_out, relu_cache)

        # 5. 卷积层
        dX_batch, dW1 = conv_layer_backward(dconv_out, W1, conv_cache, X_batch.shape[1:])  # 确保传递所有必需的参数

        # 6. 参数更新
        Wo -= learning_rate * dWo
        b_fc2 -= learning_rate * db_fc2
        W5 -= learning_rate * dW5
        b_fc1 -= learning_rate * db_fc1
        W1 -= learning_rate * dW1
    end_time = time.time()
    print('一次训练耗时:', end_time - start_time)
# 评估
predictions = predict(test_images)
accuracy = np.mean(predictions == test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
