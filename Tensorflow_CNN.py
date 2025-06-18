import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from scipy.io import loadmat

# 1. 加载数据 (使用你提供的加载方式)
images_data = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\Images.mat")
labels_data = loadmat(r"C:\Users\唐涛\Desktop\machine learning database\Labels.mat")
images = images_data['Images']  # 28*28*10000
labels = labels_data['Labels']  # 10000x1

# 2. 训练测试划分 (使用你提供的划分方式)
train_images = images[:, :, :8000]  # 28*28*8000
train_labels = labels[:8000]  # 8000x1
test_images = images[:, :, 8000:]  # 28*28*2000
test_labels = labels[8000:]  # 2000x1

# 3. 数据预处理与形状调整 (使用你提供的处理方式)
train_images = np.transpose(train_images, (2, 0, 1))  # 变为 (8000, 28, 28)
test_images = np.transpose(test_images, (2, 0, 1))  # 变为 (2000, 28, 28)

# 调整标签形状并转换为 integers (使用你提供的处理方式)
train_labels = train_labels.flatten().astype(int) - 1
test_labels = test_labels.flatten().astype(int) - 1

# 归一化图像数据 (使用你提供的处理方式)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 为了与 Conv2D 层输入要求匹配，需要将图像数据 reshape为 (batch_size, height, width, channels) 的格式
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 将标签转换为 one-hot 编码
num_classes = len(np.unique(train_labels)) # 根据你的数据自动确定类别数量
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# 4. 构建卷积神经网络 (特征提取子网络)
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=20, kernel_size=(9, 9), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    return model

# 5. 构建分类子网络 (单层全连接神经网络)
def create_classification_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,))) # 隐含层
    model.add(layers.Dense(num_classes, activation='softmax')) # 输出层
    return model

# 6. 组合 CNN 和分类子网络
def create_full_cnn_model(num_classes):
    cnn_model = create_cnn_model()
    classification_model = create_classification_model(input_shape=cnn_model.output_shape[1], num_classes=num_classes)
    full_model = models.Sequential([
        cnn_model,
        classification_model
    ])
    return full_model

# 7. 创建并编译模型
model = create_full_cnn_model(num_classes)

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 8. 训练模型
epochs = 10 # 可以根据需要调整训练轮数
batch_size = 32 # 可以根据需要调整批次大小

model.fit(train_images, train_labels_one_hot, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# 9. 评估模型
loss, accuracy = model.evaluate(test_images, test_labels_one_hot, verbose=0)
print(f"测试集损失: {loss:.4f}")
print(f"测试集准确率: {accuracy:.4f}")

# 查看模型结构
model.summary()