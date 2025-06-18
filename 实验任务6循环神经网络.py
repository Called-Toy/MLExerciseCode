import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Example 1523 RNN
temp = [50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 53, 51, 52, 51, 52, 52, 52, 53, 52, 53, 52, 52,
        53, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53.5, 54, 54, 53.5,
        53, 53, 54, 53, 54, 53, 54]
time = np.arange(30, 1530, 30)
scaler = MinMaxScaler(feature_range=(0, 1))
temp_scaled = scaler.fit_transform(np.array(temp).reshape(-1, 1)).flatten()  # 50*1
binary_dim = 5
iterations = 1000
alpha = 0.1
input_dim = 1
hidden_dim = 10
output_dim = 1
all_errors = []

# initialize neural network weights
synapse_0 = 2 * np.random.rand(input_dim, hidden_dim) - 1  # 1*10
synapse_1 = 2 * np.random.rand(hidden_dim, output_dim) - 1  # 10*1
synapse_h = 2 * np.random.rand(hidden_dim, hidden_dim) - 1  # 10*10
synapse_0_update = np.zeros_like(synapse_0)  # 1*10
synapse_1_update = np.zeros_like(synapse_1)  # 10*1
synapse_h_update = np.zeros_like(synapse_h)  # 10*10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


for j in range(iterations):  # train iteration
    overall_error = 0
    for i in range(int(len(temp) * 0.8) - binary_dim):
        a = temp_scaled[i:i + binary_dim]
        b = temp_scaled[i + 1:i + binary_dim + 1]
        c = np.zeros_like(b) # 5*1
        layer_2_deltas = []
        layer_1_values = [np.zeros((1, hidden_dim))]
        for position in range(binary_dim):  # forward
            X = np.array([a[binary_dim - 1 - position]])  # X input
            y = np.array([b[binary_dim - 1 - position]]).T  # y label
            layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
            layer_2 = sigmoid(np.dot(layer_1, synapse_1))
            layer_2_error = y - layer_2  # calculate error
            layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
            overall_error += np.abs(layer_2_error[0])
            layer_1_values.append(layer_1)
            c[binary_dim - 1 - position] = layer_2[0][0]  # Changed this line

        future_layer_1_delta = np.zeros((1, hidden_dim))  # hidden layer diff
        for position in range(binary_dim):  # backforward
            X = np.array([a[position]])
            layer_1 = layer_1_values[-1 - position]
            prev_layer_1 = layer_1_values[-2 - position]
            layer_2_delta = layer_2_deltas[-1 - position]  # error at output layer
            layer_1_delta = (np.dot(future_layer_1_delta, synapse_h.T) +
                             np.dot(layer_2_delta, synapse_1.T)) * sigmoid_output_to_derivative(
                layer_1)  # error at hidden layer
            # update all weights
            synapse_1_update += np.dot(layer_1.T, layer_2_delta)
            synapse_h_update += np.dot(prev_layer_1.T, layer_1_delta)
            synapse_0_update += np.dot(X.T, layer_1_delta)
            future_layer_1_delta = layer_1_delta

        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

    all_errors.append(overall_error)

plt.figure(figsize=(12, 9))
plt.plot(all_errors)
plt.xlabel('Epochs', fontname='SimSun', fontsize=20)
plt.ylabel('Error', fontname='SimSun', fontsize=20)
plt.xticks(fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.tight_layout()
plt.show()

# test
pred = np.zeros_like(temp_scaled)
test_indices = np.arange(int(len(temp) * 0.8) * 30, 1471, 30)
for h in range(int(len(temp) * 0.8) - 1, len(temp) - binary_dim - 1):
    at = temp_scaled[h:h + binary_dim]
    bt = temp_scaled[h + 1:h + binary_dim + 1]
    ct = np.zeros_like(bt)
    layer_1_values = [np.zeros((1, hidden_dim))]
    for position in range(binary_dim):
        X = np.array([at[binary_dim - 1 - position]])  # X input
        y = np.array([bt[binary_dim - 1 - position]]).T  # y label
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        layer_1_values.append(layer_1)
        ct[binary_dim - 1 - position] = layer_2[0][0]  # Changed this line
    pred[h + 1:h + binary_dim + 1] = ct

pred_temp = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
plt.figure(figsize=(12, 9))
plt.plot(time, temp, 'b-*', label='TrueValue')
plt.plot(test_indices, pred_temp[int(len(temp) * 0.8):], 'r-o', label='Prediction')
plt.xlabel('Time(30s)', fontname='SimSun', fontsize=18)
plt.ylabel('Temperature(C)', fontname='SimSun', fontsize=18)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.ylim([48, 55])
plt.legend(prop={'family': 'SimSun', 'size': 6})
plt.tight_layout()
plt.show()

YTest = temp[40:50]
YPred = pred_temp[int(len(temp) * 0.8):]
plt.figure(figsize=(12, 18))
plt.subplot(2, 1, 1)
plt.plot(YTest, 'b-*', label='TrueValue')
plt.plot(YPred, 'r-o', label='Prediction')
plt.ylabel('Temperature', fontname='SimSun', fontsize=18)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.ylim([52, 55])
plt.legend(prop={'family': 'SimSun', 'size': 6})

plt.subplot(2, 1, 2)
error = YPred - YTest
plt.stem(error, 'g-d')
rmse = np.sqrt(np.mean(error ** 2))
plt.ylabel('Error', fontname='SimSun', fontsize=18)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.title(f'RMSE={rmse:.4f}', fontname='SimSun', fontsize=18)
plt.ylim([-2, 2])
plt.tight_layout()
plt.show()

