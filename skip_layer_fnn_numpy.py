import numpy as np
import tensorflow as tf
import pickle
from dca_numba_cpu import DCA
import sys

def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        has_bias = layer["bias"]

        params_values["W" + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        if has_bias: params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values

# activation function
# -------------------
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def loss(Y_hat, Y):
    return -np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))

def loss_backward(Y_hat, Y):
    return -(np.divide(Y, Y_hat) - np.divide(1-Y, 1-Y_hat))

# def loss(Y_hat, Y):
#     return np.sum((Y_hat - Y)**2)

# def loss_backward(Y_hat, Y):
#     return 2*(Y_hat - Y)

# -------------------

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu", has_bias=True, S_curr=0):
    Z_curr = W_curr.dot(A_prev) + (b_curr if has_bias else 0)
    # skip layer connection
    Z_curr += S_curr

    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception("Non-supported activation function")

    A = activation_func(Z_curr)
    return A, Z_curr

def compute_skip_layer_value(skip_layer, memory, size, layer_idx):
    S = np.zeros(size)
    for k in range(layer_idx-1):
        for i,j in skip_layer.get((k, layer_idx),[]):
            A_k = memory["A" + str(k)]
            S[j,:] += A_k[i,:]
    return S

def full_forward_propagation(X, params_values, nn_architecture, skip_layer=None):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        active_function_curr = layer["activation"]
        has_bias = layer["bias"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)] if has_bias else 0
        size = (W_curr.shape[0], A_prev.shape[1])
        S_curr = compute_skip_layer_value(skip_layer, memory, size, layer_idx) if skip_layer is not None else 0
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, active_function_curr, has_bias, S_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu", has_bias=True, dS_curr=0):
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception("Non-supported activation function")

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    if has_bias:
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    else:
        db_curr = 0
    dA_prev = np.dot(W_curr.T, dZ_curr)
    # skip layer connection
    dA_prev += dS_curr

    return dA_prev, dW_curr, db_curr, dZ_curr

def compute_skip_layer_dW(skip_layer, grads_values, size, layer_idx_curr, last_layer_idx):
    dS = np.zeros(size)
    for k in range(layer_idx_curr+2, last_layer_idx+1):
        for i,j in skip_layer.get((layer_idx_curr, k), []):
            dZ = grads_values["dZ" + str(k)]
            dS[i,:] += dZ[j,:]
    return dS

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, skip_layer=None):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = loss_backward(Y_hat, Y)
    last_layer_idx = len(nn_architecture) + 1

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        active_function_curr = layer["activation"]
        has_bias = layer["bias"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)] if has_bias else 0
        dS_curr = compute_skip_layer_dW(skip_layer, grads_values, A_prev.shape, layer_idx_curr, last_layer_idx) if skip_layer is not None else 0
        dA_prev, dW_curr, db_curr, dZ_curr = single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, active_function_curr, has_bias, dS_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["dZ" + str(layer_idx_curr)] = dZ_curr
        if has_bias:
            grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        has_bias = layer["bias"]
        layer_idx += 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        if has_bias:
            params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def get_accuracy_value(Y_hat, Y):
    Y_hat = np.array(Y_hat > 0.5, dtype=int)
    tp = np.sum(Y_hat == Y)
    m = Y.shape[1]
    return tp / m

def train(X, Y, nn_architecture, epochs, learning_rate, skip_layer=None, params_values=None, cost_history=[], accuracy_history=[]):
    if params_values is None:
        params_values = init_layers(nn_architecture, 2)
    
    for i in range(epochs):
        Y_hat, memory = full_forward_propagation(X, params_values, nn_architecture, skip_layer)
        cost = loss(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, skip_layer)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        print('[*] Epoch: %d, loss: %.3f, acc: %.3f' % (i+1, cost, accuracy))

    return params_values, cost_history, accuracy_history

def dca_process(params_values, x, nn_architecture, top_n):
    y, memory = full_forward_propagation(x, params_values, nn_architecture)
    
    node_dca = np.concatenate([memory["A" + str(i)] for i in range(len(nn_architecture))] + [y]).T
    k = 5
    for i in range(k, 0, -1):
        node_dca[((i - 1) / k <= node_dca) * (node_dca < i / k)] = i
    node_dca = node_dca.astype(int)
    
    np.save('node_dca.npy', node_dca)
    DCA('node_dca.npy', 'node_dca.di')
    
    layer_node_num = [nn_architecture[0]['input_dim']]
    for layer in nn_architecture:
        layer_node_num.append(layer_node_num[-1] + layer['output_dim'])
    
    di_arr = []
    with open('node_dca.di', 'r') as f:
        for line in f:
            a, b, _, di = line.strip().split(' ')
            a = int(a)
            b = int(b)
            di = float(di)
            di_arr.append([a, b, di])
    di_arr.sort(key=lambda x: x[-1], reverse=True)
    
    n = 0
    skip_layer_connects = {}
    for a, b, di in di_arr:
        if n >= top_n: break
        
        for i in range(len(layer_node_num)):
            if a <= layer_node_num[i]:
                L1 = i
                break
        
        for i in range(len(layer_node_num)):
            if b <= layer_node_num[i]:
                L2 = i
                break
        
        if L2 - L1 >= 2:
            N1 = a - 1 - (layer_node_num[L1-1] if L1 > 0 else 0)
            N2 = b - 1 - (layer_node_num[L2-1] if L2 > 0 else 0)
            if skip_layer_connects.get((L1, L2), []):
                skip_layer_connects[(L1, L2)].append((N1, N2))
            else:
                skip_layer_connects[(L1, L2)] = [(N1, N2)]
            n += 1

    return skip_layer_connects

def main(pre_train_epoch=5, top_n=100):
    with open('../train_x_25.pkl', 'rb') as f:
        train_x_25 = pickle.load(f)
    with open('../train_y_25.pkl', 'rb') as f:
        train_y_25 = pickle.load(f)
    with open('../test_x_25.pkl', 'rb') as f:
        test_x_25 = pickle.load(f)
    with open('../test_y_25.pkl', 'rb') as f:
        test_y_25 = pickle.load(f)

    learning_rate = 0.1
    nn_architecture = [
        {"input_dim": 784, "output_dim": 400, "activation": "sigmoid", "bias": False},
        {"input_dim": 400, "output_dim": 200, "activation": "sigmoid", "bias": False},
        {"input_dim": 200, "output_dim": 100, "activation": "sigmoid", "bias": False},
        {"input_dim": 100, "output_dim": 1, "activation": "sigmoid", "bias": False}
    ]

    skip_train_results = []
    max_train_sample = 4000
    max_train_epoch = 300
    epoch_n = 5

    for sample in range(100, max_train_sample+1, 100):
        params_values = init_layers(nn_architecture)

        for epoch in range(epoch_n, pre_train_epoch+1, epoch_n):
            params_values, cost_history, accuracy_history = train(train_x_25[:, :sample], train_y_25[:, :sample], nn_architecture, epoch_n, learning_rate, params_values=params_values)
            
            # test
            Y_hat, _ = full_forward_propagation(test_x_25, params_values, nn_architecture)
            cost = loss(Y_hat, test_y_25)
            accuracy = get_accuracy_value(Y_hat, test_y_25)
            skip_train_results.append([sample, epoch, accuracy, cost_history[-1], accuracy_history[-1]])
        
        # DCA
        skip_layer = dca_process(params_values, train_x_25[:, :1000], nn_architecture, top_n)
        
        for epoch in range(pre_train_epoch+5, max_train_epoch+1, epoch_n):
            print("[*] sample: %d, epoch: %d" % (sample, epoch))
            params_values, cost_history, accuracy_history = train(train_x_25[:, :sample], train_y_25[:, :sample], nn_architecture, epoch_n, learning_rate, skip_layer=skip_layer, params_values=params_values)
            
            # test
            Y_hat, _ = full_forward_propagation(test_x_25, params_values, nn_architecture, skip_layer=skip_layer)
            cost = loss(Y_hat, test_y_25)
            accuracy = get_accuracy_value(Y_hat, test_y_25)
            
            skip_train_results.append([sample, epoch, accuracy, cost_history[-1], accuracy_history[-1]])

    with open('skip_train_results_pe%d_n%d.pkl' % (pre_train_epoch, top_n), 'wb') as f:
        pickle.dump(skip_train_results, f)

if __name__ == '__main__':
    pre_train_epoch = int(sys.argv[1])
    top_n = int(sys.argv[2])
    main(pre_train_epoch, top_n)
