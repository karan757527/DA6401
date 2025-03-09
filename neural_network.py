from derivatives import Derivatives
from loss_functions import Loss_Function
from activations import Activation_Functions

import numpy as np
import wandb

#datasets
from keras.datasets import fashion_mnist
from keras.datasets import mnist

from sklearn.model_selection import train_test_split


class Neural_Network:
    def __init__(self, config, log=1, console=1):
        self.weights, self.biases, self.a, self.h = {}, {}, {}, {}
        self.grad_weights, self.grad_biases, self.m_weights, self.m_biases = {}, {}, {}, {}
        self.v_weights, self.v_biases = {}, {}

        self.activation_function = config["activation_function"]
        self.loss_function = config["loss_function"]
        self.initialization = config["init"]
        self.hidden_layers = config["hidden_layers"]
        self.hidden_layer_sizes = config["hidden_layer_sizes"]
        self.dataset = config["dataset"]
        self.wan_log, self.console_log = log, console
        self.loss = Loss_Function()
        self.act = Activation_Functions()
        self.derivative = Derivatives()

        (train_img, train_lbl), (test_img, test_lbl) = self.load_dataset()
        train_img, val_img, train_lbl, val_lbl = train_test_split(train_img, train_lbl, test_size=0.1, random_state=41)

        self.input, self.y_true = self.preprocess_data(train_img, train_lbl)  # preprocess training data
        self.val_img, self.val_true = self.preprocess_data(val_img, val_lbl)  # preprocess validation data
        self.test_img, self.test_true = self.preprocess_data(test_img, test_lbl)  # preprocess test data
        self.layers = [self.input.shape[1]] + [self.hidden_layer_sizes] * self.hidden_layers + [10]  
        
        self.initialize_parameters()  # initialize weights and biases

    def load_dataset(self):
        if self.dataset == 'fashion_mnist':
            return fashion_mnist.load_data()
        if self.dataset == 'mnist':
            return mnist.load_data()
        else:
            raise ValueError("Unknown dataset")

    def preprocess_data(self, images, labels):
        return images.reshape(images.shape[0], -1) / 255.0 , labels  # normalize and reshape data

    def initialize_parameters(self):
        for layer in range(1, len(self.layers)):
            self.m_weights[layer] = np.zeros((self.layers[layer-1], self.layers[layer]))  # initialize momentum for weights
            self.m_biases[layer] = np.zeros((1, self.layers[layer]))  # initialize momentum for biases
            self.v_weights[layer] = np.zeros((self.layers[layer-1], self.layers[layer]))  # initialize velocity for weights
            self.v_biases[layer] = np.zeros((1, self.layers[layer]))  # initialize velocity for biases

            if self.initialization == "random":
                self.weights[layer] = np.random.randn(self.layers[layer-1], self.layers[layer])  # random initialization
                self.biases[layer] = np.random.randn(1, self.layers[layer])
            elif self.initialization == "Xavier":
                variance_w = 6.0 / (self.layers[layer-1] + self.layers[layer])  # Xavier initialization for weights
                variance_b = 6.0 / (1 + self.layers[layer])  # Xavier initialization for biases
                self.weights[layer] = np.random.randn(self.layers[layer-1], self.layers[layer]) * np.sqrt(variance_w)
                self.biases[layer] = np.random.randn(1, self.layers[layer]) * np.sqrt(variance_b)

    def forward_propagation(self, x):
        self.h[0] = x
        for layer in range(1, len(self.layers)-1):
            self.a[layer] = np.dot(self.h[layer-1], self.weights[layer]) + self.biases[layer]  # linear transformation
            self.h[layer] = self.act.activation(self.a[layer], self.activation_function)  # apply activation
        self.a[layer+1] = np.dot(self.h[layer], self.weights[layer+1]) + self.biases[layer+1]
        self.h[layer+1] = self.act.activation(self.a[layer+1], "softmax")  # softmax for output layer
        return self.h[layer+1]

    def backward_propagation(self, x, y_true, y_hat):
        activation_derivative = self.derivative.derivatives(self.a[len(self.layers) - 1], "softmax")  
        error_wrt_output = self.loss.last_output_derivative(y_hat, y_true, activation_derivative, self.loss_function)  # error at output

        for layer in range(len(self.layers)-1, 1, -1):
            self.grad_weights[layer] = np.dot(self.h[layer-1].T, error_wrt_output)  # gradient for weights
            self.grad_biases[layer] = np.sum(error_wrt_output, axis=0, keepdims=True)  # gradient for biases
            error_wrt_hidden = np.dot(error_wrt_output, self.weights[layer].T)  
            error_wrt_output = error_wrt_hidden * self.derivative.derivatives(self.a[layer-1], self.activation_function)  

        self.grad_weights[1] = np.dot(x.T, error_wrt_output)  
        self.grad_biases[1] = np.sum(error_wrt_output, axis=0, keepdims=True)  

    def one_hot_matrix(self, labels):
        mat = np.zeros((labels.shape[0], 10)) 
        mat[np.arange(labels.shape[0]), labels] = 1
        return mat

    def compute_performance(self, data, labels):
        y_pred = self.forward_propagation(data)  # forward pass
        one_hot_labels = self.one_hot_matrix(labels)  # convert labels to one-hot
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(one_hot_labels, axis=1)) * 100 
        loss = self.loss.compute_loss(one_hot_labels, y_pred, self.loss_function)  
        return loss, accuracy

    
    def probability(self, data):
        return self.forward_propagation(data)  

        
    def fit(self, batch_size, epochs, optimizer):
        total_batches = int(np.ceil(self.input.shape[0] / batch_size))  # total batches
        for epoch in range(epochs):
            t = 1
            for batch in range(total_batches):
                batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size 
                image_set, label_set = self.input[batch_start:batch_end], self.y_true[batch_start:batch_end] 
                y_hat = self.forward_propagation(image_set)  # forward pass
                self.backward_propagation(image_set, self.one_hot_matrix(label_set), y_hat)  # backward pass
                for layer in range(1, len(self.layers)):
                    self.grad_weights[layer] /= batch_size  # normalize gradients
                    self.grad_biases[layer] /= batch_size
                optimizer.update_parameters(t)  # updated parameters
                t += 1
            
            t_loss, t_acc = self.compute_performance(self.input, self.y_true)  # train performance
            v_loss, v_acc = self.compute_performance(self.val_img, self.val_true)  # validation performance
            
            if self.wan_log:
                wandb.log({'epoch': epoch + 1, 'train_loss': t_loss, 'train_acc': t_acc, 'val_loss': v_loss, 'val_acc': v_acc})  # log to wandb
            if self.console_log:
                print(f"Epoch {epoch+1}: Train Loss={t_loss:.4f}, Train Acc={t_acc:.2f}%, Val Loss={v_loss:.4f}, Val Acc={v_acc:.2f}%")  # console log
        
        return t_loss, t_acc, v_loss, v_acc 