import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.datasets import mnist

wandb.login(key="843913992a9025996973825be4ad46e4636d0610")

PROJECT = "DA6401Assignment1"
ENTITY = "karan"
SWEEPCOUNT = 10

(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()
target_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class Activation_Functions:
    def __init__(self):
        self.activation_functions = {
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "ReLU": self.ReLU,
            "softmax": self.softmax,
            "identity": self.identity
        }

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax probabilities
        
    def sigmoid(self, x):
        result = np.zeros_like(x)
        result[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))  # for positive values
        result[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))  # for negative values
        return result

    def ReLU(self, x):
        return np.maximum(0, x)  # ReLU activation

    def tanh(self, x):
        return np.tanh(x)  # hyperbolic tangent activation

    def identity(self, x):
        return x  # identity function

    def activation(self, x, fun):
        activation_function = self.activation_functions.get(fun)
        if activation_function:
            return activation_function(x)
        else:
            raise ValueError(f"Activation function '{fun}' is not supported.")


class Loss_Function:
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred, loss_function="cross_entropy"):
        if loss_function == "cross_entropy":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip to avoid log(0)
            return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))  # cross-entropy loss
        if loss_function == "mean_squared_error":
            return 0.5 * np.mean(np.sum((y_true - y_pred) ** 2, axis=1))  # MSE loss

    def last_output_derivative(self, y_pred, y_true, activation_derivative, loss_function="cross_entropy"):
        if loss_function == "mean_squared_error":
            return (y_pred - y_true) * activation_derivative / len(y_true)  # MSE derivative
        if loss_function == "cross_entropy":
            return -(y_true - y_pred)  # cross-entropy derivative


class Derivatives:
    def __init__(self):
        self.activation = Activation_Functions()
        self.derivative_functions = {
            "sigmoid": self.sigmoid_derivative,
            "tanh": self.tanh_derivative,
            "ReLU": self.ReLU_derivative,
            "softmax": self.softmax_derivative,
            "identity": self.identity_derivative
        }

    def sigmoid_derivative(self, x):
        k = self.activation.sigmoid(x)
        return k * (1 - k)  # derivative of sigmoid

    def softmax_derivative(self, x):
        k = self.activation.softmax(x)
        return k * (1 - k)  # derivative of softmax

    def tanh_derivative(self, x):
        k = self.activation.tanh(x)
        return 1 - k ** 2  # derivative of tanh

    def ReLU_derivative(self, x):
        x[x > 0] = 1  # derivative is 1 for x > 0
        x[x <= 0] = 0  # derivative is 0 for x <= 0
        return x

    def identity_derivative(self, x):
        return np.ones_like(x)  # derivative of identity is 1

    def derivatives(self, x, activation_function):
        derivative_function = self.derivative_functions.get(activation_function)
        if derivative_function:
            return derivative_function(x)
        else:
            raise ValueError(f"Derivative for activation function '{activation_function}' is not supported.")

class Optimizer:
    def __init__(self, model, config):
        self.model = model  
        self.learning_rate = config["eta"]  
        self.decay = config["weight_decay"]  
        self.optim_type = config["optimizer"]  
        self.momentum = config["momentum"] 
        self.beta1 = config["beta1"] 
        self.beta2 = config["beta2"]  
        self.epsilon_val = config["epsilon"]
        self.optimization_methods = {
            "sgd": self.stochastic_gradient_descent,
            "momentum": self.momentum_gradient_descent,
            "nesterov": self.nesterov_gradient_descent,
            "nag": self.nesterov_gradient_descent,
            "rmsprop": self.rmsprop,
            "adam": self.adam,
            "nadam": self.nadam
        }

    def update_parameters(self, timestep):
        optimization_function = self.optimization_methods.get(self.optim_type)
        
        if optimization_function:
            if self.optim_type in ["adam", "nadam"]:
                optimization_function(timestep)
            else:
                optimization_function()
        else:
            raise ValueError(f"Optimization method '{self.optim_type}' is not supported.")

    def stochastic_gradient_descent(self):
        param_layers = self.model.weights.keys()
        for l in param_layers:
            reg_term = self.model.weights[l] * self.decay  # weight decay
            self.model.grad_weights[l] += reg_term
            bias_update = self.learning_rate * self.model.grad_biases[l]  # bias update
            self.model.biases[l] -= bias_update
            weight_update = self.learning_rate * self.model.grad_weights[l]  # weight update
            self.model.weights[l] -= weight_update

    def momentum_gradient_descent(self):
        for layer in self.model.weights.keys():
            self.model.grad_weights[layer] += self.decay * self.model.weights[layer]  # weight decay
            prev_momentum_w = self.momentum * self.model.m_weights[layer]
            new_momentum_w = prev_momentum_w + self.learning_rate * self.model.grad_weights[layer]  # momentum update
            prev_momentum_b = self.momentum * self.model.m_biases[layer]
            new_momentum_b = prev_momentum_b + self.learning_rate * self.model.grad_biases[layer]
            self.model.m_weights[layer] = new_momentum_w
            self.model.m_biases[layer] = new_momentum_b
            self.model.weights[layer] -= new_momentum_w
            self.model.biases[layer] -= new_momentum_b

    def nesterov_gradient_descent(self):
        for l in self.model.weights.keys():
            momentum_w = self.momentum * self.model.m_weights[l]  # momentum term
            lookahead_w = self.model.weights[l] - momentum_w  # lookahead step
            lookahead_b = self.model.biases[l] - self.momentum * self.model.m_biases[l]
            self.model.grad_weights[l] += lookahead_w * self.decay  # weight decay
            grad_w = self.model.grad_weights[l]
            grad_b = self.model.grad_biases[l]
            new_m_w = self.momentum * self.model.m_weights[l] - self.learning_rate * grad_w  # update momentum
            new_m_b = self.momentum * self.model.m_biases[l] - self.learning_rate * grad_b
            self.model.weights[l] += new_m_w
            self.model.biases[l] += new_m_b
            self.model.m_weights[l] = new_m_w
            self.model.m_biases[l] = new_m_b

    def rmsprop(self):
        for layer in self.model.weights.keys():
            current_grad_w = self.model.grad_weights[layer] + self.model.weights[layer] * self.decay  # weight decay
            current_grad_b = self.model.grad_biases[layer]
            self.model.v_weights[layer] = self.beta2*self.model.v_weights[layer] + (1-self.beta2)*(current_grad_w**2)  # velocity update
            self.model.v_biases[layer] = self.beta2*self.model.v_biases[layer] + (1-self.beta2)*(current_grad_b**2)
            epsilon = 1e-8 if self.epsilon_val < 1e-8 else self.epsilon_val  # numerical stability
            self.model.weights[layer] -= (current_grad_w * self.learning_rate) / (np.sqrt(self.model.v_weights[layer]) + epsilon)  # update weights
            self.model.biases[layer] -= (current_grad_b * self.learning_rate) / (np.sqrt(self.model.v_biases[layer]) + epsilon)

    def adam(self, step):
        for l in self.model.weights.keys():
            decay_contribution = self.decay * self.model.weights[l] # weight decay
            self.model.grad_weights[l] += decay_contribution
            
            grad_w = np.clip(self.model.grad_weights[l], -1e3, 1e3)
            grad_b = np.clip(self.model.grad_biases[l], -1e3, 1e3)
            
            m_w = self.beta1 * self.model.m_weights[l] + (1 - self.beta1) * grad_w # momentum update with numerical stability
            m_b = self.beta1 * self.model.m_biases[l] + (1 - self.beta1) * grad_b
            
            v_w = self.beta2 * self.model.v_weights[l] + (1 - self.beta2) * (grad_w**2 + 1e-8)
            v_b = self.beta2 * self.model.v_biases[l] + (1 - self.beta2) * (grad_b**2 + 1e-8)
            
            self.model.m_weights[l], self.model.v_weights[l] = m_w, v_w
            self.model.m_biases[l], self.model.v_biases[l] = m_b, v_b
            
            beta1_t = max(self.beta1**step, 1e-8)
            beta2_t = max(self.beta2**step, 1e-8)
            mw_corrected = m_w / (1 - beta1_t + 1e-8)
            vw_corrected = v_w / (1 - beta2_t + 1e-8)
            mb_corrected = m_b / (1 - beta1_t + 1e-8)
            vb_corrected = v_b / (1 - beta2_t + 1e-8)
            
            weight_update = self.learning_rate * mw_corrected / (np.sqrt(vw_corrected) + self.epsilon_val)
            bias_update = self.learning_rate * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon_val)
            
            self.model.weights[l] -= np.clip(weight_update, -1e3, 1e3)
            self.model.biases[l] -= np.clip(bias_update, -1e3, 1e3)

    def nadam(self, step):
        for l in self.model.weights.keys():
            self.model.grad_weights[l] += self.model.weights[l] * self.decay
            grad_w = np.clip(self.model.grad_weights[l], -1e3, 1e3)
            grad_b = np.clip(self.model.grad_biases[l], -1e3, 1e3)
            
            m_w_new = self.beta1 * self.model.m_weights[l] + (1-self.beta1)*grad_w
            m_b_new = self.beta1 * self.model.m_biases[l] + (1-self.beta1)*grad_b

            v_w_new = self.beta2 * self.model.v_weights[l] + (1-self.beta2)*(grad_w**2 + 1e-8)
            v_b_new = self.beta2 * self.model.v_biases[l] + (1-self.beta2)*(grad_b**2 + 1e-8)
            
            beta1_t = max(self.beta1**step, 1e-8)
            beta2_t = max(self.beta2**step, 1e-8)
            mw_hat = m_w_new / (1 - beta1_t + 1e-8)
            vw_hat = v_w_new / (1 - beta2_t + 1e-8)
            mb_hat = m_b_new / (1 - beta1_t + 1e-8)
            vb_hat = v_b_new / (1 - beta2_t + 1e-8)
            
            nesterov_w = self.beta1 * mw_hat + (1-self.beta1)*grad_w/(1 - beta1_t + 1e-8)
            nesterov_b = self.beta1 * mb_hat + (1-self.beta1)*grad_b/(1 - beta1_t + 1e-8)
            
            weight_update = self.learning_rate * nesterov_w / (np.sqrt(vw_hat) + self.epsilon_val)
            bias_update = self.learning_rate * nesterov_b / (np.sqrt(vb_hat) + self.epsilon_val)
            
            self.model.weights[l] -= np.clip(weight_update, -1e3, 1e3)
            self.model.biases[l] -= np.clip(bias_update, -1e3, 1e3)
            
            self.model.m_weights[l], self.model.v_weights[l] = m_w_new, v_w_new
            self.model.m_biases[l], self.model.v_biases[l] = m_b_new, v_b_new


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
            return (train_img, train_labels), (test_img, test_labels)
        if self.dataset == 'mnist':
            return (mnist_train, mnist_train_label), (mnist_test, mnist_test_label)
        else:
            raise ValueError("Unknown dataset")

    def preprocess_data(self, images, labels):
        return images.reshape(images.shape[0], -1) / 255.0 , labels  # normalize and reshape data

    def initialize_parameters(self):
        for layer in range(1, len(self.layers)):
            self.v_weights[layer] = np.zeros((self.layers[layer-1], self.layers[layer]))  # initialize velocity for weights
            self.v_biases[layer] = np.zeros((1, self.layers[layer]))  # initialize velocity for biases
            
            self.m_weights[layer] = np.zeros((self.layers[layer-1], self.layers[layer]))  # initialize momentum for weights
            self.m_biases[layer] = np.zeros((1, self.layers[layer]))  # initialize momentum for biases
            
            if self.initialization == "random":
                self.weights[layer] = np.random.normal(0, 1, (self.layers[layer-1], self.layers[layer]))
                self.biases[layer] = np.random.normal(0, 1, (1, self.layers[layer]))
            elif self.initialization == "Xavier":
                scale_w = np.sqrt(2.0 / (self.layers[layer-1] + self.layers[layer]))
                scale_b = np.sqrt(2.0 / (1 + self.layers[layer]))
                self.weights[layer] = np.random.normal(0, scale_w, (self.layers[layer-1], self.layers[layer]))
                self.biases[layer] = np.random.normal(0, scale_b, (1, self.layers[layer]))

    def forward_propagation(self, x):
        self.h[0] = x
        num_layers = len(self.layers) - 1
    
        for layer in range(1, num_layers):
            linear_output = self.h[layer-1] @ self.weights[layer]
            self.a[layer] = linear_output + self.biases[layer]   # linear transformation
            self.h[layer] = self.act.activation(self.a[layer], self.activation_function)
    
        final_linear_output = self.h[num_layers-1] @ self.weights[num_layers]
        self.a[num_layers] = final_linear_output + self.biases[num_layers]
        self.h[num_layers] = self.act.activation(self.a[num_layers], "softmax") # softmax for output layer
    
        output = self.h[num_layers]
        return output

    def backward_propagation(self, x, y_true, y_hat): 
        output_layer_index = len(self.layers) - 1  # softmax derivative for output layer  
        softmax_grad = self.derivative.derivatives(self.a[output_layer_index], "softmax")  
        error = self.loss.last_output_derivative(y_hat, y_true, softmax_grad, self.loss_function)  
    
        # for hidden layers
        for current_layer in reversed(range(2, len(self.layers))):  
            prev_hidden = self.h[current_layer - 1]  
    
            self.grad_weights[current_layer] = prev_hidden.T @ error  
            self.grad_biases[current_layer] = np.sum(error, axis=0, keepdims=True)  
    
            weight_matrix = self.weights[current_layer].T  
            error_hidden = error @ weight_matrix  
            deriv_activation = self.derivative.derivatives(self.a[current_layer - 1], self.activation_function)  
            error = error_hidden * deriv_activation  
    
        input_grad_weights = x.T @ error  
        input_grad_biases = np.sum(error, axis=0, keepdims=True)  
        self.grad_weights[1] = input_grad_weights  
        self.grad_biases[1] = input_grad_biases  

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
        num_samples = self.input.shape[0]  
        num_mini_batches = int(np.ceil(num_samples / batch_size))  # total batches  
    
        for epoch_num in range(epochs):  
            step_counter = 1  
    
            for iteration in range(num_mini_batches):  
                start_idx = iteration * batch_size  
                end_idx = min(start_idx + batch_size, num_samples)  
                batch_images = self.input[start_idx:end_idx]  
                batch_labels = self.y_true[start_idx:end_idx]  
    
                predictions = self.forward_propagation(batch_images)  # forward and backward pass  
                true_labels_one_hot = self.one_hot_matrix(batch_labels)  
                self.backward_propagation(batch_images, true_labels_one_hot, predictions)  
     
                for layer_idx in self.grad_weights.keys():  
                    self.grad_weights[layer_idx] = self.grad_weights[layer_idx] / batch_size  
                    self.grad_biases[layer_idx] = self.grad_biases[layer_idx] / batch_size  
      
                optimizer.update_parameters(step_counter)  
                step_counter = step_counter + 1  
    
            training_loss, training_acc = self.compute_performance(self.input, self.y_true)  
            val_loss, val_acc = self.compute_performance(self.val_img, self.val_true)  
    
            log_data = {  
                'epoch': epoch_num + 1,  
                'train_loss': training_loss,  
                'train_acc': training_acc,  
                'val_loss': val_loss,  
                'val_acc': val_acc  
            }  
    
            if self.wan_log:  
                wandb.log(log_data)  # log to wandb  
    
            if self.console_log:  
                status_msg = (  
                    f"Epoch {epoch_num + 1}: "  
                    f"Train Loss={training_loss:.4f}, "  
                    f"Train Acc={training_acc:.2f}%, "  
                    f"Val Loss={val_loss:.4f}, "  
                    f"Val Acc={val_acc:.2f}%"  
                )  
                print(status_msg)  
    
        return training_loss, training_acc, val_loss, val_acc  

def generate_and_plot_confusion_matrix(y_pred, y_true):  
    
    wandb.init(project=PROJECT, name="Question:7")  
    
    predicted_classes = np.argmax(y_pred, axis=1)   # predicted classes 
    num_labels = len(target_classes)  
    conf_matrix = np.zeros((num_labels, num_labels), dtype=int)  
      
    for idx, actual_label in enumerate(y_true):  
        true_idx = int(actual_label)  
        pred_idx = int(predicted_classes[idx])  
        
        if (true_idx >= 0 and true_idx < num_labels) and (pred_idx >= 0 and pred_idx < num_labels):  
            conf_matrix[true_idx, pred_idx] += 1  
        else:  
            print(f"Skipping invalid index: y_true={true_idx}, y_pred={pred_idx}")  
    
    df_confusion = pd.DataFrame(conf_matrix, index=target_classes, columns=target_classes)  

    
    fig = plt.figure(figsize=(15, 15))  # plot heatmap  
    ax = sns.heatmap(  
        df_confusion,  
        annot=True,  
        fmt='d',  
        cmap="Reds",  
        linewidths=4,  
        linecolor='white'  
    )  
    ax.set_xlabel("Predicted Class")  
    ax.set_ylabel("True Class")  
    ax.set_xticklabels(target_classes, rotation=90)  
    ax.set_yticklabels(target_classes, rotation=0)  
    plt.title('Confusion Matrix', fontsize=12)  
    
    plt.show()  
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       y_true=y_true, 
                                                       preds=predicted_classes,
                                                       class_names=target_classes)})  
    wandb.finish()  

# best config was opt_rmsprop_hl_4_bs_16_e_10_act_ReLU_eta_0.0001_init_Xavier_hls_128_dataset_fashion_mnist_cross_entropy
config_nn = {
        "activation_function": "ReLU",  
        "init": "Xavier",
        "dataset": "fashion_mnist",
        "loss_function": "cross_entropy", 
        "hidden_layers": 4,
        "hidden_layer_sizes": 128
        
}

config_opt = {
    "optimizer": "rmsprop",
    "eta": 0.0001,  
    "beta": 0.9,
    "beta2": 0.999,
    "beta1": 0.9,
    "weight_decay": 0.0005,
    "epsilon": 1e-8,
    "momentum": 0.9
}

nn = Neural_Network(config_nn, log=0, console=1)
optimizer = Optimizer(nn, config_opt)

t_loss, t_acc, v_loss, v_acc = nn.fit(batch_size=16, epochs=10, optimizer=optimizer)
loss, accuracy = nn.compute_performance(nn.test_img, nn.test_true)  # evaluate on test data
print(f"Test Loss: {loss:.4f}, Test Acc: {accuracy:.2f}%")  # print test loss and accuracy
y_pred = nn.probability(nn.test_img)
generate_and_plot_confusion_matrix(y_pred, nn.test_true) # plot confusion matrox