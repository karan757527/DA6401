import numpy as np

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

    def update_parameters(self, timestep):
        if self.optim_type == "sgd":
            self.stochastic_gradient_descent()
        elif self.optim_type == "momentum":
            self.momentum_gradient_descent()
        elif self.optim_type == "nesterov" or self.optim_type == "nag" :
            self.nesterov_gradient_descent()
        elif self.optim_type == "rmsprop":
            self.rmsprop()
        elif self.optim_type == "adam":
            self.adam(timestep)
        elif self.optim_type == "nadam":
            self.nadam(timestep)

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
            decay_contribution = self.decay * self.model.weights[l]  # weight decay
            self.model.grad_weights[l] += decay_contribution
            m_w = self.beta1 * self.model.m_weights[l] + (1-self.beta1)*self.model.grad_weights[l]  # momentum update
            m_b = self.beta1 * self.model.m_biases[l] + (1-self.beta1)*self.model.grad_biases[l]
            v_w = self.beta2 * self.model.v_weights[l] + (1-self.beta2)*(self.model.grad_weights[l]**2)  # velocity update
            v_b = self.beta2 * self.model.v_biases[l] + (1-self.beta2)*(self.model.grad_biases[l]**2)
            self.model.m_weights[l], self.model.v_weights[l] = m_w, v_w
            self.model.m_biases[l], self.model.v_biases[l] = m_b, v_b
            mw_corrected = m_w / (1 - self.beta1**step)  # bias correction
            vw_corrected = v_w / (1 - self.beta2**step)
            mb_corrected = m_b / (1 - self.beta1**step)
            vb_corrected = v_b / (1 - self.beta2**step)
            self.model.weights[l] -= self.learning_rate * mw_corrected / (np.sqrt(vw_corrected) + self.epsilon_val)  # update weights
            self.model.biases[l] -= self.learning_rate * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon_val)

    def nadam(self, step):
        for l in self.model.weights.keys():
            self.model.grad_weights[l] += self.model.weights[l] * self.decay  # weight decay
            m_w_new = self.beta1 * self.model.m_weights[l] + (1-self.beta1)*self.model.grad_weights[l]  # momentum update
            m_b_new = self.beta1 * self.model.m_biases[l] + (1-self.beta1)*self.model.grad_biases[l]
            v_w_new = self.beta2 * self.model.v_weights[l] + (1-self.beta2)*(self.model.grad_weights[l]**2)  # velocity update
            v_b_new = self.beta2 * self.model.v_biases[l] + (1-self.beta2)*(self.model.grad_biases[l]**2)
            mw_hat = m_w_new / (1 - self.beta1**step)  # bias correction
            vw_hat = v_w_new / (1 - self.beta2**step)
            mb_hat = m_b_new / (1 - self.beta1**step)
            vb_hat = v_b_new / (1 - self.beta2**step)
            nesterov_w = self.beta1 * mw_hat + (1-self.beta1)*self.model.grad_weights[l]/(1-self.beta1**step)  # Nesterov component
            nesterov_b = self.beta1 * mb_hat + (1-self.beta1)*self.model.grad_biases[l]/(1-self.beta1**step)
            self.model.weights[l] -= self.learning_rate * nesterov_w / (np.sqrt(vw_hat) + self.epsilon_val)  # update weights
            self.model.biases[l] -= self.learning_rate * nesterov_b / (np.sqrt(vb_hat) + self.epsilon_val)
            self.model.m_weights[l], self.model.v_weights[l] = m_w_new, v_w_new
            self.model.m_biases[l], self.model.v_biases[l] = m_b_new, v_b_new