# importing requirements
import numpy as np
import wandb
import argparse


# importing Classes
from optimizers import Optimizer
from neural_network import Neural_Network


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network with configurable hyperparameters.")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_A1", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m021-iit-madras", help="WandB entity")
    
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="rmsprop", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
    
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function")
    
    return parser.parse_args()

def main():
    args = parse_args()

    with open("wandb_key.txt", "r") as f:
        wandb.login(key=f.read().strip())
    
    print("wandb logged in")
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    wandb.run.name = f"opt_{args.optimizer}_hl_{args.num_layers}_bs_{args.batch_size}_e_{args.epochs}_act_{args.activation}_eta_{args.learning_rate}_init_{args.weight_init}_hls_{args.hidden_size}_dataset_{args.dataset}_{args.loss}"
    

    config_nn = {
        "hidden_layers": args.num_layers,
        "hidden_layer_sizes": args.hidden_size,
        "activation_function": args.activation,
        "loss_function": args.loss,
        "init": args.weight_init,
        "dataset": args.dataset
    }
    
    config_opt = {
        "eta": args.learning_rate,
        "optimizer": args.optimizer,
        "beta": args.beta,
        "weight_decay": args.weight_decay,
        "epsilon": args.epsilon,
        "beta2": args.beta2,
        "beta1": args.beta1,
        "momentum": args.momentum
    }

    nn = Neural_Network(config_nn,log = 1,console=1)
    opt = Optimizer(nn, config_opt)
    

    t_loss, t_acc, v_loss, v_acc = nn.fit(batch_size=args.batch_size, epochs=args.epochs, optimizer=opt)
    
    loss, accuracy = nn.compute_performance(nn.test_img, nn.test_true)  # evaluate on test data
    print(f"Test Loss: {loss:.4f}, Test Acc: {accuracy:.2f}%")  # print test loss and accuracy
    wandb.log({"test_accuracy": accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()

