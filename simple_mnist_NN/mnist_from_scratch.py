import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32') / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype('int')
    return X, y

# Function to one-hot encode the labels
def one_hot_encode(y, num_classes=10):
    """Convert numeric labels to one-hot encoded vectors"""
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

# Activation functions and their derivatives
def sigmoid(z):
    """Sigmoid activation function: f(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def sigmoid_derivative(a):
    """Derivative of sigmoid: f'(z) = f(z) * (1 - f(z))"""
    return a * (1 - a)

def relu(z):
    """ReLU activation function: f(z) = max(0, z)"""
    return np.maximum(0, z)

def relu_derivative(a):
    """Derivative of ReLU: f'(z) = 1 if z > 0 else 0"""
    return (a > 0).astype(float)

def softmax(z):
    """Softmax activation function: f(z)_i = e^(z_i) / sum(e^(z_j))"""
    # Shift z for numerical stability
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, layer_dims, activations):
        """
        Initialize neural network with specified layer dimensions and activations
        
        Parameters:
        layer_dims -- array of integers, where each element represents the number of units in a layer
        activations -- array of strings, either 'relu', 'sigmoid', or 'softmax' for each layer
        """
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of layers excluding input layer
        self.activations = activations
        self.parameters = {}
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """
        Initialize weights and biases for all layers
        
        He initialization for ReLU layers: W ~ N(0, sqrt(2/n_in))
        Xavier initialization for Sigmoid layers: W ~ N(0, sqrt(1/n_in))
        """
        np.random.seed(42)
        
        for l in range(1, self.L + 1):
            n_in = self.layer_dims[l-1]
            n_out = self.layer_dims[l]
            
            # Choose initialization based on activation
            if self.activations[l-1] == "relu":
                # He initialization
                self.parameters[f"W{l}"] = np.random.randn(n_in, n_out) * np.sqrt(2/n_in)
            else:
                # Xavier initialization
                self.parameters[f"W{l}"] = np.random.randn(n_in, n_out) * np.sqrt(1/n_in)
                
            self.parameters[f"b{l}"] = np.zeros((1, n_out))
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Parameters:
        X -- input data, shape (m, n_x) where m is number of examples and n_x is input size
        
        Returns:
        A_final -- final activation value
        cache -- dictionary containing all activations and linear outputs for backprop
        """
        cache = {}
        A = X
        cache["A0"] = X
        
        for l in range(1, self.L + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            
            # Linear forward: Z = A*W + b
            Z = np.dot(A, W) + b
            cache[f"Z{l}"] = Z
            
            # Activation forward
            if self.activations[l-1] == "sigmoid":
                A = sigmoid(Z)
            elif self.activations[l-1] == "relu":
                A = relu(Z)
            elif self.activations[l-1] == "softmax":
                A = softmax(Z)
            
            cache[f"A{l}"] = A
        
        return A, cache
    
    def compute_cost(self, A_final, Y):
        """
        Compute cross-entropy cost
        
        Parameters:
        A_final -- final layer activation, shape (m, n_y)
        Y -- true labels, shape (m, n_y)
        
        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[0]
        
        # Cross-entropy loss
        log_probs = np.log(A_final + 1e-8)  # Adding small epsilon to avoid log(0)
        cost = -np.sum(Y * log_probs) / m
        
        return cost
    
    def backward_propagation(self, Y, cache):
        """
        Backward propagation to compute gradients
        
        Parameters:
        Y -- true labels, shape (m, n_y)
        cache -- dictionary containing values from forward propagation
        
        Returns:
        gradients -- dictionary containing gradients of weights and biases
        """
        m = Y.shape[0]
        gradients = {}
        
        # Compute gradient for the last layer
        A_final = cache[f"A{self.L}"]
        dA = -(Y / (A_final + 1e-8))  # Derivative of cross-entropy with softmax
        
        for l in reversed(range(1, self.L + 1)):
            A_prev = cache[f"A{l-1}"]
            Z = cache[f"Z{l}"]
            
            if self.activations[l-1] == "sigmoid":
                dZ = dA * sigmoid_derivative(cache[f"A{l}"])
            elif self.activations[l-1] == "relu":
                dZ = dA * relu_derivative(cache[f"A{l}"])
            elif self.activations[l-1] == "softmax" and l == self.L:
                # Shortcut for softmax with cross-entropy: dZ = A - Y
                dZ = A_final - Y
            
            # Calculate gradients
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients
            gradients[f"dW{l}"] = dW
            gradients[f"db{l}"] = db
            
            # Calculate dA for previous layer (if not at the input layer)
            if l > 1:
                dA = np.dot(dZ, self.parameters[f"W{l}"].T)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update parameters using gradient descent
        
        Parameters:
        gradients -- dictionary containing gradients of weights and biases
        learning_rate -- learning rate for gradient descent
        """
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]
    
    def train(self, X, Y, X_val, Y_val, epochs, learning_rate, batch_size=32, print_interval=1):
        """
        Train neural network using mini-batch gradient descent
        
        Parameters:
        X -- training data, shape (m, n_x)
        Y -- training labels, shape (m, n_y)
        X_val -- validation data
        Y_val -- validation labels
        epochs -- number of training epochs
        learning_rate -- learning rate for gradient descent
        batch_size -- size of mini-batches
        print_interval -- how often to print progress
        
        Returns:
        history -- dictionary containing loss and accuracy history
        """
        m = X.shape[0]
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            epoch_cost = 0
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward propagation
                A_final, cache = self.forward_propagation(X_batch)
                
                # Compute cost
                cost = self.compute_cost(A_final, Y_batch)
                epoch_cost += cost * len(X_batch) / m
                
                # Backward propagation
                gradients = self.backward_propagation(Y_batch, cache)
                
                # Update parameters
                self.update_parameters(gradients, learning_rate)
            
            # Compute training and validation metrics
            if epoch % print_interval == 0:
                # Training metrics
                A_train, _ = self.forward_propagation(X)
                train_loss = self.compute_cost(A_train, Y)
                train_accuracy = np.mean(np.argmax(A_train, axis=1) == np.argmax(Y, axis=1))
                
                # Validation metrics
                A_val, _ = self.forward_propagation(X_val)
                val_loss = self.compute_cost(A_val, Y_val)
                val_accuracy = np.mean(np.argmax(A_val, axis=1) == np.argmax(Y_val, axis=1))
                
                # Log metrics
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_accuracy)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                
                print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        X -- input data, shape (m, n_x)
        
        Returns:
        predictions -- predicted labels
        """
        A_final, _ = self.forward_propagation(X)
        predictions = np.argmax(A_final, axis=1)
        return predictions


# Main execution
if __name__ == "__main__":
    # Load MNIST data
    X, y = load_mnist()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode the labels
    Y_train = one_hot_encode(y_train)
    Y_test = one_hot_encode(y_test)
    
    # Define network architecture
    # Input layer: 784 (28x28 images)
    # Hidden layer 1: 128 neurons with ReLU
    # Hidden layer 2: 64 neurons with ReLU
    # Output layer: 10 neurons (0-9 digits) with softmax
    layer_dims = [784, 128, 64, 10]
    activations = ["relu", "relu", "softmax"]
    
    # Create and train neural network
    model = NeuralNetwork(layer_dims, activations)
    
    # Use a smaller subset for quick testing
    sample_size = 5000
    history = model.train(
        X_train[:sample_size], 
        Y_train[:sample_size],
        X_test[:1000], 
        Y_test[:1000], 
        epochs=10, 
        learning_rate=0.1,
        batch_size=32,
        print_interval=1
    )
    
    # Evaluate model on test set
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == np.argmax(Y_test, axis=1))
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    
    # Visualize some predictions
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        sample_idx = np.random.randint(0, len(X_test))
        image = X_test[sample_idx].reshape(28, 28)
        true_label = np.argmax(Y_test[sample_idx])
        pred_label = test_predictions[sample_idx]
        
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png') 