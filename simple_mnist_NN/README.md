# MNIST Neural Network from Scratch

This implementation builds a neural network for handwritten digit classification using only NumPy, without any deep learning frameworks like TensorFlow or PyTorch.

## Mathematical Foundation

This neural network implementation is based on the following mathematical concepts:

### 1. Neural Network Architecture

The network has:

- Input layer: 784 neurons (28×28 pixel images flattened)
- Hidden layers: Fully connected layers with configurable sizes
- Output layer: 10 neurons (one for each digit 0-9)

### 2. Forward Propagation

For each layer l:

1. **Linear transformation**: Z^[l] = A^[l-1] · W^[l] + b^[l]

   - A^[l-1] is the activation from previous layer (or input for first layer)
   - W^[l] is the weight matrix for layer l
   - b^[l] is the bias vector for layer l

2. **Activation**: A^[l] = g^[l](Z^[l])
   - Where g^[l] is the activation function for layer l, which can be:
     - ReLU: g(z) = max(0, z)
     - Sigmoid: g(z) = 1 / (1 + e^(-z))
     - Softmax (output layer): g(z)\_i = e^(z_i) / ∑_j e^(z_j)

### 3. Loss Function

The network uses cross-entropy loss:
J = -1/m · ∑_i ∑_j y_ij · log(a_ij)

Where:

- m is the number of training examples
- y_ij is the one-hot encoded true label (1 if example i is class j, 0 otherwise)
- a_ij is the predicted probability for example i being class j

### 4. Backward Propagation

The backpropagation algorithm computes gradients of the loss with respect to parameters:

1. **Output layer**:
   - For softmax with cross-entropy: dZ^[L] = A^[L] - Y
2. **Hidden layers**:
   - dZ^[l] = dA^[l] \* g'^[l](Z^[l])
     where g' is the derivative of the activation function
   - For ReLU: g'(z) = 1 if z > 0, else 0
   - For Sigmoid: g'(z) = g(z) \* (1 - g(z))
3. **Parameter gradients**:
   - dW^[l] = 1/m · A^[l-1]^T · dZ^[l]
   - db^[l] = 1/m · ∑_i dZ^[l]\_i
4. **Gradient for previous layer**:
   - dA^[l-1] = dZ^[l] · W^[l]^T

### 5. Parameter Updates

Using gradient descent:

- W^[l] = W^[l] - α · dW^[l]
- b^[l] = b^[l] - α · db^[l]

Where α is the learning rate.

### 6. Parameter Initialization

- For ReLU layers: He initialization
  W^[l] ~ N(0, sqrt(2/n^[l-1]))
- For Sigmoid/Softmax layers: Xavier initialization
  W^[l] ~ N(0, sqrt(1/n^[l-1]))

Where n^[l-1] is the number of neurons in the previous layer.

## Implementation Details

The implementation uses mini-batch gradient descent to handle large datasets efficiently. It includes:

- Data preprocessing (normalization, one-hot encoding)
- Vectorized operations for efficiency
- Performance metrics tracking (loss and accuracy)
- Visualization of results

## Usage

Run the script to train and evaluate the neural network on the MNIST dataset:

```
python mnist_from_scratch.py
```

The script will:

1. Load and preprocess the MNIST dataset
2. Train the neural network
3. Evaluate performance on test data
4. Generate visualizations of learning curves and predictions
