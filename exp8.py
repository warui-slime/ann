#8. Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
  return 1/(1+np.exp(-x))

x1 = 0.35
x2 = 0.9
t = 0.5
w13 = 0.1
w14 = 0.4
w23 = 0.8
w24 = 0.6
w45 = 0.9
w35 = 0.3


v3 = x1 * w13 + x2 * w23
y3 = sigmoid(v3)

v4 = x1 * w14 + x2 * w24
y4 = sigmoid(v4)

v5 = y3 * w35 + y4 * w45
y5 = sigmoid(v5)

error = t - y5

print("Activations:")
print(f"y3 = {y3}, y4 = {y4}, y5 = {y5}")
print("Error:")
print(f"Error = {error}")


delta5 = y5 * (1 - y5) * error

delta3 = y3 * (1 - y3) * (delta5 * w35)
delta4 = y4 * (1 - y4) * (delta5 * w45)

print("Local Gradients:")
print(f"delta3 = {delta3}, delta4 = {delta4}, delta5 = {delta5}")


eta = 1
w35 += eta * y3 * delta5
w45 += eta * y4 * delta5
w13 += eta * x1 * delta3
w14 += eta * x1 * delta4
w23 += eta * x2 * delta3
w24 += eta * x2 * delta4

print("Updated Weights:")
print(f"w13 = {w13}, w14 = {w14}, w23 = {w23}, w24 = {w24}, w35 = {w35}, w45 = {w45}")


v3 = x1 * w13 + x2 * w23
y3 = sigmoid(v3)

v4 = x1 * w14 + x2 * w24
y4 = sigmoid(v4)

v5 = y3 * w35 + y4 * w45
y5 = sigmoid(v5)

error = t - y5

print("Epoch 2 Activations:")
print(f"y3 = {y3}, y4 = {y4}, y5 = {y5}")
print("Epoch 2 Error:")
print(f"Error = {error}")


def forward_pass(x1, x2, w13, w14, w23, w24, w35, w45):
    v3 = x1 * w13 + x2 * w23
    y3 = sigmoid(v3)

    v4 = x1 * w14 + x2 * w24
    y4 = sigmoid(v4)

    v5 = y3 * w35 + y4 * w45
    y5 = sigmoid(v5)

    return y3, y4, y5


def calculate_error(y5, t):
    error = t - y5
    return error

def calculate_local_gradients(y3, y4, y5, w35, w45, error):
    delta5 = y5 * (1 - y5) * error
    delta3 = y3 * (1 - y3) * (delta5 * w35)
    delta4 = y4 * (1 - y4) * (delta5 * w45)

    return delta3, delta4, delta5

def update_weights(x1, x2, delta3, delta4, delta5, learning_rate, w13, w14, w23, w24, w35, w45, y3, y4):
    w35 += learning_rate * y3 * delta5
    w45 += learning_rate * y4 * delta5
    w13 += learning_rate * x1 * delta3
    w14 += learning_rate * x1 * delta4
    w23 += learning_rate * x2 * delta3
    w24 += learning_rate * x2 * delta4

    return w13, w14, w23, w24, w35, w45

errors = []

def train_one_epoch(x1, x2, w13, w14, w23, w24, w35, w45, t, eta):
    y3, y4, y5 = forward_pass(x1, x2, w13, w14, w23, w24, w35, w45)
    error = calculate_error(y5, t)
    delta3, delta4, delta5 = calculate_local_gradients(y3, y4, y5, w35, w45, error)
    w13, w14, w23, w24, w35, w45 = update_weights(x1, x2, delta3, delta4, delta5, eta, w13, w14, w23, w24, w35, w45, y3, y4)
    return error, w13, w14, w23, w24, w35, w45

for epoch in range(70):
    error, w13, w14, w23, w24, w35, w45 = train_one_epoch(x1, x2, w13, w14, w23, w24, w35, w45, t, eta)
    errors.append(error)

plt.plot(range(1, 71), errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

import matplotlib.pyplot as plt

x1 = 0.35
x2 = 0.9
w13 = 0.1
w14 = 0.4
w23 = 0.8
w24 = 0.6
w35 = 0.3
w45 = 0.9
t = 0.5
learning_rate = 1
alpha = 0.5  t

def update_weights(x1, x2, delta3, delta4, delta5, learning_rate, w13, w14, w23, w24, w35, w45, y3, y4, alpha):
    w35 = alpha * w35 + learning_rate * y3 * delta5
    w45 = alpha * w45 + learning_rate * y4 * delta5
    w13 = alpha * w13 + learning_rate * x1 * delta3
    w14 = alpha * w14 + learning_rate * x1 * delta4
    w23 = alpha * w23 + learning_rate * x2 * delta3
    w24 = alpha * w24 + learning_rate * x2 * delta4
    return w13, w14, w23, w24, w35, w45

errors = []

def train_one_epoch(x1, x2, w13, w14, w23, w24, w35, w45, t, eta, alpha):
    y3, y4, y5 = forward_pass(x1, x2, w13, w14, w23, w24, w35, w45)
    error = calculate_error(y5, t)
    delta3, delta4, delta5 = calculate_local_gradients(y3, y4, y5, w35, w45, error)
    w13, w14, w23, w24, w35, w45 = update_weights(x1, x2, delta3, delta4, delta5, eta, w13, w14, w23, w24, w35, w45, y3, y4, alpha)
    return error, w13, w14, w23, w24, w35, w45

# Training loop
for epoch in range(10):
    error, w13, w14, w23, w24, w35, w45 = train_one_epoch(x1, x2, w13, w14, w23, w24, w35, w45, t, learning_rate, alpha)
    errors.append(error)

# Plotting
plt.plot(range(1, 11), errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income']

df = pd.read_csv(url, header=None, names=names, na_values=' ?')


df.dropna(inplace=True)


label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


X = df.drop('income', axis=1)
y = df['income']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate


        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

        self.training_loss = []

    def forward(self, X):

        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output):
        error = y - output
        output_delta = error * sigmoid_derivative(output)

        error_hidden = output_delta.dot(self.weights2.T)
        hidden_delta = error_hidden * sigmoid_derivative(self.hidden_output)


        self.weights2 += self.learning_rate * self.hidden_output.T.dot(output_delta)
        self.bias2 += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * X.T.dot(hidden_delta)
        self.bias1 += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)


            loss = np.mean((y - output)**2)
            self.training_loss.append(loss)

    def predict(self, X):
        return np.round(self.forward(X))

input_size = X_train.shape[1]
hidden_size = 5

output_size = 1

model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
model.train(X_train, y_train.values.reshape(-1, 1), epochs=50)


y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test.values.reshape(-1, 1), y_pred)
print("Test Accuracy:", test_accuracy)
print()
plt.plot(range(1, len(model.training_loss) + 1), model.training_loss, color='blue')
plt.title('Training Loss (Error) vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid(True)
plt.show()

# Conclusion
BP Algorthim is used to capture the non linearity at different hidden layers by updating weights using gradient descent.

Momentum factor can drastically decrease the training epoch in BPA [from 100 to 6]

On Income dataset, achieve 75% accuracy using simple BPA.
