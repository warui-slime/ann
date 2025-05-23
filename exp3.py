# **3.Write a program to implement AND, OR, XOR gates to understand Linearly separable and non-linearly separable problems**

import numpy as np
import matplotlib.pyplot as plt

# AND gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# OR gate data
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# XOR gate data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

def plot_data_and_boundary(X, y, gate_type):
    plt.figure(figsize=(8, 6))


    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')

    if gate_type == 'AND':
        plt.plot([-0.5, 1.5], [1.9, -0.4], color='green', linestyle='--', label='Decision Boundary')
    elif gate_type == 'OR':
        plt.plot([-0.5, 1.5], [0.5, -0.5], color='green', linestyle='--', label='Decision Boundary')
    elif gate_type == 'XOR':
        plt.plot([-0.5, 1.5], [1.9, -0.4], color='green', linestyle='--', label='Decision Boundary 1')
        plt.plot([-0.5, 1.5], [0.5, -0.5], color='purple', linestyle='--', label='Decision Boundary 2')

    plt.title(f'{gate_type} Gate')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_data_and_boundary(X_and, y_and, gate_type='AND')

plot_data_and_boundary(X_or, y_or, gate_type='OR')

plot_data_and_boundary(X_xor, y_xor, gate_type='XOR')
