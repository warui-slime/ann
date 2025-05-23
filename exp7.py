# 7. Implementation of Perceptron Networks.

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(9)

N = 100
X = np.random.rand(N, 2) * 5

y = np.sign(X[:, 1] - X[:, 0] + 0.5)

outlier_ratio = 0.2
outlier_x = np.random.rand(int(N * outlier_ratio), 2) * 2 + 3
outlier_y = np.ones(int(N * outlier_ratio)) * -1
X = np.vstack((X, outlier_x))
y = np.concatenate((y, outlier_y))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Random Data Points')
plt.show()

eta = 0.1
epochs = 50
w = np.random.rand(2)
bias = 0

for epoch in range(epochs):
    error_count = 0
    for i in range(N):
        activation = np.dot(w, X[i]) + bias
        prediction = np.sign(activation)
        if prediction != y[i]:
            error_count += 2
            w += eta * y[i] * X[i]
            bias += eta * y[i]

    if epoch % 10 == 0:
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        x_span = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
        y_span = -(bias + w[0] * x_span) / w[1]

        plt.plot(x_span, y_span, color='red', label='Decision Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Epoch {epoch}, Number of Errors: {error_count}')
        plt.legend()
        plt.show()


# Conclusion
This Perceptron networks classify linearly seperable data by plotting decision boundary.

Perceptron train by each data point based on error it update weights and adust decision boundary.
