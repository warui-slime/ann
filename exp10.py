import numdifftools as nd

def fun(x):
    return 2*x[0]*x[1] - x[0]**2 - 2*x[1]**2 + 2*x[0]

def fun2(x):
    return 2*x[0]*x[1] + x[1]**2 + 6*x[0] + 2*x[1]

# Calculate gradient at point [2, -2]
grad2 = nd.Gradient(fun)([2, -2])
print(grad2)  # Output: [-6. 12.]

import matplotlib.pyplot as plt

def gradient_descent(epoch, input):
    eta = 0.1  # Learning rate
    history = [[input[0], input[1]]]  # Store optimization path

    for i in range(epoch):
        grad = nd.Gradient(fun)(input)
        input[0] = input[0] - eta * grad[0]
        input[1] = input[1] - eta * grad[1]
        history.append([input[0], input[1]])

    return history

# Run gradient descent for 10 epochs starting at [2, -2]
history = gradient_descent(10, [2, -2])

# Plot optimization trajectory
x_history = [point[0] for point in history]
y_history = [point[1] for point in history]

plt.plot(x_history, y_history, 'o-')
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Gradient Descent Trajectory")
plt.grid(True)
plt.show()

def gradient_descent(epoch, input):
    eta = 0.1  # Learning rate
    history = [[input[0], input[1]]]  # Store optimization path

    for i in range(epoch):
        grad = nd.Gradient(fun)(input)
        input[0] = input[0] - eta * grad[0]  # Changed + to - for gradient descent
        input[1] = input[1] - eta * grad[1]  # Changed + to - for gradient descent
        history.append([input[0], input[1]])

    return history

# Run gradient descent for 500 epochs starting at [2, -2]
history = gradient_descent(500, [2, -2])

# Plot optimization trajectory
x_history = [point[0] for point in history]
y_history = [point[1] for point in history]

plt.plot(x_history, y_history, 'b-', linewidth=1)
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Gradient Descent Trajectory")  # Corrected title
plt.grid(True)
plt.show()

def gradient_descent(epoch, input):
    eta = 0.1  # Learning rate
    history = [[input[0], input[1]]]  # Store optimization path

    for i in range(epoch):
        grad = nd.Gradient(fun2)(input)  # Using fun2 for gradient calculation
        input[0] = input[0] - eta * grad[0]  # Gradient descent update
        input[1] = input[1] - eta * grad[1]  # Gradient descent update
        history.append([input[0], input[1]])

    return history

# Run gradient descent for 10 epochs starting at [2, -2]
history = gradient_descent(10, [2, -2])

# Extract x and y coordinates from history
x_history = [point[0] for point in history]
y_history = [point[1] for point in history]

# Plot optimization trajectory
plt.plot(x_history, y_history, 'o-')  # Added marker and line style
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Gradient Descent Trajectory")
plt.grid(True)  # Added grid for better visualization
plt.show()

def gradient_descent(epoch, input):
    eta = 0.1  # Learning rate
    history = [[input[0], input[1]]]  # Store optimization path

    for i in range(epoch):
        grad = nd.Gradient(fun2)(input)
        input[0] = input[0] - eta * grad[0]  # Changed + to - for proper gradient descent
        input[1] = input[1] - eta * grad[1]  # Changed + to - for proper gradient descent
        history.append([input[0], input[1]])

    return history

# Run gradient descent for 10 epochs starting at [2, -2]
history = gradient_descent(10, [2, -2])

# Extract x and y coordinates from history
x_history = [point[0] for point in history]
y_history = [point[1] for point in history]

# Plot optimization trajectory
plt.plot(x_history, y_history, 'b-', linewidth=1)
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Gradient Descent Trajectory")  # Corrected title
plt.grid(True)
plt.show()

# Print final position
print("Final position:", history[-1])

import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    return x[0]*x[1] + 4*x[1] - 3*x[0]**2 - x[1]**2

def newton_method(fun, x0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    x_history = [x.copy()]

    for i in range(max_iter):
        grad = nd.Gradient(fun)(x)
        hess = nd.Hessian(fun)(x)

        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print(f"Hessian is singular at iteration {i}")
            return x, x_history

        x_new = x - np.dot(hess_inv, grad)
        x_history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            return x_new, x_history

        x = x_new

    print("Maximum iterations reached without convergence")
    return x, x_history

# Initial guess
x0 = [1, 1]
optimal_x, x_history = newton_method(fun, x0)
print(f"Optimal point: {optimal_x}")

# Plot optimization path if desired
x_history = np.array(x_history)
plt.plot(x_history[:, 0], x_history[:, 1], 'o-')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Newton Method Optimization Path')
plt.grid(True)
plt.show()

# Conclusion
The Newton's method efficiently optimizes the function xy + 4y - 3*x2 - y2 by iteratively updating the
variables using the inverse Hessian and gradient The visualization aids in understanding the convergence
trajectory from the initial guess to the optimal point.
