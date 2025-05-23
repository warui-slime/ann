# **6. Implementation of different learning rules**

## Learning Rules in ANN

**5 types of Learning Rules:**
1. Hebbian Learning Rule  
2. Error Correction  
3. Memory Based  
4. Competitive  
5. Boltzmann  

---

### 1. Hebbian Learning Rule

- Set all weights to zero, \( w_i = 0 \) for \( i = 1 \) to \( n \), and bias to zero.  
- For each input vector, \( S \) (input vector) : \( t \) (target output pair), repeat steps 3-5.  
- Set activations for input units with the input vector \( X_i = S_i \) for \( i = 1 \) to \( n \).  
- Set the corresponding output value to the output neuron, i.e., \( y = t \).  
- Update weight and bias by applying the Hebb rule for all \( i = 1 \) to \( n \):

    wi(new) = wi(old) + xiy
  
    b(new) = b(new) +y


# OR GATE using Hebbian Rule

import numpy as np
x1 = np.array([1,1,-1,-1])
x2 = np.array([1,-1,1,-1])
t=np.array([1,1,1,-1])
w1=0
w2=0
bias=0
def hebbian_learning(x1, x2, t, w1, w2, bias):
    # Update weights
    w1 += np.dot(x1, t)
    w2 += np.dot(x2, t)
    bias += np.sum(t)
    return w1, w2, bias

print("x1\tx2\tt\ty_pred")
for i in range(len(x1)):
  y_pred = bias + w1 * x1[i] + w2 * x2[i]
  print(x1[i], "\t", x2[i], "\t", t[i], "\t", y_pred)
  if y_pred != t[i]:
  # print("Incorrect prediction, updating weights...")
    w1 = w1 + x1[i] * t[i]
    w2 = w2 + x2[i] * t[i]
    bias = bias + t[i]
    print("Updated w1:", w1, " Updated w2:", w2, " Updated bias:", bias)

# AND GATE using Hebbian Rule

import numpy as np
x1 = np.array([1, 1, -1, -1])
x2 = np.array([1, -1, 1, -1])
t = np.array([1, -1, -1, -1])
w1 = 0
w2 = 0
bias = 0

def threshold(y_pred):
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = -1
        else:
            y_pred[i] = 1

def hebbian_learning(x1, x2, t, w1, w2, bias):
    # Update weights
    w1 += np.dot(x1, t)
    w2 += np.dot(x2, t)
    bias += np.sum(t)
    return w1, w2, bias

for epoch in range(2):
    print("Epoch", epoch)
    print("x1\tx2\tt\ty_pred")
    for i in range(len(x1)):
        y_pred = bias + w1 * x1[i] + w2 * x2[i]
        print(x1[i], "\t", x2[i], "\t", t[i], "\t", y_pred)
        if y_pred != t[i]:
            w1 = w1 + x1[i] * t[i]
            w2 = w2 + x2[i] * t[i]
            bias = bias + t[i]
            print("Updated w1:", w1, " Updated w2:", w2, " Updated bias:", bias)

# Convert predictions to binary
y_pred = bias + w1 * x1 + w2 * x2
threshold(y_pred)
print("\nPredictions after thresholding:", y_pred)

# Adaline Learning Rule

Step 1: Initialize weight not zero but small random values are used. Set learning rate α.

Step 2: While the stopping condition is False do steps 3 to 7.

Step 3: For each training set perform steps 4 to 6.

Step 4: Set activation of input unit xi = si for (i = 1 to n).

Step 5: Compute net input to output unit
        y_in = sum(w_i * x_i) + b
        # Here, b is the bias and n is the total number of neurons.

Step 6: Update the weights and bias for i = 1 to n
        w_i(new) = w_i(old) + η * (t - y_in) * x_i
        b(new) = b(old) + α * (t - y_in)

        # and calculate the error:
        error = (t - y_in)^2
        # When the predicted output and the true value are the same, then the weight will not change.

Step 7: Test the stopping condition.
        # The stopping condition may be when the weight changes at a low rate or no change.


# OR GATE using Adaline Learning Rule

import numpy as np

# Bipolar OR gate input patterns and expected outputs
x1 = np.array([1, 1, -1, -1])
x2 = np.array([1, -1, 1, -1])
t = np.array([1, 1, 1, -1])

# Initialize weights, bias, learning rate, total_error, and iteration
w1 = 0.1
w2 = 0.1
b = 0.1
eta = 0.1
total_error = 0
iteration = 0

# OR Gate using Addline Learning rate
print("Iteration\tInput\tTarget\tYin\tError\tW1\tW2\tBias\tFinal Error\tTotal Error")
for j in range(3):
    total_error = 0
    for i in range(4):
        y_in = b + w1 * x1[i] + w2 * x2[i]
        error = t[i] - y_in
        final_error = error**2
        total_error += final_error
        w1 += eta * error * x1[i]
        w2 += eta * error * x2[i]
        b += eta * error

        print(f"{iteration+1}\t{x1[i], x2[i]}\t{t[i]}\t{y_in:.4f}\t{error:.4f}\t{w1:.4f}\t{w2:.4f}\t{b:.4f}\t{final_error:.4f}\t{total_error:.4f}")
        iteration += 1
        if total_error <= 2:
            break

print("\nFinal Weights and Bias:")
print(f"w1: {w1:.4f}")
print(f"w2: {w2:.4f}")
print(f"Bias: {b:.4f}")


## AND GATE using ADALINE Learning Rule

import numpy as np

# Bipolar OR gate input patterns and expected outputs
x1 = np.array([1, 1, -1, -1])
x2 = np.array([1, -1, 1, -1])
t = np.array([1, -1, -1, -1])
w1 = 0.1
w2 = 0.1
b = 0.1
eta = 0.1

total_error = 0
iteration = 0

# AND Gate using Adaline Learning rate
print("Iteration\tInput\tTarget\tYin\tError\tW1\tW2\tBias\tFinal Error\tTotal Error")
for j in range(3):
    total_error = 0
    for i in range(4):
        y_in = b + w1 * x1[i] + w2 * x2[i]
        error = t[i] - y_in
        final_error = error**2
        total_error += final_error
        w1 += eta * error * x1[i]
        w2 += eta * error * x2[i]
        b += eta * error
        print(f"{iteration+1}\t{x1[i], x2[i]}\t{t[i]}\t{y_in:.4f}\t{error:.4f}\t{w1:.4f}\t{w2:.4f}\t{b:.4f}\t{final_error:.4f}\t{total_error:.4f}")
    iteration += 1
    if total_error <= 2:
        break

print("\nFinal Weights and Bias:")
print(f"w1: {w1:.4f}")
print(f"w2: {w2:.4f}")
print(f"Bias: {b:.4f}")

# 3. Memory based learning

Memory-based learning in artificial neural networks (ANNs) involves storing and utilizing past experiences or training examples directly rather than learning explicit parameters. This approach is also known as instance-based learning or lazy learning. One of the most popular memory-based learning algorithms is the k-nearest neighbors (k-NN) algorithm.


x1 = [0, 0, 0, 1, 1, 1]
x2 = [0, 0, 1, 1, 0, 1]
x3 = [0, 1, 0, 1, 0, 1]

y = [0, 0, 0, 1, 0, 1]

x_test = [1, 0, 1]


import matplotlib.pyplot as plt

# Separate the data points based on their labels
class_0_x1 = [x1[i] for i in range(len(x1)) if y[i] == 0]
class_0_x2 = [x2[i] for i in range(len(x2)) if y[i] == 0]
class_0_x3 = [x3[i] for i in range(len(x3)) if y[i] == 0]

class_1_x1 = [x1[i] for i in range(len(x1)) if y[i] == 1]
class_1_x2 = [x2[i] for i in range(len(x2)) if y[i] == 1]
class_1_x3 = [x3[i] for i in range(len(x3)) if y[i] == 1]

# Plot the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(class_0_x1, class_0_x2, class_0_x3, c='blue', label='Class 0')
ax.scatter(class_1_x1, class_1_x2, class_1_x3, c='red', label='Class 1')
ax.scatter(x_test[0], x_test[1], x_test[2], c='green', marker='x', label='Test Point')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

plt.legend()
plt.title('Data Points')
plt.show()


n = len(x1)
y_eucli = [0,0,0,0,0,0]
for i in range(n):
    y_eucli[i] = ((x1[i]-x_test[0])**2 + (x2[i]-x_test[1])**2 + (x3[i]-x_test[2])**2 )**0.5

# Combine distances with labels
combined_data = list(zip(y, y_eucli))

# Sort the combined data based on distances
sorted_data = sorted(combined_data, key=lambda x: x[1])

k = 3
nearest_neighbors = sorted_data[:k]

# Extract coordinates of k nearest neighbors
nearest_neighbor_indices = [x[0] for x in nearest_neighbors]
nearest_neighbor_coords = [(x1[i], x2[i], x3[i]) for i in nearest_neighbor_indices]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, x3, c=y, cmap='coolwarm', label='Data Points')
ax.scatter(x_test[0], x_test[1], x_test[2], c='green', marker='x', label='Test Point')

for i in range(k):
    ax.scatter(nearest_neighbor_coords[i][0], nearest_neighbor_coords[i][1], nearest_neighbor_coords[i][2],
               c='red', s=100, label=f'Neighbor {i+1}')
    ax.plot([x_test[0], nearest_neighbor_coords[i][0]],
            [x_test[1], nearest_neighbor_coords[i][1]],
            [x_test[2], nearest_neighbor_coords[i][2]], c='black')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

plt.legend()
plt.title('Data Points with Nearest Neighbors and Connecting Lines')
plt.show()

# Print distances
print("y", "y_euclidean")
for label, distance in sorted_data:
    print(label, distance)


import plotly.graph_objects as go

# Create traces for data points, test point, and connecting lines
data_points = go.Scatter3d(x=x1, y=x2, z=x3, mode='markers', marker=dict(color=y, size=8))
test_point = go.Scatter3d(x=[x_test[0]], y=[x_test[1]], z=[x_test[2]], mode='markers', marker=dict(color='green', size=10, symbol='x'))

neighbors = []
for i, coord in enumerate(nearest_neighbor_coords):
    neighbors.append(go.Scatter3d(x=[x_test[0], coord[0]], y=[x_test[1], coord[1]], z=[x_test[2], coord[2]],
                                  mode='lines', line=dict(color='black')))

fig = go.Figure(data=[data_points, test_point] + neighbors)

# Update layout
fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'), title='3D k-NN Visualization')

fig.show()


# Competitive Learning Rule

c_x1 = [0.2, 0.6, 0.4, 0.9, 0.2]
c_x2 = [0.3, 0.5, 0.7, 0.6, 0.8]
x1 = 0.3
x2 = 0.4

d = [0 for i in range(len(c_x1))]  # list representing distance square d**2

for i in range(len(c_x1)):
    d[i] = ((c_x1[i] - x1)**2 + (x2 - c_x2[i])**2)

print(d)
min_value = min(d)
min_index = d.index(min_value)

print("Minimum value:", min_value)
print("Cluster of minimum value:", min_index + 1)


eta = 0.3
c_x1[min_index] = c_x1[min_index] + eta * (x1 - c_x1[min_index])
c_x2[min_index] = c_x2[min_index] + eta * (x2 - c_x2[min_index])

print(c_x1)
print(c_x2)


d = [0 for i in range(len(c_x1))]  # list representing distance square d**2

for i in range(len(c_x1)):
    d[i] = ((c_x1[i] - x1)**2 + (x2 - c_x2[i])**2)

print(d)
min_value = min(d)
min_index = d.index(min_value)

print("Minimum value:", min_value)
print("Cluster of minimum value:", min_index + 1)
