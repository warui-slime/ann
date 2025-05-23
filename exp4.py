
X1 = [1, 1, -1, -1]
print(X1)

X2 = [1, -1, 1, -1]
print(X2)

y = [1, -1, -1, -1]
print(y)



import numpy as np
def tanh_function(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

w1 = 10
w2 = 11

y_pred = [0, 0, 0, 0]
for i in range(4):
    y_pred[i] = (X1[i] * w1) + (X2[i] * w2)
print(y_pred)

for i in y_pred:
  print(tanh_function(i))

# Sigmoid

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# ReLU

def relu(x):
    return max(0, x)

def test(y, y_p):
    flag = "Correct"
    index = -1
    for i in range(4):
        if y[i] != y_p[i]:
            flag = "Incorrect"
            index = i
            break
    return flag, index


print(test(y, y_pred))

# Effect of Sigmoid and Tanh on input

x1 = [1, -10, 0, 15, -2]
tanhOutput = [0, 0, 0, 0, 0]
SigmoidOutput = [0, 0, 0, 0, 0]
ReLUOutput = [0, 0, 0, 0, 0]

for i in range(len(x1)):
    tanhOutput[i] = tanh_function(x1[i])
    SigmoidOutput[i] = sigmoid(x1[i])


print("Tanh Output:", [float(x) for x in tanhOutput])
print("Sigmoid Output:", [float(x) for x in SigmoidOutput])


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 5))
plt.plot(x1, label='Input')
plt.plot(tanhOutput, label='Tanh')
plt.plot(SigmoidOutput, label='Sigmoid')
plt.plot(ReLUOutput, label='ReLU')


plt.legend(loc="upper left")
plt.title("Activation Function Comparison")
plt.xlabel("Input Index")
plt.ylabel("Output Value")
plt.grid(True)
plt.show()

# Multilayer perceptron for XOR


X1=[0,0,1,1]
X2=[0,1,0,1]
y=[0,1,1,0]
w1=0
w2=1
w3=1
w4=0
w5=1
w6=-1
y_pred = [0, 0, 0, 0]
z1 = [0, 0, 0, 0]
z2 = [0, 0, 0, 0]

for i in range(4):
    z1[i] = (X1[i] * w1) + (X2[i] * w3)  # First weighted sum
    z2[i] = (X2[i] * w4) + (X1[i] * w2)  # Second weighted sum

    # Conditional prediction
    if ((z1[i] * w5) + (z2[i] * w6) < 0):
        y_pred[i] = 1
    else:
        y_pred[i] = (z1[i] * w5) + (z2[i] * w6)  # Original value if ≥ 0

print("z1:", z1)
print("z2:", z2)
print("Predictions:", y_pred)
print(test(y, y_pred))
