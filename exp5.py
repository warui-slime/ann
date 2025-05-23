# **Weights and Bias effect on Output**

Effect of weight on network

import matplotlib.pyplot as plt
weight = 1
bias = 1
x = range(-10, 11)
y = [weight * i + bias for i in x]

legend_label = f"weight = {weight}, bias = {bias}"

plt.figure(figsize=(8, 6))  # Added for better visualization
plt.plot(x, y, label=legend_label)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Move spines to center
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

plt.title("Weight and Bias Visualization")
plt.legend()
plt.grid(True)
plt.show()

weight = 3
bias = 1
x = range(-10, 11)
y = [weight * i + bias for i in x]  # Fixed variable name (1 → i)

legend_label = f"weight = {weight}, bias = {bias}"

# Plot configuration
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=legend_label)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Center the axes
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')  # Hide right spine
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')    # Hide top spine

# Add labels and grid
plt.title("Weight and Bias Visualization")
plt.legend()
plt.grid(True)
plt.show()

# Define parameters and generate data
weight = -2
bias = 1
x = range(-10, 11)
y = [weight * i + bias for i in x]

# Corrected f-string (was using parentheses instead of curly braces)
legend_label = f"weight = {weight}, bias = {bias}"

# Plot configuration
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=legend_label)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Center the axes
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')  # Hide right spine
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')    # Hide top spine

# Add labels and grid
plt.title("Weight and Bias Visualization")
plt.legend()
plt.grid(True)
plt.show()

weight = 1
bias = 5
x = range(-10, 11)
y = [weight * i + bias for i in x]  # Linear equation: y = 1x + 5

# Plot configuration
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f"weight = {weight}, bias = {bias}")
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Axis customization
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')  # Hide right spine
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')    # Hide top spine

# Labels and styling
plt.title("Weight and Bias Effect on Linear Function")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Parameters and linear function definition
weight = 1
bias = -2
x = range(-10, 11)
y = [weight * i + bias for i in x]  # y = 1x - 2

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f"y = {weight}x + ({bias})")  # Improved label format
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Center axes and customize spines
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')  # Hide right spine
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')    # Hide top spine

# Add labels and grid
plt.title("Effect of Negative Bias on Linear Function")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)  # Dotted grid lines
plt.show()


### Conclusion

- Weights determine the relative importance of each input feature by adjusting/shifting the slope of the decision line. As on decreasing the weights its slope increases and on negative weights it becomes negative.
  

- Bias acts as a constant threshold. On increasing and decreasing the bias, the decision line moves upwards or downwards.
