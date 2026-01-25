#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, ConstantKernel as C, WhiteKernel as W
from sklearn.metrics import mean_squared_error

#################################################################
# Inputs
#################################################################
# Define a function to generate the target data
def my_function(x):
    y = (x**2 + x + 1) / 100
    return y

# Define the kernel for GPR
kernel = RBF(length_scale=3) + W(noise_level=0)

#################################################################
# Main Code
#################################################################
# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(-10, 10, 10).reshape(-1, 1)  # 10 training data points
y = my_function(X).ravel()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate and fit the GPR model
gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=42)
gpr.fit(X_train, y_train)

# Make predictions
y_pred = gpr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Gaussian Process Regression MSE: {mse:.4f}")

# Plot the actual vs predicted values
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = my_function(x).ravel()
y_pred, sigma = gpr.predict(x, return_std=True)

plt.figure(figsize=(6, 5))
plt.gca().set_facecolor('black')  # Set the background color to black
plt.gcf().patch.set_facecolor('black')  # Set the figure's facecolor
plt.plot(x, y, color='#FF2F92', label='Actual')
plt.plot(x, y_pred, color='#0096FF', label='Predicted Mean')
plt.fill_between(x.ravel(), y_pred - 1.645*sigma, y_pred + 1.645*sigma, alpha=0.25, color='white', label='90% Confidence Interval')
plt.scatter(X_train, y_train, color='#FF2F92', s=50, label='Training Data')

# Customize the axes and labels
plt.xlabel('X', color='white')
plt.ylabel('y', color='white')
plt.xlim([-10, 10])  # Set x limits
plt.ylim([-1, 2])  # Set y limits
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.gca().tick_params(axis='x', colors='white')
plt.gca().tick_params(axis='y', colors='white')
plt.legend(facecolor='black', edgecolor='white', labelcolor='white')

# Show the plot
plt.show()
#plt.savefig('plot.png', facecolor='black', dpi=300)
