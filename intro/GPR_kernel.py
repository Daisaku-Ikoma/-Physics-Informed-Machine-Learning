import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel

# Define a function to generate sample data
def generate_data():
    X = np.array([[1], [3], [5], [6], [7], [8]])
    y = np.sin(X).ravel()
    return X, y

# Define a function to plot the results in subplots
def plot_gpr_results_subplots(X, y, X_test, kernels):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # Set the entire figure background to black
    plt.gcf().patch.set_facecolor('black')
    
    for i, (kernel, kernel_name) in enumerate(kernels):
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(X, y)
        y_pred, sigma = gpr.predict(X_test, return_std=True)
        
        row, col = divmod(i, 3)
        axs[row, col].scatter(X, y, color='#FF2F92', s=50, label='Observed Data')
        axs[row, col].plot(X_test, y_pred, '#0096FF', label='Prediction')
        axs[row, col].fill_between(X_test.ravel(), 
                                   y_pred - 1.96 * sigma, 
                                   y_pred + 1.96 * sigma, 
                                   alpha=0.25, color='white')
        axs[row, col].set_title(f'{kernel_name} Kernel', color='white')
        axs[row, col].set_xlim([0, 10])
        axs[row, col].set_ylim([-2, 2])
        
        # Set background color to black
        axs[row, col].set_facecolor('black')
        
        # Set axis spines, ticks, and labels to white
        axs[row, col].spines['bottom'].set_color('white')
        axs[row, col].spines['top'].set_color('white') 
        axs[row, col].spines['right'].set_color('white')
        axs[row, col].spines['left'].set_color('white')
        axs[row, col].xaxis.label.set_color('white')
        axs[row, col].yaxis.label.set_color('white')
        axs[row, col].tick_params(axis='x', colors='white')
        axs[row, col].tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.show()
    #plt.savefig('plot.png', facecolor='black', dpi=300)

# Generate sample data
X, y = generate_data()

# Generate test data
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# List of kernels to compare
kernels = [
    (RBF(length_scale=1.0), "RBF"),
    (Matern(length_scale=1.0, nu=1.5), "Matern"),
    (RationalQuadratic(length_scale=1.0, alpha=0.1), "RationalQuadratic"),
    (DotProduct(), "DotProduct"),
    (DotProduct() ** 3, "DotProduct^3"),    
    (DotProduct() ** 3 +WhiteKernel(), "DotProduct^3+noise")
]

# Plot the GPR results with different kernels
plot_gpr_results_subplots(X, y, X_test, kernels)