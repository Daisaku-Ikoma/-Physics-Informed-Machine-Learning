#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import warnings
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, ConstantKernel as C, WhiteKernel as W
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#####################################################################################
# Input parameters
#####################################################################################
# Defining inputs and outputs
data_file = 'adhesive.csv'
df = pd.read_csv(data_file)

# Selection option: 1 = traditional ML, 2: Physics-informed ML
option = 1

if option == 1:
    # Features
    X_columns = ['T_F', 'RH']
    
    # GPR Kernels to try
    kernels = [
        ("RBF+noise", RBF() + W()),
        ("Dot+noise", DotProduct() + W()),
        ("Matern+noise", Matern() + W()),
        ("Dot^2+noise", DotProduct()**2 + W()),
        ("RBF+Dot+noise", RBF() + DotProduct() + W()),
        ("Matern+Dot+noise", Matern() + DotProduct() + W()),
    ]  
    
    # Output
    Y_columns = ['strength_psi']  
    
if option == 2:
    # physics-informed feature
    X_columns = ['T_Tg']
    df['T_Tg'] = df['T_F'] - df['Tg_F'] 
    
    #physics-informed GPR Kernel
    kernels = [
        ("Dot^2+Dot+C+noise", C() * DotProduct()**2 + C() * DotProduct() + C() + W()),
    ]  
    
    # Output
    Y_columns = ['strength_psi'] 
    
#####################################################################################
# Functions
#####################################################################################
def surrogate_models(X_train, X_test, Y_train, Y_test, kernels):  
    # Dictionary to store model names and their R² scores
    results = []
    best_model = None 
    best_rmse = np.inf    
    gpr_model = None    
    
    for name, kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=50, random_state=42)
        try:
            gpr.fit(X_train, Y_train)
            
            # Predict on training and testing data
            Y_pred_train, std_train = gpr.predict(X_train, return_std=True)
            Y_pred_test, std_test = gpr.predict(X_test, return_std=True)
            
            # Calculate RMSE for training and testing data
            rmse_train = round(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
            rmse_test = round(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))
                      
            results.append({"Model": name, "RMSE_test": rmse_test, "RMSE_train": rmse_train})
            
            if rmse_test < best_rmse:
                best_rmse = rmse_test
                best_model = gpr
        except Exception as e:
            print(f"Failed to train GPR with {name}: {e}")

    ################################################### 
    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values(by='RMSE_test', ascending=True).reset_index(drop=True)

    return results_df, best_model

#####################################################################################
# Main Code
#####################################################################################
# Read and filter input file
X = df[X_columns]
Y = df[Y_columns].values.ravel()

# Split the data based on the 'Type' column
X_train = X[df['Type'] == 'Train']
X_test = X[df['Type'] == 'Test']
Y_train = Y[df['Type'] == 'Train']
Y_test = Y[df['Type'] == 'Test']

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Select the best model
results_df, best_model = surrogate_models(X_train_scaled, X_test_scaled, Y_train, Y_test, kernels)
print(results_df)

# Adding predictions to the dataframe with uncertainty
mean_prediction, std_prediction = best_model.predict(X_scaled, return_std=True)
df['predicted_strength'] = mean_prediction
df['upper_bound'] = mean_prediction + 1.96 * std_prediction  # 95% confidence interval
df['lower_bound'] = mean_prediction - 1.96 * std_prediction  # 5% confidence interval

if option == 1:
    # 3D scatter plot for strength vs T_F and RH with bounds
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 10))   
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black') 

    # Original data points
    ax.scatter(X_train['T_F'], X_train['RH'], Y_train, label='Training data', color='#FF2F92', alpha=0.5, s=80)
    ax.scatter(X_test['T_F'], X_test['RH'], Y_test, label='Test data', color='#0096FF', alpha=0.5, s=80)

    # Create a meshgrid for T_F and RH
    T_F_range = np.linspace(df['T_F'].min(), df['T_F'].max(), 100)
    RH_range = np.linspace(df['RH'].min(), df['RH'].max(), 100)
    T_F_grid, RH_grid = np.meshgrid(T_F_range, RH_range)
    X_grid = np.column_stack((T_F_grid.ravel(), RH_grid.ravel()))
    X_grid_scaled = scaler.transform(X_grid)

    # Predict the mean, upper bound, and lower bound for the grid
    mean_grid, std_grid = best_model.predict(X_grid_scaled, return_std=True)
    mean_grid = mean_grid.reshape(T_F_grid.shape)
    std_grid = std_grid.reshape(T_F_grid.shape)
    upper_grid = mean_grid + 1.96 * std_grid
    lower_grid = mean_grid - 1.96 * std_grid

    # Plot mean prediction surface
    mean_surface = ax.plot_surface(T_F_grid, RH_grid, mean_grid, color='#FF9A4D', alpha=0.85)
    upper_surface = ax.plot_surface(T_F_grid, RH_grid, upper_grid, alpha=0.60,color='white')
    lower_surface = ax.plot_surface(T_F_grid, RH_grid, lower_grid, alpha=0.60,color='white')

    # Adding custom legend
    custom_lines = [
        plt.Line2D([0], [0], color='#FF2F92', marker='o', linestyle='', alpha=0.5, label='Training data'),
        plt.Line2D([0], [0], color='#0096FF', marker='o', linestyle='', alpha=0.5, label='Testing data'),
        plt.Line2D([0], [0], color='#FF9A4D', linewidth=3, alpha=0.85, label='Mean prediction'),
        plt.Line2D([0], [0], color='white', linewidth=3, alpha=0.60, label='Upper/Lower bound (95% and 5%)')
    ]

    ax.legend(handles=custom_lines, facecolor='black', edgecolor='white', labelcolor='white')

    ax.set_xlabel('T (F)', fontsize=20, color='white', labelpad=15)
    ax.set_ylabel('RH%', fontsize=20, color='white', labelpad=15)
    ax.set_zlabel('Strength (psi)', fontsize=20, color='white', labelpad=15)
    ax.set_title('3D Scatter Plot of Strength vs Temperature and Relative Humidity', fontsize=20, color='white')
    ax.tick_params(colors='white', labelsize=15) 
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False 
    plt.show()

elif option == 2:
    # 2D plot for strength vs T_Tg with bounds
    plot = plt.figure(figsize=(13, 10))
    plt.gca().set_facecolor('black') 
    plot.patch.set_facecolor('black')

    # Original data points
    plt.scatter(X_train['T_Tg'], Y_train, label='Training data', color='#FF2F92', alpha=0.85, s=80)
    plt.scatter(X_test['T_Tg'], Y_test, label='Test data', color='#0096FF', alpha=0.85, s=80)  

    # Create a range for T_Tg
    T_Tg_range = np.linspace(df['T_Tg'].min(), df['T_Tg'].max(), 100)
    X_grid = T_Tg_range.reshape(-1, 1)
    X_grid_scaled = scaler.transform(X_grid)

    # Predict the mean, upper bound, and lower bound for the grid
    mean_grid, std_grid = best_model.predict(X_grid_scaled, return_std=True)
    upper_grid = mean_grid + 1.96 * std_grid
    lower_grid = mean_grid - 1.96 * std_grid

    # Plot mean prediction line
    plt.plot(T_Tg_range, mean_grid, color='#FF9A4D', alpha=0.85, label='Mean prediction', linewidth=3)  

    # Plot upper and lower bounds
    plt.fill_between(T_Tg_range, lower_grid, upper_grid, color='gray', alpha=0.50, label='Upper/Lower bound (95% and 5%)')
  
    # Formatting
    plt.xlabel('T - Tg (F)', color='white', fontsize=20)
    plt.ylabel('Strength (psi)', color='white', fontsize=20)
    plt.gca().tick_params(colors='white', labelsize=15)
    plt.gca().spines[:].set_color('white')
    custom_lines = [
        plt.Line2D([0], [0], color='#FF2F92', marker='o', linestyle='', alpha=0.5, label='Training data'),
        plt.Line2D([0], [0], color='#0096FF', marker='o', linestyle='', alpha=0.5, label='Testing data'),
        plt.Line2D([0], [0], color='#FF9A4D', linewidth=3, alpha=0.85, label='Mean prediction'),
        plt.Line2D([0], [0], color='gray', linewidth=3, alpha=0.50, label='Upper/Lower bound (95% and 5%)')
    ]

    plt.legend(handles=custom_lines, facecolor='black', edgecolor='white', labelcolor='white', fontsize=20)    
    
    plt.show()
    
    
    