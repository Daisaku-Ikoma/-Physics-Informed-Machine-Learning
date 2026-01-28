#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import pandas as pd
import numpy as np
import itertools
import pysindy as ps
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

#####################################################################################
# Input parameters
#####################################################################################
# Load the data
#1: SC-1008, condensation deceleration reaction:
#           dx/dt = f(T) * (1 - x)**1.37
#2: HEXCEL 8552, autocatalytic epoxy-amine reaction:
#           dx/dt = f(T) * (x)**0.8129 * (1 - x)**2.736
#3: Toray 3900, multi autocatalytic reactions (primary and secondary amine-epoxide):
#           dx/dt = [f(T) * (1 - x)**1] + [g(T) * (x)**1 * (1 - x)**2.5]
option = 1

#####################################################################################
# Functions
#####################################################################################
# Generate both functions and their names
def generate_functions_and_names():
    # Chemical reaction (deceleration) 
    def gen_func_1(n):
        return lambda x: (1 - x)**n, lambda _: f'(1-x)^{n}'

    # Chemical reaction (acceleration and deceleration autocatalytic) 
    def gen_func_2(n, m):
        return lambda x: (x**m) * (1 - x)**n, lambda _: f'x^{m} * (1-x)^{n}'

    # Nucleation or Chemical reaction (acceleration) 
    def gen_func_3(m):
        return lambda x: (x**m), lambda _: f'x^{m}'

    # Sigmoid rate reaction, random nucleation and growth
    def gen_func_4(n):
        return lambda x: (-np.log(1 - x))**n * (1 - x), lambda _: f'(-ln(1-x))^{n} * (1-x)'

    functions = []
    names = []

    # Chemical reaction (deceleration)   
    for n in [1/3, 3/4, 1, 4/3, 3/2, 2, 3, 4]:
        func, name = gen_func_1(round(n, 3))
        functions.append(func)
        names.append(name)

    # Chemical reaction (acceleration and deceleration autocatalytic)
    for n in [1/3, 3/4, 1, 4/3, 3/2, 2, 3, 4]:
        for m in [-1/2, 1/2, 2/3, 3/4, 1]:
            func, name = gen_func_2(round(n, 3), round(m, 3))
            functions.append(func)
            names.append(name)
         
    # Chemical reaction (acceleration)
    for m in [-1/2, 1/2, 2/3, 3/4, 1]:
        func, name = gen_func_3(round(m, 3))
        functions.append(func)
        names.append(name)
        
    # Sigmoid rate reaction (random nucleation and growth)
    for n in [-3, -2, -1, 1/3, 1/2, 2/3, 3/4]:
        func, name = gen_func_4(round(n, 3))
        functions.append(func)
        names.append(name)

    return [functions], [names]

#####################################################################################
# Main Code
#####################################################################################
# Prepare data for SINDy
df = pd.read_csv('data_'+str(option)+'.csv')

xdot_train_multi = [df[['xdot']].values]
x_train_multi = [df[['x']].values]
t_train_multi = [df['t'].values]
u_train_multi = [df[['T']].values]

# Generate functions and names
library_functions_, library_function_names_ = generate_functions_and_names()
library_functions = list(itertools.chain(*library_functions_))
library_function_names = list(itertools.chain(*library_function_names_))

# Create and fit custom feature library
feature_lib = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)
feature_lib.fit(x_train_multi)
n_features = feature_lib.n_output_features_
print("# Features: ", n_features)
print("------------------------")
for i, x in enumerate(x_train_multi):
    print(f"Trajectory {i} shape: {x.shape}")


# SINDy Model using Lasso with positive constraint
optimizer = Lasso(alpha=1.5E-7, max_iter=10000, fit_intercept=False, positive=True)
model = ps.SINDy(optimizer=optimizer, feature_library=feature_lib)
model.fit(x_train_multi, t=t_train_multi)

# Get model coefficients and score
coefficients = model.coefficients()
feature_names = model.get_feature_names()
try:
    score = model.score(x_train_multi, t=t_train_multi)
except Exception as e:
    print(f"Warning: model.score failed with error: {e}")
    score = 0


#####################################################################################
# Reporting
#####################################################################################
# Filter for non-zero coefficients
non_zero_indices = np.nonzero(coefficients.flatten())[0]
non_zero_coefficients = coefficients.flatten()[non_zero_indices]
non_zero_features = np.array(feature_names)[non_zero_indices]

# Create a DataFrame for non-zero coefficients and features
df_non_zero = pd.DataFrame({
    'Feature': non_zero_features,
    'Coefficient': non_zero_coefficients
})
df_non_zero = df_non_zero.sort_values(by='Coefficient', ascending=False)
print(df_non_zero)
print("------------------------")
try:
    s = model.score(x_train_multi, t=t_train_multi)
    print("Model Score: %", round(100 * s, 4))
except Exception as e:
    print("Model Score: calculation failed")


#####################################################################################
# Plotting
#####################################################################################
# Initialize variables
x_values = df['x'].values
t_values = df['t'].values
dx_dt_pred = np.zeros(len(x_values))
feature_values = np.zeros((len(x_values), len(non_zero_features)))

# Compute predicted dx/dt
for i, (feature_name, coef) in enumerate(zip(non_zero_features, non_zero_coefficients)):
    for func, name in zip(library_functions, library_function_names):
        if name(0) == feature_name:
            feature_values[:, i] = coef * func(x_values)
            dx_dt_pred += feature_values[:, i]

# Plot each feature multiplied by its coefficient
plt.figure(figsize=(13, 9))
ax = plt.gca() 
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
for i in range(feature_values.shape[1]):
    plt.plot(t_values, feature_values[:, i], '--', label=f'Feature: {non_zero_features[i]}, Coefficient: {non_zero_coefficients[i]:.4f}', linewidth=4)
plt.plot(t_values, df['xdot'].values, label='Original dx/dt', color='#EACF73', linewidth=4)
plt.plot(t_values, dx_dt_pred, label='Predicted dx/dt', color='#FF2F92', linewidth=4)

# Customize the axes, labels, and title with white text
plt.xlabel('Time', color='white', fontsize=20)
plt.ylabel('dx/dt', color='white', fontsize=20)
plt.title('Features Multiplied by Coefficients Over Time', color='white', fontsize=20)

for spine in ax.spines.values():
    spine.set_edgecolor('white')

# Set the tick colors to white for better visibility
plt.tick_params(axis='x', colors='white', labelsize=20)
plt.tick_params(axis='y', colors='white', labelsize=20)

# Customize the legend to fit the black background
plt.legend(fontsize=20, loc='upper right', frameon=False, labelcolor='white')

# plt.show()
plt.savefig('plot.png', facecolor='black', dpi=300)

