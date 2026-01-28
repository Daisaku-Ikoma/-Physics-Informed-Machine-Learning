#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

#####################################################################################
# Input parameters
#####################################################################################
# Defining inputs and outputs
script_dir = os.path.dirname(__file__)
data_file = os.path.join(script_dir, '3Dprint_data.csv')

# Selection option: 1 = traditional ML, 2: Physics-informed ML
option = 1

if option == 1:
    # Inputs
    X_columns = ['nozzle_temp', 'printing_speed', 'layer_thickness', 'nozzle_diameter']
if option == 2:
    # Inputs
    X_columns = ['nozzle_temp', 'printing_speed', 'layer_thickness', 'nozzle_diameter', 'Flow_Index']
  
# Output
Y_columns = ['tensile_strength']

#####################################################################################
# Functions
#####################################################################################
def surrogate_models(X, Y):  
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Dictionary to store model names and their R² scores
    results = []
    best_score = -np.inf  # Initialize best_score
    best_model = None
    
    # Regressors configurations
    model_configs = [
        {
            "name": "Random Forest",
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [5, 10, 25, 50],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        },
        {
            "name": "Gradient Boosting Machine",
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [5, 10, 25, 50],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
            }
        },
        {
            "name": "DecisionTreeRegressor",
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [None, 5, 10, 25, 50],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 6],
            }
        },
        {
            "name": "XGBRegressor",
            "model": XGBRegressor(random_state=42),
            "params": {
                "n_estimators": [10, 25, 50],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
        }
    ]

    ################################################### 
    # Various models in model_configs
    for config in model_configs:
        rnd_search_cv = RandomizedSearchCV(config["model"], config["params"], n_iter=30, cv=5, scoring='r2', random_state=42, n_jobs=-1)
        rnd_search_cv.fit(X_train, Y_train)
        model = rnd_search_cv.best_estimator_
        
        # Predict on training and testing data
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
        
        # Calculate R2 for training and testing data
        r2_train = round(r2_score(Y_train, Y_pred_train) * 100, 1)
        r2_test = round(r2_score(Y_test, Y_pred_test) * 100, 1)    
        results.append({"Model": config["name"], "R2_test": r2_test, "R2_train": r2_train})
        
        if r2_test > best_score:
            best_score = r2_test
            best_model = model

    ################################################### 
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)
    
    # Predict on training and testing data
    Y_pred_train = linear_model.predict(X_train)
    Y_pred_test = linear_model.predict(X_test)
    
    # Calculate R2 for training and testing data
    r2_train = round(r2_score(Y_train, Y_pred_train) * 100, 1)
    r2_test = round(r2_score(Y_test, Y_pred_test) * 100, 1)    
    results.append({"Model": "Linear Regression", "R2_test": r2_test, "R2_train": r2_train})
    
    if r2_test > best_score:
        best_score = r2_test
        best_model = linear_model

    ################################################### 
    # Convert results to DataFrame and sort by R2
    results_df = pd.DataFrame(results).sort_values(by='R2_test', ascending=False).reset_index(drop=True)

    return results_df, best_model, X_train, X_test, Y_train, Y_test

#####################################################################################
# Main Code
#####################################################################################
# Read and filter input file
df = pd.read_csv(data_file)
X = df[X_columns]
Y = df[Y_columns].values.ravel()

# Select the best model
results_df, best_model, X_train, X_test, Y_train, Y_test = surrogate_models(X, Y)
print(results_df)

# Adding predictions to the dataframe
df['predicted_tensile_strength'] = best_model.predict(X)

# Plotting the actual vs predicted tensile_strength
plt.figure(figsize=(13, 9))
ax = plt.gca()
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.scatter(Y_train, best_model.predict(X_train), label='Training data', color='#FF2F92', alpha=0.5, s=100)  # Purple
plt.scatter(Y_test, best_model.predict(X_test), label='Testing data', color='#0096FF', alpha=0.5, s=100)    # Blue
plt.plot([0, 100], [0, 100], linestyle='--', color='white', lw=2) 
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('Tensile Strength (MPa)', color='white', fontsize=20)
plt.ylabel('Predicted Tensile Strength (MPa)', color='white', fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='x', colors='white', labelsize=20)
plt.tick_params(axis='y', colors='white', labelsize=20)
# Set spine color to white
for spine in ax.spines.values():
    spine.set_edgecolor('white')
plt.legend(title='Data', title_fontsize=14, fontsize=20, loc='upper left', frameon=False, labelcolor='white')
plt.show()
#plt.savefig('plot.png', facecolor='black', dpi=300)
