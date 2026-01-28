#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#####################################################################
# Input parameters
#####################################################################

# Load the data
script_dir = os.path.dirname(__file__)
data_file = os.path.join(script_dir, 'data.csv')
df = pd.read_csv(data_file)

# Define input parameters
part_thickness = 0.01273        # m
T_hold, T0 = 180, 20            # C, ambient and initial temperature
Tdot = 3/60                     # C/sec, heat-up rate
Time = 180                      # min, hold time
part_K, tool_K = 0.167, 51.7    # W/(m K), conductivity
part_Cp, tool_Cp = 1260, 465    # J/(kg K), specific heat capacity
part_ro, tool_ro = 1600, 7860   # kg/m3, density

# Calculate transformed temperature and feature
r, M = Tdot, part_ro*part_Cp*part_thickness + tool_ro*tool_Cp*df["tool_thickness"]
delta = r * M / (2*df["heat_transfer_coefficient"])
t1, t2 = (T_hold-T0)/r, (T_hold-T0)/r + Time*60
T2 = T_hold - delta * (np.exp(-r*(t2-t1)/delta) - np.exp(-r*t2/delta))  
df['max_temperature_transformed'] = df['max_temperature'] - T2
df["feature"] = (2/df["heat_transfer_coefficient"]+(part_thickness/2)/part_K + df["tool_thickness"]/tool_K)/(part_ro*part_Cp*part_thickness+tool_ro*tool_Cp*df["tool_thickness"])
X_columns = ['feature']
Y_column = 'max_temperature_transformed'

#####################################################################
# Functions
#####################################################################

def build_and_train_nn(X_train, Y_train, X_test, Y_test):
    # Build a neural network model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        *[Dense(16, activation='relu') for _ in range(3)],  # Three hidden layers
        Dense(1)
    ])

    model.compile(optimizer=Adamax(learning_rate=0.01), loss='mse')
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=0, validation_data=(X_test, Y_test))

    # Evaluate the model
    Y_pred_test = model.predict(X_test)
    r2_test = round(r2_score(Y_test, Y_pred_test) * 100, 1)
    return model, r2_test

#####################################################################
# Main Code
#####################################################################

# Extract features (X) and target variable (Y)
X = df[X_columns]
Y = df[Y_column].values

# Standardize the data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# Split the main data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.25, random_state=42)

# Build and train the neural network model
nn_model, nn_r2_test = build_and_train_nn(X_train, Y_train, X_test, Y_test)
print(f"Neural Network R2 Test Score: {nn_r2_test}")

# Generate meshgrid for plotting
tool_thickness_range = np.linspace(0.001, 0.05, 100)
heat_transfer_coefficient_range = np.linspace(10, 100, 100)
tool_thickness_mesh, heat_transfer_coefficient_mesh = np.meshgrid(tool_thickness_range, heat_transfer_coefficient_range)

# Calculate T2 for the entire range of tool_thickness and heat_transfer_coefficient
r = Tdot
M = part_ro * part_Cp * part_thickness + tool_ro * tool_Cp * tool_thickness_mesh
delta = r * M / (2 * heat_transfer_coefficient_mesh)
t1 = (T_hold - T0) / r
t2 = t1 + Time * 60
T2_mesh = T_hold - delta * (np.exp(-r * (t2 - t1) / delta) - np.exp(-r * t2 / delta))

# Generate input mesh based on selected option
feature_mesh = (1 / heat_transfer_coefficient_mesh + (part_thickness / 2) / part_K + tool_thickness_mesh / tool_K + 1 / heat_transfer_coefficient_mesh) / (
            part_ro * part_Cp * part_thickness + tool_ro * tool_Cp * tool_thickness_mesh)
input_mesh = np.c_[feature_mesh.ravel()]

# Scale the input mesh
input_mesh_scaled = scaler_X.transform(input_mesh)

# Predict using the trained model and inverse scale the predictions
predicted_mesh = scaler_Y.inverse_transform(nn_model.predict(input_mesh_scaled)).reshape(tool_thickness_mesh.shape)
predicted_mesh = predicted_mesh + T2_mesh

# Load the entire dataset
df_all = pd.read_csv(os.path.join(script_dir, 'data_all.csv'))

# Plotting
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
#ax.plot_surface(tool_thickness_mesh, heat_transfer_coefficient_mesh, T2_mesh, color='#FF9A4D', alpha=0.65, rstride=100, cstride=100)
#ax.scatter(df['tool_thickness'], df['heat_transfer_coefficient'], df['max_temperature_transformed'], c='#FF2F92', marker='o', alpha=0.75, label='Training Data', s=20)
ax.scatter(df['tool_thickness'], df['heat_transfer_coefficient'], df['max_temperature'], c='#FF2F92', marker='o', alpha=0.75, label='Training Data', s=20)
ax.plot_surface(tool_thickness_mesh, heat_transfer_coefficient_mesh, predicted_mesh, color='#FF2F92', alpha=0.65, rstride=100, cstride=100)
ax.scatter(df_all['tool_thickness'], df_all['heat_transfer_coefficient'], df_all['max_temperature'], c='#0096FF', marker='o', alpha=0.75, label='Extrapolation Data', s=20)

# Set labels and style
ax.set_xlabel("Tool Thickness (m)", color='white')
ax.set_ylabel(r"Heat Transfer Coefficient (W/m$^2\cdot$K)", color='white')
ax.set_zlabel(r"Max Temperature ($^\circ$C)", color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
ax.w_xaxis.line.set_color("white")
ax.w_yaxis.line.set_color("white")
ax.w_zaxis.line.set_color("white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_zlim(125, 275)
ax.view_init(elev=15, azim=-60)
ax.legend(fontsize=15, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, labelcolor='white')
plt.show()
#plt.savefig('plot.png', facecolor='black', dpi=300)
