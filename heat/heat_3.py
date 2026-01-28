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
data_file = 'data.csv'
df = pd.read_csv(data_file)

X_columns = ['tool_thickness', 'heat_transfer_coefficient']
Y_column = 'max_temperature'

#####################################################################
# Functions
#####################################################################
@tf.function

def custom_loss_with_constraint(y_true, y_pred, X_constraint1, X_constraint2, model):
    # Calculate the mean squared error loss
    mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

    # Compute first derivative for the first column of X_constraint1
    with tf.GradientTape() as tape1:
        tape1.watch(X_constraint1)
        y_pred_constraint1 = model(X_constraint1, training=True)
    first_derivative_x1 = tape1.gradient(y_pred_constraint1, X_constraint1)[:, 0]
    constraint_loss1 = tf.reduce_sum(tf.maximum(0.0, first_derivative_x1))

    # Compute first derivative for the second column of X_constraint2
    with tf.GradientTape() as tape2:
        tape2.watch(X_constraint2)
        y_pred_constraint2 = model(X_constraint2, training=True)
    first_derivative_x2 = tape2.gradient(y_pred_constraint2, X_constraint2)[:, 1]
    constraint_loss2 = tf.reduce_sum(tf.abs(first_derivative_x2))

    # Combine the mean squared error and physics-based loss
    lambda1 = 1
    lambda2 = 1
    total_loss = mse_loss + lambda1 * constraint_loss1 + lambda2 * constraint_loss2

    return mse_loss, constraint_loss1, constraint_loss2, total_loss

def build_and_train_nn(X_train, Y_train, X_test, Y_test, X_constraint1, X_constraint2):
    # Build a neural network model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        *[Dense(16, activation='relu') for _ in range(3)],  # Three hidden layers
        Dense(1)
    ])

    penalty_weight = tf.Variable(1.0, dtype=tf.float32)
    ii = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def custom_loss(y_true, y_pred):
        ii.assign_add(1)
        mse_loss, constraint_loss1, constraint_loss2, total_loss = custom_loss_with_constraint(y_true, y_pred, X_constraint1, X_constraint2, model)
        if ii % 10 == 0:
            tf.print("MSE Loss:", tf.round(mse_loss * 1000) / 1000,
                     "Constraint Loss1:", tf.round(constraint_loss1 * 1000) / 1000,
                     "Constraint Loss2:", tf.round(constraint_loss2 * 1000) / 1000,
                     "Total Loss:", tf.round(total_loss * 1000) / 1000)
        return total_loss

    model.compile(optimizer=Adamax(learning_rate=0.01), loss=custom_loss)
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

# Option for enforcing a physics-informed loss1
tool_thickness_mesh = np.linspace(0.001, 0.030, 10).astype(np.float32)
heat_transfer_coefficient_mesh = np.linspace(10, 100, 10).astype(np.float32)
X_all = np.array(np.meshgrid(tool_thickness_mesh, heat_transfer_coefficient_mesh)).T.reshape(-1, 2)
np.random.seed(42)  # for reproducibility
X_constraint = X_all[np.random.choice(X_all.shape[0], 100, replace=False)]
X_constraint_scaled = scaler_X.transform(X_constraint).astype(np.float32)

# Option for enforcing a physics-informed loss1
tool_thickness_mesh = np.linspace(0.035, 0.05, 10).astype(np.float32)
heat_transfer_coefficient_mesh = np.linspace(50, 100, 10).astype(np.float32)
X_all = np.array(np.meshgrid(tool_thickness_mesh, heat_transfer_coefficient_mesh)).T.reshape(-1, 2)
np.random.seed(42)  # for reproducibility
X_constraint2 = X_all[np.random.choice(X_all.shape[0], 100, replace=False)]
X_constraint_scaled2 = scaler_X.transform(X_constraint2).astype(np.float32)

# Build and train the neural network model
nn_model, nn_r2_test = build_and_train_nn(X_train, Y_train, X_test, Y_test, X_constraint_scaled, X_constraint_scaled2)
print(f"Neural Network R2 Test Score: {nn_r2_test}")

# Generate meshgrid for plotting
tool_thickness_range = np.linspace(0.001, 0.05, 100)
heat_transfer_coefficient_range = np.linspace(10, 100, 100)
tool_thickness_mesh, heat_transfer_coefficient_mesh = np.meshgrid(tool_thickness_range, heat_transfer_coefficient_range)
input_mesh = np.c_[tool_thickness_mesh.ravel(), heat_transfer_coefficient_mesh.ravel()]

# Scale the input mesh
input_mesh_scaled = scaler_X.transform(input_mesh)

# Predict using the trained model and inverse scale the predictions
predicted_mesh = scaler_Y.inverse_transform(nn_model.predict(input_mesh_scaled)).reshape(tool_thickness_mesh.shape)
    
# Load the entire dataset
df_all = pd.read_csv('data_all.csv')

# Plotting
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.plot_surface(tool_thickness_mesh, heat_transfer_coefficient_mesh, predicted_mesh, color='#FF2F92', alpha=0.50, rstride=100, cstride=100)
ax.scatter(df['tool_thickness'], df['heat_transfer_coefficient'], df['max_temperature'], c='#FF2F92', marker='o', alpha=0.75, label='Training Data', s=20)
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
