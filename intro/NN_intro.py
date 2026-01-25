#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import necessary functions and libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#################################################################
# Main Inputs
#################################################################
# Define a function to generate the target data
def my_function(x):
    y = (x**2 + x + 1) / 100
    return y

# Number of hidden layers in Neural Network
layers = 1

# Number of nodes per hidden layer in Neural Network
nodes = 1

# Activation function for Neural Network: relu, tanh ...
activation = 'tanh'

# Initialization: Glorot Uniform, He Normal, Random Normal...
initializer = HeNormal()

# Loss Function: mean_squared_error, mean_absolute_error...
loss = 'mean_squared_error'

# Optimizer for training: SGD, RMSProp, Adagrad, Adam, Adamax...
learning_rate = 0.002
my_optimizer = optimizers.SGD(learning_rate)

# Saving the trained NN model
output_model = 'NN_intro.h5'

#################################################################
# Main Program
#################################################################
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate and normalize training data
x_min, x_max, data_number = -10, 10, 100
x = np.random.uniform(x_min, x_max, data_number)
y = my_function(x)
x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1

# Split data into training and validation sets
x_train, x_validate, y_train, y_validate = train_test_split(x_normalized, y, train_size=0.7, random_state=42)

# Create a Neural Network
model = Sequential()
model.add(Dense(nodes, input_dim=1, activation=activation, kernel_initializer=initializer))
for i in range(layers - 1):
    model.add(Dense(nodes, activation=activation, kernel_initializer=initializer))
model.add(Dense(1, kernel_initializer=initializer))
model.compile(loss=loss, optimizer=my_optimizer, metrics=['mse'])

# Fit the model
history = model.fit(x_train, y_train, epochs=1000, batch_size=32, shuffle=True, verbose=2, validation_data=(x_validate, y_validate))

# Save the trained model
model.save(output_model)

# Evaluate the model for the entire dataset
pred = model.predict(x_normalized)
MSE = mean_squared_error(pred, y)
RMSE = (MSE)**0.5
Max_error_val = np.amax(np.absolute(pred - y))
print("------------------------------------------------------------------\n")
print("Root Mean Squared Error (RMSE) for the entire dataset: %.2f" % RMSE)
print("Maximum Absolute Error (MAE) for the entire dataset: %.2f" % Max_error_val)

# # Plot the loss
# plt.plot(history.history['mse'])
# plt.plot(history.history['val_mse'])
# plt.ylabel('Mean Squared Error')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.ylim([0, 0.25])
# plt.show()

#####################################################################
# Predict using the trained model
#####################################################################
x_pred = np.linspace(x_min, x_max, 50)
y_true = my_function(x_pred)

# Normalize
x_pred_normalized = 2 * (x_pred - x_min) / (x_max - x_min) - 1

# Load the trained model and predict
model = keras.models.load_model(output_model)
y_pred = model.predict(x_pred_normalized)

# Plot and compare
plt.figure(figsize=(13, 10))
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_color('white')  # x=0 axis
plt.gca().spines['left'].set_color('white')  # y=0 axis
plt.gca().tick_params(axis='x', colors='white', labelsize=20)
plt.gca().tick_params(axis='y', colors='white', labelsize=20)
plt.plot(x_pred, y_true, color='#FF2F92', linewidth=4, label='True Values')
plt.plot(x_pred, y_pred, color='#EACF73', linewidth=4, label='Predicted Values')
plt.legend(fontsize=20, loc='upper left', frameon=False, labelcolor='white')
#plt.savefig('plot.png', bbox_inches='tight', facecolor='black', dpi=300)
plt.show()
