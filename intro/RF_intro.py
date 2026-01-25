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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz

#################################################################
# Inputs
#################################################################
# Define a function to generate the target data
def my_function(x):
    y = (x**2 + x + 1) / 100
    return y
    
# Define hyperparameter options
n_estimators = 2  
max_depth = 3
max_leaf_nodes = 5  

#################################################################
# Mian Code
#################################################################
# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y = my_function(X).ravel()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,  max_leaf_nodes=max_leaf_nodes,  random_state=42)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, MSE: {mse:.4f}")

# Plot the actual vs predicted values
x = np.linspace(-10, 10, 1000).reshape(-1, 1) 
y = my_function(x).ravel()
y_pred = regr.predict(x)
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('black')  # Set the background color to black
plt.gcf().patch.set_facecolor('black')  # Set the figure's facecolor
plt.plot(x, y, color='#FF2F92', label='Actual')
plt.plot(x, y_pred, color='#EACF73', label='Predicted')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.gca().tick_params(axis='x', colors='white', labelsize=20)
plt.gca().tick_params(axis='y', colors='white', labelsize=20)
plt.legend(fontsize=20, facecolor='black', edgecolor='white', labelcolor='white')
plt.show()
#plt.savefig('plot.png', facecolor='black', dpi=300)

# Visualize all trees from the random forest
for i, tree in enumerate(regr.estimators_):
    # Export the tree to a dot file
    dot_data = export_graphviz(tree, out_file=None, 
                               feature_names=['X'],
                               filled=True, rounded=True,
                               special_characters=True)

    # Add the background color, font color, node border, and fill colors to the DOT data
    dot_data = ('digraph Tree {\n'
                'bgcolor="#000000";\n'  # Set background color to black
                'node [style=filled, fontcolor="#000000", color="#EACF73", fillcolor="#0096FF"];\n'  # Set node colors
                'edge [color="#EACF73"];\n'  # Set edge color to purple
                + dot_data.split('digraph Tree {\n')[1])

    # Use graphviz to render the dot file
    graph = graphviz.Source(dot_data)
    
    # Save and display each tree
    graph.render(f"random_forest_tree_{i+1}")  # This will save each tree as a PDF file
    graph.view()
