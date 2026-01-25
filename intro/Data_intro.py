#####################################################################
# Date:         June 2024
# Author:       Navid Zobeiry, navidz@uw.edu
# Institution:  University of Washington, Seattle, WA
# Website:      http://composites.uw.edu/AI/
#####################################################################

# Import Libraries
import numpy as np
import pandas as pd

#################################################################
# Main Code
#################################################################
# Create a 3x3 list
my_list = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

# Print the list
print("Python List:")
for row in my_list:
    print(row)

# Convert the list to a NumPy array
my_array = np.array(my_list)
print("\nNumPy Array:")
print(my_array)

# Convert the list to a pandas DataFrame
my_dataframe = pd.DataFrame(my_list)
my_dataframe.columns = ['A', 'B', 'C']
print("\nPandas DataFrame:")
print(my_dataframe)

# Access a specific cell (1, 2) in each data structure
print("\nAccessing a cell:")
print("my_list[1][2]:", my_list[1][2]) # Access cell in list
print("my_array[1, 2]:", my_array[1, 2]) # Access cell in NumPy array
print("my_dataframe.iloc[1, 2]:", my_dataframe.iloc[1, 2]) # Access cell in DataFrame using iloc
print("my_dataframe.at[1, 'C']:", my_dataframe.at[1, 'C']) # Access cell in DataFrame using at
