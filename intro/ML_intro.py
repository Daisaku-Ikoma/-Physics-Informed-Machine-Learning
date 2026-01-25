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
from matplotlib import cm
import scipy.linalg
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings("ignore") 

#################################################################
# Options
#################################################################
# Choose an option
# 1: Show the hidden truth
# 2: OFTA solution (one factor at a time)
# 3: DOE solution (design of experiments)
# 4: ML solution (machine learning)
option = 2

#################################################################
# Main Code
#################################################################
LB = -18            #Domain lower bound
UB = 10             #Domain upper bound
CP = (UB+LB)/2      #Domain center point 

# Define the hidden function that takes two inputs, x and y, and returns z
def hidden_function (x,y):
    z = np.sin(x/2) + np.sin(x/4) + np.sin(y/2) + np.sin(y/8)
    return z

# If option is set to 1, plot the hidden true response
if option ==1:
    # Create a meshgrid of X and Y values within the defined domain using the minimum search distance
    X = np.linspace(LB, UB, num=100)
    Y = np.linspace(LB, UB, num=100)
    X, Y = np.meshgrid(X, Y)
    
    # Calculate the Z values using the hidden function
    Z = hidden_function(X,Y)  
    
    # Print the maximum value of Z
    print ("max value: %.2f" % Z.max()) 
    
    # Create a 3D plot of the hidden function's surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 10))   
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')    
    ax.plot_surface(X, Y, Z, cmap=cm.plasma,alpha=1,rstride=2,cstride=2,linewidth=0.25, edgecolors='gray')   
    ax.set_xlim(LB,UB)
    ax.set_ylim(LB,UB)

    # Set the viewing options of the plot    
    ax.set_xlabel("X", fontsize=25, color='white')
    ax.set_ylabel("Y", fontsize=25, color='white')
    ax.set_zlabel("Z", fontsize=25, color='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors='white', labelsize=15)       
    ax.set_zlim(-5,5)
    ax.view_init(elev=30, azim=-130) 

    # Show the plot    
    plt.show()
    #fig.savefig('plot.png', dpi=300)
         
# If option is set to 2, use the OFAT approach
if option ==2:  
    # Set the starting point and distance to evaluate
    x0 = CP
    y0 = CP
    dL = (UB-LB)/10  
    
    # Keep one parameter constant (x) and vary the other (y)  
    X1 = np.array([x0 - 4 * dL, x0 - 2 * dL, x0, x0 + 2 * dL, x0 + 4 * dL])
    Y1 = np.array([y0, y0, y0, y0, y0])
    Z1 = hidden_function(X1, Y1)    

    # Keep another parameter constant (y) and vary the other (x)
    X2 = np.array([x0, x0, x0, x0, x0])
    Y2 = np.array([y0 - 4 * dL, y0 - 2 * dL, y0, y0 + 2 * dL, y0 + 4 * dL])
    Z2 = hidden_function(X2, Y2)  
    
    # Report max value
    print("max value: %.2f" % max(Z1.max(), Z2.max()))
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(13, 10))

    # Set background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plotting data with specified colors
    ax.scatter(X1, Z1, s=50, c="#EACF73", label="x variable, y=-4")  # Yellow
    ax.plot(X1, Z1, c="#EACF73")

    ax.scatter(Y2, Z2, s=50, c="#0096FF", label="y variable, x=-4")  # Blue
    ax.plot(Y2, Z2, c="#0096FF")

    # Set axis limits
    ax.set_xlim(LB, UB)
    ax.set_ylim(-5, 5)

    # Set labels with specified font properties
    ax.set_xlabel('x or y', fontsize=25, color='white')
    ax.set_ylabel('z=f(x, y)', fontsize=25, color='white')

    # Set ticks color and font size
    ax.tick_params(colors='white', labelsize=25)

    # Set axis spines to white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Set legend with specified font properties and white text
    legend = ax.legend(frameon=False, prop={'size': 25})
    for text in legend.get_texts():
        text.set_color('white')

    # Display the plot
    plt.show()
    
# If option is set to 3, use the DOE approach
if option ==3: 
    # Select points based on centeral composite design     
    X0 = np.array([LB,LB,LB,CP,CP,CP,UB,UB,UB])
    Y0 = np.array([LB,CP,UB,LB,CP,UB,LB,CP,UB])
    Z0 = hidden_function(X0,Y0)
    data = np.stack((X0,Y0,Z0),axis=1)    

    # Create a regular grid covering the domain of the data
    X,Y = np.meshgrid(np.linspace(LB, UB, num=100), np.linspace(LB, UB, num=100))
    XX = X.flatten()
    YY = Y.flatten()

    # Best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    print ("max value: %.2f" % Z.max())
    
    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 10)) 
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')      
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color="#EACF73")    
    ax.scatter(X0, Y0, Z0, s=40, c="#0096FF", alpha=0.9, marker='o') 
    ax.set_xlim(LB,UB)
    ax.set_ylim(LB,UB)
    ax.set_xlabel("X", fontsize=25, color='white')
    ax.set_ylabel("Y", fontsize=25, color='white')
    ax.set_zlabel("Z", fontsize=25, color='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors='white', labelsize=15)       
    ax.set_zlim(-5,5)
    ax.view_init(elev=30, azim=-130) 
    plt.show()
    
# If option is set to 4, use the ML approach
if option ==4:  
    #Minimum distance to consider in search
    dS = (UB-LB)/100    

    # Generate 50 random points within LB and UB, ensuring they are not closer than dS
    points = []
    while len(points) < 50:
        xi, yi = np.random.uniform(LB, UB, size=2)
        if all(((xi - xj)**2 + (yi - yj)**2)**0.5 >= 10*dS for xj, yj in points):
            points.append((xi, yi))
    x0, y0 = np.array(points).T

    # Evaluate the first 4 random points
    print("No.\tType\t    X\t    Y\t    Z")
    df = pd.DataFrame(columns=["X", "Y", "Z"])
    for ii in range(4):
        xi = x0[ii] 
        yi = y0[ii] 
        zi = hidden_function(xi, yi)        #doing the test
        df.loc[len(df.index)] = [xi, yi, zi] 
        print(f"{ii+1}\t{'Init'}\t{xi:7.2f}\t{yi:7.2f}\t{zi:7.2f}")

    # ML training   
    for i in range (5,35): 
        # Select training X and Y 
        X_train = df[['X','Y']].values
        Z_train = df['Z'].values
        
        # Select a Kernel
        kernel = kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) 
            
        # Train GPR
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
        gpr.fit(X_train, Z_train)
        
        # Prepare a meshgrid data to predict with trained GPR and plot
        n = 250
        X0 = np.meshgrid(np.linspace(LB, UB, n), np.linspace(LB, UB, n))   
        X = np.hstack(((X0[0]).reshape(-1,1), (X0[1]).reshape(-1,1)))            
        # Mean and standard deviation
        Z, sigma = gpr.predict(X, return_std=True)
        # 95 percentile
        Z_95p = Z + 1.6450 * sigma
        # 5 percentile
        Z_5p = Z - 1.6450 * sigma
    
        # Find the potential location of the maximum value based on the 95 percentile GPR surface
        i_max = np.argmax(Z_95p)
        xi = X[i_max,0]
        yi = X[i_max,1]
        
        # Check if this point is close to an existing evaluated point
        i_result = True
        for j in range(0,len(df)):
            df_temp = df.iloc[j]
            if abs((df_temp["X"]-xi)**2+(df_temp["Y"]-yi)**2)**0.5<3*dS:
                i_result = False 
                
        # If too close, evaluate a new random point
        if not i_result:
            ii += 1
            xi = x0[ii] 
            yi = y0[ii] 
            zi = hidden_function(xi,yi)
            df.loc[len(df.index)] = [xi, yi, zi] 
            print(f"{i}\t{'Rand'}\t{xi:7.2f}\t{yi:7.2f}\t{zi:7.2f}")            
        
        # If not too close, evaluate this point as a targeted evaluation
        if i_result:
            zi = hidden_function(xi,yi)
            df.loc[len(df.index)] = [xi, yi, zi]        
            print(f"{i}\t{'ML'}\t{xi:7.2f}\t{yi:7.2f}\t{zi:7.2f}") 
        
        # Set a limit to break the search
        if zi>1.75:
            break; 
        
    #print(f"trained params : {gpr.kernel_}")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 10))   
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')   
    ax.scatter(df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values, s=200, c="#0096FF", zorder=10, alpha=1)   
    ax.plot_surface(X[:,0].reshape(n,n),X[:,1].reshape(n,n),Z_95p.reshape(n,n),alpha=0.25,color='white')  
    ax.plot_surface(X[:,0].reshape(n,n),X[:,1].reshape(n,n),Z_5p.reshape(n,n),alpha=0.25,color='white')    
    ax.plot_surface(X[:,0].reshape(n,n),X[:,1].reshape(n,n),Z.reshape(n,n),cmap=cm.plasma,alpha=0.75,color='blue',rstride=5,cstride=5,linewidth=0.25, edgecolors='gray')
      
    ax.set_xlim(LB,UB)
    ax.set_ylim(LB,UB)
    ax.set_xlabel("X", fontsize=25, color='white')
    ax.set_ylabel("Y", fontsize=25, color='white')
    ax.set_zlabel("Z", fontsize=25, color='white') 
    ax.tick_params(colors='white', labelsize=15) 
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False        
    ax.set_zlim(-5,5)
    ax.view_init(50,-130)   
    plt.show()