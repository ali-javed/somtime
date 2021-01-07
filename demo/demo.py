
#import libraries
from somtime.SelfOrganizingMap import SelfOrganizingMap
import numpy as np


##############################
# author: Ali Javed
# October 14 2020
# email: ajaved@uvm.edu
#############################
        
        
        

#create a multivariate time series  4,2,4. 4 observations, 2 variables, and 4 time steps each.
#In case there is only  variable, or one list of featrues the number of variables will be 1. Therefore, number of observation, 1, number of features (or time steps in case of a time series).
time_series = [[[1,2,3,4,5],[6,7,8,9,5]], [[0,1,2,3,5],[6,7,8,9,5]], [[0,1,2,3,3],[6,5,7,8,3]], [[1,2,2,3,4],[5,3,1,6,8]]]

time_series = np.asarray(time_series)

#optionall have labels (i.e., ground truth if they are available).
labels = [0,1,0,1]

#optionally have a dictionary defining each label.
legends_dict = {}
legends_dict[0] = 'False'
legends_dict[1] = 'True'

#optionall have targets/key/or ids
targets = [0,1,2,3]




print('Creating Multivariate SOM...')
#############################
# Description: Class initilizer. This function creates an instance of neural network saving the parameters and setting random weights
# inputsSize: number of input nodes needed i.e. 5.
# hiddenSize: number of hidden layer nodes [2,3] will create a 2x3 node grid
########################################
hiddenSize = [10,10]
SOM = SelfOrganizingMap(inputSize = np.shape(time_series)[2], variables= np.shape(time_series)[1],hiddenSize = hiddenSize)


##################################
# Description: Function iterates to organize the Kohonen layer

# inputs: all inputs
# epochs: epochs to iterate for
# path: Path to save SOM plots

# windowSize: windowSize (between 0 and 1) to be used by DTW (for project). Only valid when data is time series, set to 0 otherwise.
# targets: keys or ids to be saved in output file. Can be an empty list
# labels: labels or ground truth, must be integers such as 0,1,2. Can be an empty list
# legends_dict: used in legend to map an integer from labels to its name in string. such as legends_dict[0] = '4 legged animal'
# showPlot: call the plt.show() usually in an editor or jupyter notebook but not command prompt.
#returns
#all_data: list of dictionary objects with 4 keys, 'target','x','y', and 'labels'
# x: x coordinate on SOM
# y: y cordinate on SOM
# target: keys or name of observation if provided.
# label: label (ground truth) of observation if provided.
##################################
fname = 'Demo_'
all_data = SOM.iterate(time_series,epochs = 40,path = fname ,windowSize = 0, targets = targets, labels = labels,legend_dict = legends_dict, showPlot = 'False')

#if you want to get the weights and Umatrix for custom plotting.
weights = SOM.getWeights()
Umatrix = SOM.getUmatrix()
