# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:48:05 2019

@author: Bryan
"""

# In[34]:

import numpy as np
import pandas as pd
import time, os

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# To visualize the RNN network
from keras.utils.vis_utils import plot_model

import utility  # Contains various helper utility functions
# In[35]:
# Hardware (GPU or CPU)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line and restart the kernel)

device_name = tf.test.gpu_device_name()

if device_name:
    print('GPU device found: {}. Using GPU'.format(device_name))
else:
    print("GPU device not found. Using CPU")

# In[36]:
# Variables that determines how the script behaves

# Data was convereted from CSV to HDF then truncated
hdf_key = 'my_key'

# Change the following to point to proper local paths
truncated_train_hdf_file = '../Spyder Code/truncated_train_hdf.h5' 
validation_hdf_file = '../Spyder Code/validation_hdf.h5'
test_hdf_file = '../Spyder Code/test_hdf.h5'

# Folder to save results
results_dir = 'results/current_run'

do_plot_series = False   # Whether training, validation, testing series should be plotted (time and memory consuming: keep this disabled unless necessary)
# In[37]:
# Tunable parameters relating to the operation of the algorithm

# Data preprocessing
scaling_type = 'None'   # Supports: None, StandardScaler, MinMaxScaler

# LSTM network architecture
time_steps = 100
rnn_layer_units = [50, 50, 20]   # The length of this list = no. of hidden layers
rnn_layer_dropout_rate = [0.2, 0.2, 0]   # Dropout rate for each layer (0 for no dropout)

# Training
epochs = 10
batch_size = 64

# Some checks to ensure the parameters are valid
assert len(rnn_layer_units) == len(rnn_layer_dropout_rate)
# In[9]:
train_df = utility.read_hdf(truncated_train_hdf_file, hdf_key)
valid_df = utility.read_hdf(validation_hdf_file, hdf_key)
test_df = utility.read_hdf(test_hdf_file, hdf_key)
# In[19]:
train_df.max()
train_df.min()
# In[11]:
# Importing the training set
"""Temporary: we downsample the training datatset to reduce time!"""
down_sample=10000

dataset_train = train_df.iloc[::down_sample,:]
training_set = dataset_train.iloc[:, 0:2].values
print("Training will be performed on downsampled dataset which consists of ",dataset_train.shape[0],
      " examples out of the original number of training examples which is ", train_df.shape[0])

utility.plot_series(dataset_train, "Downsampled training series", results_dir)
# In[22]:

dataset_train.info()
dataset_train.head()
# In[12]:
# Feature Scaling
print('Scaling the training set. scaling_type={}'.format(scaling_type))
t0 = time.time()
    
if scaling_type == 'None':
    training_set_scaled = training_set
else:
    if scaling_type == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaling_type == 'StandardScaler':
        scaler = StandardScaler()
    
    signal_scaled = scaler.fit_transform(training_set[:,0].reshape(-1,1))

    training_set_scaled = training_set.copy()   # May not be necessary
    training_set_scaled[:,0] = signal_scaled.reshape(-1)

print('Scaling complete. time_to_scale={:.2f} seconds'.format(time.time() - t0))
# In[13]:
# Creating the training dataset (X_train and y_train) 
# X_train is a numpy array with some no. of examples. Each example is a seismic signal window of length time_steps
# y_train has the same no. of examples. Each example is the time_to_eq value that corresponds to the last element of seismic signal window (just 1 value)

print('Preparing input to the RNN (training set)')
t0 = time.time()

X_train = []
y_train = []
    
for i in range (time_steps, training_set_scaled.shape[0]): 
    X_train.append (training_set_scaled[i - time_steps:i, 0])
    y_train.append (training_set_scaled[i, 1])
X_train, y_train = np.array (X_train), np.array (y_train)

# Reshaping since RNN accepts 3d input
X_train = np.reshape (X_train, (X_train.shape[0], X_train.shape[1], 1))
print ("The 3d shape necessary for RNN's input is ", X_train.shape, " . Note how the number of examples is reduced by the defined time steps, i.e. ", time_steps)

assert X_train.shape[1] == time_steps

print('Preparing input complete. time_to_prepare={:.2f} seconds'.format(time.time() - t0))

# In[15]:
# Initialising the RNN
regressor = Sequential ()

# Adding the hidden layers as given in the parameters

for i, (units, dropout_rate) in enumerate(zip(rnn_layer_units, rnn_layer_dropout_rate)):
    # Common args for all layers
    input_shape = (None,)
    return_sequences = True
    
    # Set special args for first and last layer
    if i == 0:  # First hidden layer
        input_shape = (time_steps, 1)
    if i == len(rnn_layer_units) - 1:   # Last hidden layer
        return_sequences = False
        
    regressor.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
    regressor.add (Dropout(dropout_rate))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.summary()
# In[16]:
print('Training the RNN with the training set')
t0 = time.time()

history = regressor.fit (X_train, y_train, epochs=epochs, batch_size=batch_size)

time_to_train = time.time() - t0
print('Training complete. time_to_train={:.2f} seconds ({:.2f} minutes)'.format(time_to_train, time_to_train/60))
# In[17]:
# Save the final trained model (in case we need to continue training from this point on)

model_filepath = results_dir + '/final_model.h5'
regressor.save(model_filepath, overwrite=True)

print('RNN model saved to {}'.format(model_filepath))
# In[19]:
utility.plot_training_history(history, results_dir)

model_plot_filename = results_dir + '/' + 'rnn_plot.png'
plot_model(regressor, to_file=model_plot_filename, show_shapes=True, show_layer_names=True)

print('RNN plot saved to {}'.format(model_plot_filename))
# In[20]:
# Import validation set
"""Temporary: we downsample the testing datatset to reduce time!"""


dataset_test = valid_df.iloc[::down_sample,:]
true_test_time = dataset_test.iloc[:,1].values
print("Validation will be performed on truncated dataset which consists of ",dataset_test.shape[0],
      " examples out of the original number of training examples which is ", valid_df.shape[0])
# In[37]:
#dataset_test.info()
#dataset_test.head()
# In[21]:
#Because we have time_steps time steps and we we want to predict the first entry of time_to_failure in the validation set, we have to look back time_steps samples. 
#Hence, we get these time_steps past samples from the training set. This is why we first concatenate both training and validation. This step may be omitted if we just need to predict one value
#for the whole test set (such as in the provided test files where one value is only needed so we can look back in the same data provided ) 
dataset_total = pd.concat((dataset_train['acoustic_data'], dataset_test['acoustic_data']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_steps:].values
inputs = inputs.reshape(-1,1)

# Feature Scaling
if scaling_type == 'None':
    inputs_scaled=inputs
else:
    print('Scaling the inputs set. scaling_type={}'.format(scaling_type))
    t0 = time.time()
    inputs_scaled = scaler.transform(inputs) 
    print('Scaling complete. time_to_scale={:.2f} seconds'.format(time.time() - t0))

inputs_scaled.shape # So we end up with input size = size of validation set + time_steps
# In[22]:
X_test = []

for i in range(time_steps, inputs_scaled.shape[0]):
    X_test.append(inputs_scaled[i-time_steps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape

# In[28]:
# Predict on test set

print('Predicting on the test set using the trained RNN')
t0 = time.time()
test_predicted_time = regressor.predict(X_test)
#predicted_time = sc.inverse_transform(predicted_time)
print('Predicting on the test set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))
# In[29]:
# Save predictions on test set

test_prediction = pd.DataFrame(test_predicted_time)
test_pred_filename = results_dir + '/' + 'test_prediction.csv'
test_prediction.to_csv(test_pred_filename)
print('Predictions on test set saved to {}'.format(test_pred_filename))
# In[31]:
# Visualize predictions on test set

test_res_plot_filename = results_dir + '/' + 'test_true_vs_pred' + '.png'
utility.plot_results(true_test_time, test_prediction, 'True vs predicted time_to_earthquake on test set', test_res_plot_filename)
# In[32]:
# Compute error metrics on test set

test_mse = mean_squared_error(true_test_time, test_predicted_time)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(true_test_time, test_predicted_time)

print('Error metrics on test set. test_mse: {:.4f}, test_rmse: {:.4f}, test_mae: {:.4f}'.format(test_mse, test_rmse, test_mae))
