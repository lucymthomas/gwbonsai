import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import tensorflow.keras as tfkeras
import tensorflow as tf
import time
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout
import logging
logging.basicConfig(filename='build_train_test_log.log', level=logging.INFO) #encoding='utf-8', 
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

## GPU-relevant stuff
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.list_physical_devices('GPU')
tf.test.gpu_device_name()

### Network compiling
def compile_nn(file_prefix, output_shape=1,num_hidden_layers=4, nodes_per_layer=100, activation='relu', lr=1e-3):
    """
    Function to define and compile a sequential keras ANN with user-specified hyperparameters.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)
        method: str, the sampling method, do we want systematic uniform sampling across the whole parameter space, or random sampling? Default is 'mixture',
                can be 'mixture', 'systematic' or 'random'.

    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """
    logging.info('Initiating neural network with parameters:')
    logging.info(str(num_hidden_layers)+' hidden layers,')
    logging.info(str(nodes_per_layer)+' nodes per layer,')
    logging.info(str(activation)+' activation function in hidden layers,')
    logging.info(str(lr)+' learning rate.')
    
    model = tfkeras.Sequential()
    model.add(tfkeras.layers.InputLayer(input_shape=(7,)))

    for layer in range(num_hidden_layers):
        model.add(tfkeras.layers.Dense(nodes_per_layer, activation=activation))
        #model.add(layers.Dropout(0.5))
    
    model.add(tfkeras.layers.Dense(output_shape, activation='linear'))

    logging.info('Compiling the model.')
    model.compile(
    # Optimization algorithm, specify learning rate
    optimizer=tfkeras.optimizers.Adam(learning_rate=lr),
    # Loss function for a binary classifier
    loss=tfkeras.losses.MeanSquaredError(),
    # Diagnostic quantities
    metrics=[tfkeras.metrics.MeanSquaredError()]
    )
    logging.info('Saving summary of compiled model to model_summary.txt.')
    with open(file_prefix+'model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    logging.info('Done, model compile successfully.')
            
    return model


### Network training
def train_nn(file_prefix, training_params, training_mfs, validation_params, validation_mfs, model, num_epochs=1000, batch_size=128, plot=True, standard_scale=False):
    """
    Function to train a pre-compiled sequential keras ANN with user-specified hyperparameters.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)
        method: str, the sampling method, do we want systematic uniform sampling across the whole parameter space, or random sampling? Default is 'mixture',
                can be 'mixture', 'systematic' or 'random'.

    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """
    if standard_scale==True:
        logging.info('Standard scaling the training and validation data.')
        params_scaler = StandardScaler()
        params_scaler.fit(training_params)
        mfs_scaler = StandardScaler()
        mfs_scaler.fit(training_mfs)
        
        training_params = params_scaler.transform(training_params)
        validation_params = params_scaler.transform(validation_params)
        training_mfs = mfs_scaler.transform(training_mfs)
        validation_mfs = mfs_scaler.transform(validation_mfs)
        
        dump(params_scaler, file_prefix+'input_params_scaler.bin', compress=True)
        logging.info('Binary params standard scaler saved to '+file_prefix+'input_params_scaler.bin')
        dump(mfs_scaler, file_prefix+'output_params_scaler.bin', compress=True)
        logging.info('Output params standard scaler saved to '+file_prefix+'output_params_scaler.bin')
        
        
    logging.info('Training model with training hyperparameters:')
    logging.info(str(num_epochs)+' number of epochs,')
    logging.info(str(batch_size)+' batch size.')
    #loggingcallback = [LoggingCallback(logging.info)]
    start = time.time()
    result = model.fit(training_params,training_mfs,
                       epochs=num_epochs,batch_size=batch_size,verbose=True,
                       validation_data=(validation_params, validation_mfs),
                       #callbacks=loggingcallback,
                       shuffle=False)

    end = time.time()
    logging.info('Training took '+str(end-start)+' seconds.')
    
    with open(file_prefix+'training_history', 'wb') as file_pi:
        pickle.dump(result.history, file_pi)
    logging.info('Training history saved to training_history.')
    
    if plot==True:
        plt.plot(result.epoch,result.history['loss'], label='Training loss')
        plt.plot(result.epoch,result.history['val_loss'], label='Validation loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss (MSE)')
        plt.savefig(file_prefix+'training.png')
        logging.info('Training history plot saved to training.png.')
        
    model.save(file_prefix+'model.keras')
    logging.info('Trained model saved to model.keras.')

### Helper class to print out progress of training epochs line by line to log file
# class LoggingCallback(Callback):
#     """Callback for logging metrics at the end of each epoch.
#     Default format looks like "Epoch: 3 - loss: 3.4123 - val_loss: 5.4321"
#     # Arguments
#         print_fcn: function for printing. default is print, which will print to standard output.
#         format_epoch: format string for each epoch [default="Epoch: {} - {}"]
#         format_keyvalue: format string for key value pairs [default="{}: {}"]
#         format_separator= separator string between each pair [default=" - "]
#     # Example
#         ```python
#         # Write to Python logging
#         import logging
#         model.fit(x, y, callbacks = [LoggingCallback(logging.info)]
#         # Write to a file
#         with open("log.txt", 'w') as f:
#             def print_fcn(s):
#                 f.write(s)
#                 f.write("\n")
#             model.fit(x, y, callbacks = [LoggingCallback(print_fcn)]
#         ```
#     """

#     def __init__(self, print_fcn=print,
#                  format_epoch="Epoch: {} - {}",
#                  format_keyvalue="{}: {:0.4f}",
#                  format_separator=" - "):
#         Callback.__init__(self)
#         self.print_fcn = print_fcn
#         self.format_epoch = format_epoch
#         self.format_keyvalue = format_keyvalue
#         self.format_separator = format_separator

#     def on_epoch_end(self, epoch, logs={}):
#         values = self.format_separator.join(self.format_keyvalue.format(k, v) for k, v in iteritems(logs))
#         msg = self.format_epoch.format(epoch, values)
#         self.print_fcn(msg)
