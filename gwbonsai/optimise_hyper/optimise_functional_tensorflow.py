# -*- coding: utf-8 -*-
#
#       Copyright 2025
#       Lucy M Thomas <lmthomas@caltech.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#
#       This script is a part of the GWBonsai package, which builds and optimises 
#       surrogate models for gravitational wave data analysis.
#
#       This file contains routines to optimise the size and shape hyperparameters of a the full 7D
#       configuration of the remnant properties neural network surrogate NRSur7dq4Remnant_NN, 
#       using optuna and a tensorflow backend.
#       It requires the functional optimisation to be run first to optimise the functional hyperparameters.  
#       It requires the optuna package to be installed, as well as training data, validation and testing data.
#       This script is modified from the original version so as to have no explicit 
#       dependence on the gwbonsai routines.

# # Define the model
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.optimizers import Adam
import tensorflow
tensorflow.random.set_seed(123456)
from contextlib import redirect_stdout
from keras.layers import Dense, BatchNormalization, Dropout
import tensorflow as tf

def create_functional_model(trial, input_dim, output_dim, functional_options_dict, fixed_dict):
    model = Sequential()

    num_layers_1 = 2 # These are the default values and will be optimised in the next stage
    num_layers_2 = 2 # These are the default values and will be optimised in the next stage
    
    activation_selected = trial.suggest_categorical("activation", functional_options_dict['activation'])
    weight_init = trial.suggest_categorical('weight_init', functional_options_dict['weight_init'])
    normalisation = trial.suggest_categorical('normalisation', functional_options_dict['normalisation'])
    
    if fixed_dict['dropout'] > 0.5: # if dropout is selected, choose a place in the middle of the network to apply it
        num_layers_1 = trial.suggest_int('num_layers_1', 0, fixed_dict['num_hidden_layers'] - 1)
        num_layers_2 = fixed_dict['num_hidden_layers'] - num_layers_1
    
    model.add(Dense(fixed_dict['nodes_per_layer'], input_dim=input_dim, kernel_initializer=weight_init, activation=activation_selected))
    if normalisation > 0.5:
        model.add(BatchNormalization())
    for i in range(num_layers_1):
        model.add(Dense(fixed_dict['nodes_per_layer'], kernel_initializer=weight_init, activation=activation_selected))
    if fixed_dict['dropout'] > 0.5:
        model.add(Dropout(fixed_dict['dropout_rate'], seed=123))
    for i in range(num_layers_2):
        model.add(Dense(fixed_dict['nodes_per_layer'], kernel_initializer=weight_init, activation=activation_selected))
    model.add(Dense(output_dim, activation='linear'))
    return model

# Define the optimiser
def create_functional_optimiser(trial, functional_options_dict):
    kwargs={}
    optimiser_selected = trial.suggest_categorical('optimiser', functional_options_dict['optimiser'])
    kwargs["learning_rate"] = trial.suggest_categorical("learning_rate", functional_options_dict['learning_rate'])
    
    optimiser = getattr(tf.optimizers, optimiser_selected)(**kwargs)
    return optimiser

# Define the objective function
def functional_objective(trial, input_dim, output_dim, functional_options_dict, fixed_dict, x_train, train_eim_data, x_validation, val_eim_data, x_test, test_eim_data, eim):
    model = create_functional_model(trial, input_dim, output_dim, functional_options_dict, fixed_dict)
    optimiser = create_functional_optimiser(trial, functional_options_dict)
    
    model.compile(loss='mean_squared_error', optimizer=optimiser, metrics=['mean_squared_error'])

    #es = EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', verbose=0, patience=50)
    model.fit(x_train, train_eim_data, epochs=fixed_dict['num_epochs'], batch_size=fixed_dict['batch_size'], verbose=False,
              validation_data=(x_validation, val_eim_data), shuffle=False) #callbacks=[es],

    nodes = model.predict(x_test)
    predicted_test_data = np.dot(nodes, eim.B)
    test_data = np.dot(test_eim_data, eim.B)
    # Calculate the mean absolute error
    predicted_abs_error = np.abs(predicted_test_data - test_data)
    predicted_mean_abs_error = np.mean(np.mean(predicted_abs_error, axis=1), axis=0)

    print("Params:")
    print(trial.params)
    print("Mean Absolute Error: %.5f" % (predicted_mean_abs_error))

    return predicted_mean_abs_error