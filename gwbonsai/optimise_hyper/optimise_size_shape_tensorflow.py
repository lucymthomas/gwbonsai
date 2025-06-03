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

# Define the model
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.optimizers import Adam
import tensorflow
tensorflow.random.set_seed(123456)
from contextlib import redirect_stdout
from keras.layers import Dense, BatchNormalization, Dropout
import tensorflow as tf

def create_shape_model(trial, input_dim, output_dim, shape_options_dict, fixed_dict):
    model = Sequential()

    nodes_per_layer = trial.suggest_categorical('nodes_per_layer', shape_options_dict['nodes_per_layer'])
    num_layers = trial.suggest_categorical('num_hidden_layers', shape_options_dict['num_hidden_layers']) 
    num_layers_1 = trial.suggest_int('num_layers_1', 1, num_layers - 1)
    num_layers_2 = num_layers - num_layers_1
    dropout = trial.suggest_categorical('dropout', shape_options_dict['dropout'])
    dropout_rate_selected = trial.suggest_categorical("dropout_rate", shape_options_dict['dropout_rate'])
    
    model.add(Dense(nodes_per_layer, input_dim=input_dim, kernel_initializer=fixed_dict['weight_init'], activation=fixed_dict['activation']))
    if fixed_dict['normalisation'] > 0.5:
        model.add(BatchNormalization())
    for i in range(num_layers_1):
        model.add(Dense(nodes_per_layer, kernel_initializer=fixed_dict['weight_init'], activation=fixed_dict['activation']))
    if dropout > 0.5:
        model.add(Dropout(dropout_rate_selected, seed=123))
    for i in range(num_layers_2):
        model.add(Dense(nodes_per_layer, kernel_initializer=fixed_dict['weight_init'], activation=fixed_dict['activation']))
    model.add(Dense(output_dim, activation='linear'))
    return model

# Define the optimiser
def create_shape_optimiser(trial, fixed_dict):
    kwargs={}
    optimiser_selected = fixed_dict['optimiser']
    kwargs["learning_rate"] = fixed_dict['learning_rate']
    
    optimiser = getattr(tf.optimizers, optimiser_selected)(**kwargs)
    return optimiser

# Define the objective function
def shape_objective(trial, input_dim, output_dim, shape_options_dict, fixed_dict, x_train, train_eim_data, x_validation, val_eim_data, x_test, test_eim_data, eim):
    model = create_shape_model(trial, input_dim, output_dim, shape_options_dict, fixed_dict)
    optimiser = create_shape_optimiser(trial, fixed_dict)

    model.compile(loss='mean_squared_error', optimizer=optimiser, metrics=['mean_squared_error'])

    #es = EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', verbose=0, patience=50)
    batch_size = trial.suggest_categorical('batch_size', shape_options_dict['batch_size'])
    model.fit(x_train, train_eim_data, epochs=fixed_dict['num_epochs'], batch_size=batch_size, verbose=False,
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