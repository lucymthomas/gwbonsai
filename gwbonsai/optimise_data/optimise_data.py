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

import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow
tensorflow.random.set_seed(123456)
from contextlib import redirect_stdout
from keras.layers import Dense, BatchNormalization, Dropout

# Defining our neural network training function
def train_func(best_params, input_dim, output_dim, num_epochs, train_inputs, train_outputs, holdout_inputs, holdout_outputs, test_inputs, test_outputs, eim):
    model = Sequential()
    model.add(Dense(best_params['nodes_per_layer'], input_dim=input_dim, kernel_initializer=best_params['weight_init'], activation=best_params['activation']))
    if best_params['normalisation'] > 0.5:
        model.add(BatchNormalization())
    num_hidden_layers_1 = np.random.randint(1, best_params['num_hidden_layers'])
    num_hidden_layers_2 = best_params['num_hidden_layers'] - num_hidden_layers_1
    for i in range(num_hidden_layers_1):
        model.add(Dense(best_params['nodes_per_layer'], kernel_initializer=best_params['weight_init'], activation=best_params['activation']))
    if best_params['dropout'] > 0.5:
        model.add(Dropout(best_params['dropout_rate'], seed=123))
    for i in range(num_hidden_layers_2):
        model.add(Dense(best_params['nodes_per_layer'], kernel_initializer=best_params['weight_init'], activation=best_params['activation']))
    model.add(Dense(output_dim, activation='linear'))
    
    kwargs={}
    kwargs["learning_rate"] = best_params['learning_rate']    
    optimiser = getattr(tensorflow.optimizers, best_params['optimiser'])(**kwargs)
    
    model.compile(
        # Loss function for a regression
        loss='mean_squared_error', 
        # Optimization algorithm, specify learning rate
        optimizer=optimiser, 
        # Diagnostic quantities
        metrics=['mean_squared_error'])

    # Early stop when the validation MSE has stopped decreasing, with a patience of 50 epochs
    #es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=0, patience=50)
    # Train the model
    model.fit(train_inputs, train_outputs,
                       epochs=num_epochs,batch_size=best_params['batch_size'],verbose=False,
                       #validation_data=(x_validation, val_eim_data),
                       #callbacks=es,
                       shuffle=False)
    
    # Test the model on the holdout parameters, from which the new training data will be drawn
    if np.shape(holdout_inputs)[0] > 0:
        nodes = model.predict(holdout_inputs)
        predicted_holdout_data = np.dot(nodes, eim.B)
        holdout_data = np.dot(holdout_outputs, eim.B)
        # Calculate the mean absolute error
        predicted_abs_error_holdout = np.abs(predicted_holdout_data - holdout_data)
    else:
        predicted_abs_error_holdout = np.array([])

    # Test the model on the test parameters
    nodes = model.predict(test_inputs)
    predicted_test_data = np.dot(nodes, eim.B)
    test_data = np.dot(test_outputs, eim.B)
    # Calculate the mean absolute error
    predicted_abs_error_test = np.abs(predicted_test_data - test_data)

    return predicted_abs_error_holdout, predicted_abs_error_test

def train_iteration(append_sizes, best_params, df_train, df_test, input_cols, output_cols, num_epochs, eim):
    input_dim = len(input_cols)
    output_dim = len(output_cols)
    test_inputs = df_test[input_cols].to_numpy()
    test_outputs = df_test[output_cols].to_numpy()
    total_training_points = np.shape(df_train.to_numpy())[0]
    print('Total number of training points: '+str(total_training_points))
    
    for i in range(len(append_sizes)):
        train_inputs = df_train[input_cols][df_train['first_training_iteration'].notna()].to_numpy()
        train_outputs = df_train[output_cols][df_train['first_training_iteration'].notna()].to_numpy()
        num_training_points = np.shape(train_inputs)[0]
        holdout_inputs = df_train[input_cols][df_train['first_training_iteration'].isna()].to_numpy()
        holdout_outputs = df_train[output_cols][df_train['first_training_iteration'].isna()].to_numpy()
        print('Starting training iteration number '+str(i)+' with '+str(num_training_points)+' training points.')


        error, error_test = train_func(best_params, input_dim, output_dim, num_epochs, train_inputs, train_outputs, holdout_inputs, holdout_outputs, test_inputs, test_outputs, eim)
        print('Calculating mean errors.')
        training_errors = np.zeros((total_training_points,))
        k = 0
        for j in range(total_training_points):
            if np.isnan(df_train['first_training_iteration'].loc[j]):
                training_errors[j] = np.mean(error[k])
                k += 1
            else:
                training_errors[j] = np.nan
        
        df_train['error_'+str(i)] = training_errors
        df_train['mean_error_'+str(i)] = [np.mean(row) for row in training_errors]
        df_test['error_'+str(i)] = [list(row) for row in error_test]
        df_test['mean_error_'+str(i)] = [np.mean(row) for row in error_test]

        if num_training_points < total_training_points:
            append_num = append_sizes[i]
            print('Adding worst '+str(append_num)+' performing points to the training data.')
            # Take the rows which have NaN in the first_training_iteration column, find the largest one, get the indexes of these
            # For those rows assign the value of 'first_training iteration to be i+1
            df_train.loc[df_train[df_train['first_training_iteration'].isna()].nlargest(append_num, 'mean_error_'+str(i)).index, 'first_training_iteration'] = i + 1
            print('Added indexes.')

    print('Saving results of this iteration to csv, iterative_training.csv and iterative_test.csv')
    df_train.to_csv('iterative_training.csv')
    df_test.to_csv('iterative_test.csv')

    print('Done.')
    return df_train, df_test