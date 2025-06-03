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

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import MeanSquaredError
from contextlib import redirect_stdout


def compile_mlp(file_prefix,input_shape=1,output_shape=1,num_hidden_layers=4,nodes_per_layer=10,dropout=False,activation='relu', lr=1e-3, verbose=False):

    """
    Function to compile a sequential keras MLP ANN with user-specified hyperparameters.

    Inputs:
        file_prefix: path to folder in which to save the compiled model.
        output_shape: int, the number of outputs of the model, default is 1.
        num_hidden_layers: int, the number of hidden layers in the model, default is 4.
        nodes_per_layer: int, the number of nodes per hidden layer, default is 10.
        activation: str, the activation function to use in the hidden layers, default is 'relu'.
        lr: float, the learning rate for the optimizer, default is 1e-3.


    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """
    if verbose:
        print('Initiating neural network with parameters:')
        print(str(num_hidden_layers)+' hidden layers,')
        print(str(nodes_per_layer)+' nodes per layer,')
        print(str(activation)+' activation function in hidden layers,')
        print(str(lr)+' learning rate.')
    
    model = Sequential()
    model.add(InputLayer(shape=(input_shape,)))

    for layer in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation=activation))
        #model.add(layers.Dropout(0.5))
    
    model.add(Dense(output_shape, activation='linear'))

    if verbose:
        print('Compiling the model.')
    model.compile(
    # Optimization algorithm, specify learning rate
        optimizer=Adam(learning_rate=lr),
    # Loss function for a binary classifier
        loss=MeanSquaredError(),
    # Diagnostic quantities
        metrics=[MeanSquaredError()]
    )
    if verbose:
        print('Saving summary of compiled model to model_summary.txt.')
    with open(file_prefix+'model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    if verbose:
        print('Done, model compile successfully.')
        
    return model