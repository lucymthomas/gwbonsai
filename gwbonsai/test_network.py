import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import tensorflow.keras as tfkeras
import time
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
import logging
logging.basicConfig(filename='build_train_test_log.log', encoding='utf-8', level=logging.INFO)
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def test_nn(file_prefix, testing_params, testing_mfs, scaling=False, timing=True, standard_scale=False):
    """
    Function to define and compile a sequential keras ANN with user-specified hyperparameters.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)
        method: str, the sampling method, do we want systematic uniform sampling across the whole parameter space, or random sampling? Default is 'mixture',
                can be 'mixture', 'systematic' or 'random'.

    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """
    logging.info('Testing the speed and accuracy of the neural network.')
    if standard_scale==True:
        logging.info('Scaling the inputs before evaluating the model.')
        params_scaler=load(file_prefix+'input_params_scaler.bin')
        mfs_scaler=load(file_prefix+'output_params_scaler.bin')
        
    logging.info('Loading trained model.')
    num_evals = np.shape(testing_mfs)[0]
    model = tfkeras.models.load_model(file_prefix+'model.keras')
    logging.info('Evaluating model.')
    if timing==True:
        start = time.time()
    if standard_scale==True:
        testing_params = params_scaler.transform(testing_params)
    mfs_predicted = model.predict(testing_params)
    if standard_scale==True:
        mfs_predicted = mfs_scaler.inverse_transform(mfs_predicted)
    if timing==True:
        end = time.time()
    logging.info('Model evaluated.')
    if timing==True:
        logging.info('Took '+str(end - start)+' seconds for '+str(num_evals)+' evaluations.')
        logging.info('This is '+str((end - start)/num_evals)+' seconds per binary.')
    logging.info('Calculating errors.')
    mf_error = mfs_predicted - testing_mfs
            
    if timing==True:
        return mfs_predicted, mf_error, (end - start)/num_evals
    else:
        return mfs_predicted, mf_error

def plot_nn_test(file_prefix, df, label, plot_masses=True, plot_spins=False, plot_velocities=False):
    """
    Function to define and compile a sequential keras ANN with user-specified hyperparameters.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)
        method: str, the sampling method, do we want systematic uniform sampling across the whole parameter space, or random sampling? Default is 'mixture',
                can be 'mixture', 'systematic' or 'random'.

    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """
    if plot_masses==True:
        logging.info('Plotting the mf ANN errors with a linear x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['mf_ann_error'],label='NN error',bins=500)
        ax.set_xlabel('$\Delta Mf$');
        #ax.set_xlim(-0.001,0.0014)
        ax.legend();
        fig.savefig(file_prefix+'mf_ANN_errors_linear_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'mf_ANN_errors_linear_x_'+label+'.png')
        
        logging.info('Plotting the absolute mf ANN errors with a linear x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['mf_ann_error_abs'],label='absolute NN error',bins=500)
        ax.set_xlabel('$\Delta Mf$');
        #ax.set_xlim(-0.001,0.0014)
        ax.legend();
        fig.savefig(file_prefix+'mf_ANN_errors_abs_linear_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'mf_ANN_errors_abs_linear_x_'+label+'.png')
        
        logging.info('Plotting the absolute mf ANN errors with a log x axis and the GPR uncertainty.')
        fig, ax = plt.subplots()
        ax.hist(df['mf_ann_error_abs'],label='absolute NN error',bins=np.logspace(np.log10(df['mf_ann_error_abs'].min()),np.log10(df['mf_ann_error_abs'].max()), 50))
        ax.hist(df['mf_err'],label='GPR uncertainty',bins=np.logspace(np.log10(df['mf_err'].min()),np.log10(df['mf_err'].max()), 50))
        ax.set_xlabel('$|\Delta Mf|$');
        ax.set_ylabel('Frequency');
        #ax.set_xlim(0.,0.022)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend();
        fig.savefig(file_prefix+'mf_ANN_errors_abs_GPR_uncertainty_log_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'mf_ANN_errors_abs_GPR_uncertainty_log_x_'+label+'.png')

    if plot_spins==True:
        
        logging.info('Plotting the absolute chif magnitude ANN errors with a log x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['chif_mag_error_abs'],label='absolute NN error',bins=np.logspace(np.log10(df['chif_mag_error_abs'].min()),np.log10(df['chif_mag_error_abs'].max()), 50))
        ax.set_xlabel('$|\Delta \chi_f|$');
        ax.set_ylabel('Frequency');
        #ax.set_xlim(0.,0.022)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend();
        fig.savefig(file_prefix+'chif_mag_ANN_errors_abs_log_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'chif_mag_ANN_errors_abs_log_x_'+label+'.png')

        logging.info('Plotting the chif angle ANN errors with a log x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['chif_angle'],label='angle',bins=np.logspace(np.log10(df['chif_angle'].min()),np.log10(df['chif_angle'].max()), 50))
        ax.set_xlabel('$\cos^{-1}(\chi_f \dot \chi_f*)$');
        ax.set_ylabel('Frequency');
        #ax.set_xlim(0.,0.022)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend();
        fig.savefig(file_prefix+'chif_angle_ANN_errors_log_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'chif_angle_ANN_errors_log_x_'+label+'.png')

    if plot_velocities==True:
        logging.info('Plotting the absolute vf magnitude ANN errors with a log x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['vf_mag_error_abs'],label='absolute NN error',bins=np.logspace(np.log10(df['vf_mag_error_abs'].min()),np.log10(df['vf_mag_error_abs'].max()), 50))
        ax.set_xlabel('$|\Delta v_f|$');
        ax.set_ylabel('Frequency');
        #ax.set_xlim(0.,0.022)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend();
        fig.savefig(file_prefix+'vf_mag_ANN_errors_abs_log_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'vf_mag_ANN_errors_abs_log_x_'+label+'.png')

        logging.info('Plotting the vf angle ANN errors with a log x axis.')
        fig, ax = plt.subplots()
        ax.hist(df['vf_angle'],label='angle',bins=np.logspace(np.log10(df['vf_angle'].min()),np.log10(df['vf_angle'].max()), 50))
        ax.set_xlabel('$\cos^{-1}(v_f \dot v_f*)$');
        ax.set_ylabel('Frequency');
        #ax.set_xlim(0.,0.022)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend();
        fig.savefig(file_prefix+'vf_angle_ANN_errors_log_x_'+label+'.png')   
        plt.close(fig)
        logging.info('Saved to '+file_prefix+'vf_angle_ANN_errors_log_x_'+label+'.png')
    
    return
    
    