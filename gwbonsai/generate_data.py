import numpy as np
from tqdm import tqdm
import pandas as pd
import surfinBH
import time
fit_name = 'NRSur7dq4Remnant'
fit = surfinBH.LoadFits(fit_name)

def remnant_properties_sampling(num_systematic_samples=1000,num_random_samples=1000,method='mixture',unique=False,debug=False,timing=False,path_to_csv=None,chi1_min=0.,chi1_max=0.8,chi2_min=0.,chi2_max=0.8,q_min=1.,q_max=4.,theta1_min=0., theta1_max = np.pi, theta2_min=0., theta2_max = np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    """
    Function to produce remnant mass data sets for training, testing and validation.
    Sampling methods can be systematic, random, or a mixture of the two.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)
        method: str, the sampling method, do we want systematic uniform sampling across the whole parameter space, or random sampling? Default is 'mixture',
                can be 'mixture', 'systematic' or 'random'.

    Returns:
        samples: a pandas dataframe containing the mass ratio, spins (in both Cartesian and spherical coordinates), final masses and mass errors.
    """

    if method=='systematic':
        # Sample systematically in q, spin magnitudes, tilts and azimuths
        sampled_params = param_sampling_systematic(num_systematic_samples,chi1_min=chi1_min,chi1_max=chi1_max,chi2_min=chi2_min,chi2_max=chi2_max,q_min=q_min,q_max=q_max,theta1_min=theta1_min,theta1_max=theta1_max,theta2_min=theta2_min,theta2_max=theta2_max,phi1_min=phi1_min,phi1_max=phi1_max,phi2_min=phi2_min,phi2_max=phi2_max)

    elif method == 'random':
        # Sample randomly in q, spin magnitudes, tilts and azimuths
        sampled_params = param_sampling_random(num_random_samples,chi1_min=chi1_min,chi1_max=chi1_max,chi2_min=chi2_min,chi2_max=chi2_max,q_min=q_min,q_max=q_max,theta1_min=theta1_min,theta1_max=theta1_max,theta2_min=theta2_min,theta2_max=theta2_max,phi1_min=phi1_min,phi1_max=phi1_max,phi2_min=phi2_min,phi2_max=phi2_max)
        
    elif method == 'mixture':
        # Sample systematically in q, spin magnitudes, tilts and azimuths
        sampled_params_sys = param_sampling_systematic(num_systematic_samples,chi1_min=chi1_min,chi1_max=chi1_max,chi2_min=chi2_min,chi2_max=chi2_max,q_min=q_min,q_max=q_max,theta1_min=theta1_min,theta1_max=theta1_max,theta2_min=theta2_min,theta2_max=theta2_max,phi1_min=phi1_min,phi1_max=phi1_max,phi2_min=phi2_min,phi2_max=phi2_max)
        # Sample randomly in q, spin magnitudes, tilts and azimuths
        sampled_params_rand = param_sampling_random(num_random_samples,chi1_min=chi1_min,chi1_max=chi1_max,chi2_min=chi2_min,chi2_max=chi2_max,q_min=q_min,q_max=q_max,theta1_min=theta1_min,theta1_max=theta1_max,theta2_min=theta2_min,theta2_max=theta2_max,phi1_min=phi1_min,phi1_max=phi1_max,phi2_min=phi2_min,phi2_max=phi2_max)
        # Combine the two sets of samples
        sampled_params = np.concatenate((sampled_params_sys,sampled_params_rand),axis=0)

    elif method == 'sparse_grids':
        # Sample randomly in q, spin magnitudes, tilts and azimuths
        sampled_params = param_sampling_sparse_grid(chi1_min=chi1_min,chi1_max=chi1_max,chi2_min=chi2_min,chi2_max=chi2_max,q_min=q_min,q_max=q_max,theta1_min=theta1_min,theta1_max=theta1_max,theta2_min=theta2_min,theta2_max=theta2_max,phi1_min=phi1_min,phi1_max=phi1_max,phi2_min=phi2_min,phi2_max=phi2_max)

    elif method == 'csv_params':
        sampled_params_df = pd.read_csv(path_to_csv)
        sampled_params = sampled_params_df[['q','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','chi1_mag','chi2_mag',\
                                              'chi1_tilt','chi2_tilt','chi1_phi','chi2_phi']].to_numpy()
        
    if unique==True:
        sampled_params_df = pd.DataFrame(data=sampled_params, columns=['q','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','chi1_mag','chi2_mag',\
                                              'chi1_tilt','chi2_tilt','chi1_phi','chi2_phi'])
        sampled_params_unique = check_unique_samples(sampled_params_df)
        sampled_params_unique = sampled_params_unique[sampled_params_unique['unique']==True].drop(columns=['unique']).to_numpy()
        print(np.shape(sampled_params))
        
    num_evals = np.shape(sampled_params)[0]
    mf_grid = np.zeros((num_evals,))
    mf_err_grid = np.zeros((num_evals,))
    chif_grid = np.zeros((num_evals,3))
    chif_err_grid = np.zeros((num_evals,3))
    vf_grid = np.zeros((num_evals,3))
    vf_err_grid = np.zeros((num_evals,3))
    if timing==True:
        eval_time = np.zeros(num_evals,)

    for i in tqdm(range(num_evals)):
        try:
            q = sampled_params[i,0]
            chi1 = [sampled_params[i,1],sampled_params[i,2],sampled_params[i,3]]
            chi2 = [sampled_params[i,4],sampled_params[i,5],sampled_params[i,6]]
            #print(sampled_params[i,:])
            if debug == True:
                print('Attempting to generate remnant properties with params:')
                print(q, chi1, chi2)
            if timing==True:
                start = time.time()
            #mf_grid[i], mf_err_grid[i] = fit.mf(q, chi1, chi2);
            mf_grid[i], chif_grid[i,:], vf_grid[i,:], mf_err_grid[i], chif_err_grid[i,:], vf_err_grid[i,:] = fit.all(q, chi1, chi2)
            if timing==True:
                end = time.time()
                eval_time[i] = end - start
        except:
            print('Failed to evaluate remnant properties for binary number '+str(i)+'.')
            print('Params:')
            print(sampled_params[i,:])
            mf_grid[i] = np.nan
            mf_err_grid[i] = np.nan
            chif_grid[i,:] = [np.nan, np.nan, np.nan]
            chif_err_grid[i,:] = [np.nan, np.nan, np.nan]
            vf_grid[i,:] = [np.nan, np.nan, np.nan]
            vf_err_grid[i,:] = [np.nan, np.nan, np.nan]
            
    if timing==True:
        data = np.column_stack((sampled_params,mf_grid,mf_err_grid,chif_grid,chif_err_grid,vf_grid,vf_err_grid,eval_time))
        df = pd.DataFrame(data=data, columns=['q','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','chi1_mag','chi2_mag',\
                                              'chi1_tilt','chi2_tilt','chi1_phi','chi2_phi','mf','mf_err','chif_1','chif_2','chif_3','chif_err_1','chif_err_2','chif_err_3','vf_1','vf_2','vf_3','vf_err_1','vf_err_2','vf_err_3','eval_times'])       
    else:
        data = np.column_stack((sampled_params,mf_grid,mf_err_grid,chif_grid,chif_err_grid,vf_grid,vf_err_grid))
        df = pd.DataFrame(data=data, columns=['q','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','chi1_mag','chi2_mag',\
                                              'chi1_tilt','chi2_tilt','chi1_phi','chi2_phi','mf','mf_err','chif_1','chif_2','chif_3','chif_err_1','chif_err_2','chif_err_3','vf_1','vf_2','vf_3','vf_err_1','vf_err_2','vf_err_3'])

    return df

def param_sampling_systematic(num_systematic_samples,chi1_min=0.,chi1_max=0.8,chi2_min=0.,chi2_max=0.8,q_min=1.,q_max=4.,theta1_min=0., theta1_max = np.pi, theta2_min=0., theta2_max = np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    """
    Function to produce systematically sampled binary parameter data sets for training, testing and validation.

    Inputs:
        num samples: int, the number of samples we want in the dataset (note that for systematic sampling, this is the number of samples in each dimension)

    Returns:
        samples: a numpy array containing the mass ratio, and spins (in both Cartesian and spherical coordinates).
    """

    # First sample systematically in q, spin magnitudes, tilts and azimuths
    q_grid_sys = np.linspace(q_min, q_max, num=num_systematic_samples)
    chi1_mag_grid_sys = np.linspace(chi1_min, chi1_max, num=num_systematic_samples)
    chi2_mag_grid_sys = np.linspace(chi2_min, chi2_max, num=num_systematic_samples)
    chi1_tilt_grid_sys = np.linspace(theta1_min, theta1_max, num=num_systematic_samples)
    chi2_tilt_grid_sys = np.linspace(theta2_min, theta2_max, num=num_systematic_samples)
    chi1_phi_grid_sys = np.linspace(phi1_min, phi1_max, num=num_systematic_samples)
    chi2_phi_grid_sys = np.linspace(phi2_min, phi2_max, num=num_systematic_samples)

    q_meshgrid_sys, chi1_mag_meshgrid_sys, chi1_tilt_meshgrid_sys, chi1_phi_meshgrid_sys,\
    chi2_mag_meshgrid_sys, chi2_tilt_meshgrid_sys, chi2_phi_meshgrid_sys = np.meshgrid(q_grid_sys,\
                                                                chi1_mag_grid_sys, chi1_tilt_grid_sys,\
                                                                chi1_phi_grid_sys, chi2_mag_grid_sys,\
                                                                chi2_tilt_grid_sys, chi2_phi_grid_sys)

    # Now convert the spin samples to Cartesian coordinates (for input into NRSur7dq4Remnant)
    chi1x_grid_sys = chi1_mag_grid_sys * np.sin(chi1_tilt_grid_sys) * np.cos(chi1_phi_grid_sys)
    chi1y_grid_sys = chi1_mag_grid_sys * np.sin(chi1_tilt_grid_sys) * np.sin(chi1_phi_grid_sys)
    chi1z_grid_sys = chi1_mag_grid_sys * np.cos(chi1_tilt_grid_sys)
    chi2x_grid_sys = chi2_mag_grid_sys * np.sin(chi2_tilt_grid_sys) * np.cos(chi2_phi_grid_sys)
    chi2y_grid_sys = chi2_mag_grid_sys * np.sin(chi2_tilt_grid_sys) * np.sin(chi2_phi_grid_sys)
    chi2z_grid_sys = chi2_mag_grid_sys * np.cos(chi2_tilt_grid_sys)

    q_meshgrid_sys, chi1x_meshgrid_sys, chi1y_meshgrid_sys, chi1z_meshgrid_sys,\
    chi2x_meshgrid_sys, chi2y_meshgrid_sys, chi2z_meshgrid_sys = np.meshgrid(q_grid_sys,\
                                                                chi1x_grid_sys, chi1y_grid_sys,\
                                                                chi1z_grid_sys, chi2x_grid_sys,\
                                                                chi2y_grid_sys, chi2z_grid_sys)

    # Flatten our samples into a 2D array, where each row has the mass ratio and Cartesian spin components for one sampled binary
    sampled_params_sys = np.column_stack((q_meshgrid_sys.flatten(),chi1x_meshgrid_sys.flatten(),chi1y_meshgrid_sys.flatten(),\
                                        chi1z_meshgrid_sys.flatten(),chi2x_meshgrid_sys.flatten(),chi2y_meshgrid_sys.flatten(),\
                                        chi2x_meshgrid_sys.flatten(),chi1_mag_meshgrid_sys.flatten(),chi2_mag_meshgrid_sys.flatten(),\
                                        chi1_tilt_meshgrid_sys.flatten(),chi2_tilt_meshgrid_sys.flatten(),chi1_phi_meshgrid_sys.flatten(),\
                                        chi2_phi_meshgrid_sys.flatten()))
    
    return sampled_params_sys

def param_sampling_random(num_random_samples,chi1_min=0.,chi1_max=0.8,chi2_min=0.,chi2_max=0.8,q_min=1.,q_max=4.,theta1_min=0., theta1_max = np.pi, theta2_min=0., theta2_max = np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    """
    Function to produce randomly sampled binary parameter data sets for training, testing and validation.

    Inputs:
        num samples: int, the number of samples we want in the dataset

    Returns:
        samples: a numpy array containing the mass ratio, and spins (in both Cartesian and spherical coordinates).
    """
    
    # First sample randomly in q, spin magnitudes, tilts and azimuths
    q_grid_rand = np.random.uniform(low=q_min, high=q_max, size=num_random_samples)
    chi1_mag_grid_rand = np.random.uniform(low=chi1_min, high=chi1_max, size=num_random_samples)
    chi2_mag_grid_rand = np.random.uniform(low=chi2_min, high=chi2_max, size=num_random_samples)
    chi1_tilt_grid_rand = np.random.uniform(low=theta1_min, high=theta1_max, size=num_random_samples)
    chi2_tilt_grid_rand = np.random.uniform(low=theta2_min, high=theta2_max, size=num_random_samples)
    chi1_phi_grid_rand = np.random.uniform(low=phi1_min, high=phi1_max, size=num_random_samples)
    chi2_phi_grid_rand = np.random.uniform(low=phi2_min, high=phi2_max, size=num_random_samples)

    # Now convert the spin samples to Cartesian coordinates (for input into NRSur7dq4Remnant)
    chi1x_grid_rand = chi1_mag_grid_rand * np.sin(chi1_tilt_grid_rand) * np.cos(chi1_phi_grid_rand)
    chi1y_grid_rand = chi1_mag_grid_rand * np.sin(chi1_tilt_grid_rand) * np.sin(chi1_phi_grid_rand)
    chi1z_grid_rand = chi1_mag_grid_rand * np.cos(chi1_tilt_grid_rand)
    chi2x_grid_rand = chi2_mag_grid_rand * np.sin(chi2_tilt_grid_rand) * np.cos(chi2_phi_grid_rand)
    chi2y_grid_rand = chi2_mag_grid_rand * np.sin(chi2_tilt_grid_rand) * np.sin(chi2_phi_grid_rand)
    chi2z_grid_rand = chi2_mag_grid_rand * np.cos(chi2_tilt_grid_rand)

    # Flatten our samples into a 2D array, where each row has the mass ratio and Cartesian spin components for one sampled binary
    sampled_params_rand = np.column_stack((q_grid_rand.flatten(),chi1x_grid_rand.flatten(),chi1y_grid_rand.flatten(),\
                                        chi1z_grid_rand.flatten(),chi2x_grid_rand.flatten(),chi2y_grid_rand.flatten(),\
                                        chi2z_grid_rand.flatten(),chi1_mag_grid_rand.flatten(),chi2_mag_grid_rand.flatten(),\
                                        chi1_tilt_grid_rand.flatten(),chi2_tilt_grid_rand.flatten(),chi1_phi_grid_rand.flatten(),\
                                        chi2_phi_grid_rand.flatten()))
    
    return sampled_params_rand

########### Helper functions for sparse grid sampling (see Appendix A in Blackamn:2017pcm) ##############
def f_q(n):
    return int(1 + 2**n)

def f_chi(n):
    return f_q(n)

def f_theta(n):
    return int(1 + 2**(n+1))

def f_phi(n):
    return int(1 + 3 * 2**n)

def get_uniform_grid(a, b, N):
    grid = np.linspace(a, b, N)
    return grid

def get_dense_grid(n, q_min=1., q_max=2., chi1_min=0., chi1_max=0.8, chi2_min=0., chi2_max=0.8, theta1_min=0., theta1_max = np.pi, theta2_min=0., theta2_max = np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    q_grid = get_uniform_grid(q_min, q_max, f_q(n))
    chi1_grid = get_uniform_grid(chi1_min, chi1_max, f_chi(n))
    chi2_grid = get_uniform_grid(chi2_min, chi2_max, f_chi(n))
    theta1_grid = get_uniform_grid(theta1_min, theta1_max, f_theta(n))
    phi1_grid = get_uniform_grid(phi1_min, phi1_max, f_phi(n))
    theta2_grid = get_uniform_grid(theta2_min, theta2_max, f_theta(n))
    phi2_grid = get_uniform_grid(phi2_min, phi2_max, f_phi(n))

    q_meshgrid, chi1_meshgrid, theta1_meshgrid, phi1_meshgrid,\
    chi2_meshgrid, theta2_meshgrid, phi2_meshgrid = np.meshgrid(q_grid,\
                                                                chi1_grid, theta1_grid,\
                                                                phi1_grid, chi2_grid,\
                                                                theta2_grid, phi2_grid)
    chi1x_grid = chi1_meshgrid * np.sin(theta1_meshgrid) * np.cos(phi1_meshgrid)
    chi1y_grid = chi1_meshgrid * np.sin(theta1_meshgrid) * np.sin(phi1_meshgrid)
    chi1z_grid = chi1_meshgrid * np.cos(theta1_meshgrid)
    chi2x_grid = chi2_meshgrid * np.sin(theta2_meshgrid) * np.cos(phi2_meshgrid)
    chi2y_grid = chi2_meshgrid * np.sin(theta2_meshgrid) * np.sin(phi2_meshgrid)
    chi2z_grid = chi2_meshgrid * np.cos(theta2_meshgrid)

    sampled_params = np.column_stack((q_meshgrid.flatten(),chi1x_meshgrid.flatten(),chi1y_meshgrid.flatten(),\
                                        chi1z_meshgrid.flatten(),chi2x_meshgrid.flatten(),chi2y_meshgrid.flatten(),\
                                        chi2x_meshgrid.flatten(),chi1_meshgrid.flatten(),chi2_meshgrid.flatten(),\
                                        theta1_meshgrid.flatten(),theta2_meshgrid.flatten(),phi1_meshgrid.flatten(),\
                                        phi2_meshgrid.flatten()))
    
    return sampled_params

def get_nonuniform_dense_grid(n_q, n_chi1, n_chi2, n_theta1, n_theta2, n_phi1, n_phi2, q_min=1., q_max=2., chi1_min=0., chi1_max=0.8, chi2_min=0., chi2_max=0.8, theta1_min=0., theta1_max=np.pi, theta2_min=0., theta2_max=np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    q_grid = get_uniform_grid(q_min, q_max, f_q(n_q))
    chi1_grid = get_uniform_grid(chi1_min, chi1_max, f_chi(n_chi1))
    chi2_grid = get_uniform_grid(chi2_min, chi2_max, f_chi(n_chi2))
    theta1_grid = get_uniform_grid(theta1_min, theta1_max, f_theta(n_theta1))
    phi1_grid = get_uniform_grid(phi1_min, phi1_max, f_phi(n_phi1))
    theta2_grid = get_uniform_grid(theta2_min, theta2_max, f_theta(n_theta2))
    phi2_grid = get_uniform_grid(phi2_min, phi2_max, f_phi(n_phi2))

    q_meshgrid, chi1_meshgrid, theta1_meshgrid, phi1_meshgrid,\
    chi2_meshgrid, theta2_meshgrid, phi2_meshgrid = np.meshgrid(q_grid,\
                                                                chi1_grid, theta1_grid,\
                                                                phi1_grid, chi2_grid,\
                                                                theta2_grid, phi2_grid)
    chi1x_grid = chi1_meshgrid * np.sin(theta1_meshgrid) * np.cos(phi1_meshgrid)
    chi1y_grid = chi1_meshgrid * np.sin(theta1_meshgrid) * np.sin(phi1_meshgrid)
    chi1z_grid = chi1_meshgrid * np.cos(theta1_meshgrid)
    chi2x_grid = chi2_meshgrid * np.sin(theta2_meshgrid) * np.cos(phi2_meshgrid)
    chi2y_grid = chi2_meshgrid * np.sin(theta2_meshgrid) * np.sin(phi2_meshgrid)
    chi2z_grid = chi2_meshgrid * np.cos(theta2_meshgrid)

    sampled_params = np.column_stack((q_meshgrid.flatten(),chi1x_grid.flatten(),chi1y_grid.flatten(),\
                                        chi1z_grid.flatten(),chi2x_grid.flatten(),chi2y_grid.flatten(),\
                                        chi2z_grid.flatten(),chi1_meshgrid.flatten(),chi2_meshgrid.flatten(),\
                                        theta1_meshgrid.flatten(),theta2_meshgrid.flatten(),phi1_meshgrid.flatten(),\
                                        phi2_meshgrid.flatten()))
    
    return sampled_params

def param_sampling_sparse_grid(q_min=1., q_max=2., chi1_min=0., chi1_max=0.8, chi2_min=0., chi2_max=0.8, theta1_min=0., theta1_max=np.pi, theta2_min=0., theta2_max=np.pi, phi1_min=0., phi1_max = 2*np.pi, phi2_min=0., phi2_max = 2*np.pi):
    print(q_max)
    all_samples = pd.DataFrame()
    for i in range(7):
        n_array = np.zeros((7),)
        n_array[i] = 1
        samples = get_nonuniform_dense_grid(n_array[0],n_array[1],n_array[2],n_array[3],n_array[4],n_array[5],n_array[6],q_min=q_min, q_max=q_max, chi1_min=chi1_min, chi1_max=chi1_max, chi2_min=chi2_min, chi2_max=chi2_max, theta1_min=theta1_min, theta1_max=theta1_max, theta2_min=theta2_min, theta2_max=theta2_max, phi1_min=phi1_min, phi1_max = phi1_max, phi2_min=phi2_min, phi2_max = phi2_max)
        samples_df = pd.DataFrame(data=samples,columns=['q','chi1x','chi1y','chi1z','chi2x','chi2y','chi2z','chi1_mag','chi2_mag',\
                                              'chi1_tilt','chi2_tilt','chi1_phi','chi2_phi'])
        all_samples = pd.concat([all_samples,samples_df])

    return all_samples

################## Helper function to check if binary samples are unique ####################
def check_unique_samples(samplesdf,tolerance=1e-8):
    unique_q = np.full((len(samplesdf)),True)
    indexes = samplesdf.index.values.tolist()
    samplesdf['unique']=np.full((len(samplesdf)),True)
    print('Checking sample uniqueness')
    for i in tqdm(range(len(samplesdf))):
        idx = indexes[i]
        q_lower_tol, q_upper_tol = samplesdf['q'].iloc[i] - tolerance, samplesdf['q'].iloc[i] + tolerance
        chi1x_lower_tol, chi1x_upper_tol = samplesdf['chi1x'].iloc[i] - tolerance, samplesdf['chi1x'].iloc[i] + tolerance
        chi1y_lower_tol, chi1y_upper_tol = samplesdf['chi1y'].iloc[i] - tolerance, samplesdf['chi1y'].iloc[i] + tolerance
        chi1z_lower_tol, chi1z_upper_tol = samplesdf['chi1z'].iloc[i] - tolerance, samplesdf['chi1z'].iloc[i] + tolerance
        chi2x_lower_tol, chi2x_upper_tol = samplesdf['chi2x'].iloc[i] - tolerance, samplesdf['chi2x'].iloc[i] + tolerance
        chi2y_lower_tol, chi2y_upper_tol = samplesdf['chi2y'].iloc[i] - tolerance, samplesdf['chi2y'].iloc[i] + tolerance
        chi2z_lower_tol, chi2z_upper_tol = samplesdf['chi2z'].iloc[i] - tolerance, samplesdf['chi2z'].iloc[i] + tolerance
        for j in range(len(samplesdf)):
            idx_j = indexes[j]
            if j <= i:
                pass
            else:
                if ((q_lower_tol < samplesdf['q'].iloc[j]) & (samplesdf['q'].iloc[j] < q_upper_tol)):
                    if ((chi1x_lower_tol < samplesdf['chi1x'].iloc[j]) & (samplesdf['chi1x'].iloc[j] < chi1x_upper_tol)):
                        if ((chi1y_lower_tol < samplesdf['chi1y'].iloc[j]) & (samplesdf['chi1y'].iloc[j] < chi1y_upper_tol)):
                            if ((chi1z_lower_tol < samplesdf['chi1z'].iloc[j]) & (samplesdf['chi1z'].iloc[j] < chi1z_upper_tol)):
                                if ((chi2x_lower_tol < samplesdf['chi2x'].iloc[j]) & (samplesdf['chi2x'].iloc[j] < chi2x_upper_tol)):
                                    if ((chi2y_lower_tol < samplesdf['chi2y'].iloc[j]) & (samplesdf['chi2y'].iloc[j] < chi2y_upper_tol)):
                                        if ((chi2z_lower_tol < samplesdf['chi2z'].iloc[j]) & (samplesdf['chi2z'].iloc[j] < chi2z_upper_tol)):
                                            samplesdf['unique'].iloc[j] = False
    return samplesdf