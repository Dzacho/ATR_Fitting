#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from tmm.tmm_core import coh_tmm
import numpy as np
from numpy import pi, inf, array
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import multiprocessing as mp

# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi/180

# wavelength of your ATR laser in nm
lam_vac = 632.8

output = False
form = 'pdf'
mode = 'eps'

# Normalization data loading using the sample 100 % intensity file:
norm_path = 'path to data representing 100 % intensity on the detector'
norm = np.loadtxt(norm_path, delimiter=' ', skiprows=1)
peak_start_index = 18
peak_end_index = 22
intensity = np.average(norm[peak_start_index:peak_end_index,1])

class Sample:
    def __init__(self, file_path, sample_name, n_prism, prism_apex, n_substrate, substrate_thickness, init_guess, enable = 1):
        self.file_path = file_path
        self.sample_name = sample_name
        self.n_prism = n_prism
        self.prism_apex = prism_apex * degree
        self.n_substrate = n_substrate
        self.substrate_thickness = substrate_thickness
        self.init_guess = init_guess
        self.enable = enable
        self.theta_array = None
        self.data = None
        self.fit_pars = None
        self.fit_covs = None
        self.rmse = None
        self.mae = None
        self.fit_curve = None
        self.SqErr = None
        self.fig = None
        self.ax = None

    def process(self):
        #real data loading
        data = np.loadtxt(self.file_path, delimiter=' ', skiprows=1)
        
        self.theta_array = data[:,0]*degree/1000
        self.data = data[:,1]
        self.data = self.data/np.max([self.data.max(), intensity]) # normalization of the R data

        #correction from outer Theta(from data) to inner (real incident angle)
        self.theta_array = (self.prism_apex/2) + np.arcsin(np.sin(self.theta_array - self.prism_apex/2)/self.n_prism) # recalculation to inner angle
        
        #fitting
        if mode == 'eps':
            self.fit_pars, self.fit_covs = curve_fit(self.EPSPR, self.theta_array, self.data, self.init_guess) # in the init_guess array: list of estimated thicknesses, n, k for each layer and surface roughness
            '''print(self.fit_pars)
            print(np.sqrt(np.diag(self.fit_covs)))
            print(self.fit_covs)'''
            self.fit_curve = self.EPSPR(self.theta_array, *self.fit_pars)
        else:
            self.fit_pars, self.fit_covs = curve_fit(self.SPR, self.theta_array, self.data, self.init_guess) # in the init_guess array: list of estimated thicknesses, n, k for each layer and surface roughness
            '''print(self.fit_pars)
            print(np.sqrt(np.diag(self.fit_covs)))
            print(self.fit_covs)'''
            self.fit_curve = self.SPR(self.theta_array, *self.fit_pars)

        self.SqErr = ((self.fit_curve - self.data) ** 2)
        self.rmse = np.sqrt(np.mean(self.SqErr))
        self.mae = np.mean(np.abs(self.fit_curve - self.data))
        
        norm_scale = np.max([self.fit_pars[4], np.max(self.data), np.max(self.fit_curve)])
        self.fit_curve = self.fit_curve/norm_scale
        self.data = self.data/norm_scale
        
    def SPR(self, theta_list, d1, n1r, n1i, d2, n2r, n2i, scale):
        # EMA - Bruggeman's model
        eps_1_eff = (n1r + 1j*n1i)**2
        H_b = 0.5 * (eps_1_eff + 1)
        eps_r_eff = (H_b - np.sqrt(H_b**2 + 8 * eps_1_eff))/4
        n_eff = np.sqrt(eps_r_eff)
        
        d = array([inf, self.substrate_thickness, d1, d_r, inf])
        n = array([self.n_prism, self.n_substrate, n1r + 1j*n1i, n_eff, 1])
        result = np.empty(len(theta_list))
        for i in range(theta_list.size):
            result[i] = scale * coh_tmm('p', n, d, theta_list[i], lam_vac)['R']
        return result

    def EPSPR(self, theta_list, d1, eps1r, eps1i, d2, eps2r, eps2i, scale):
        # EMA - Bruggeman's model
        eps_1_eff = eps1r + 1j*eps1i
        H_b = 0.5 * (eps_1_eff + 1)
        eps_r_eff = (H_b - np.sqrt(H_b**2 + 8 * eps_1_eff))/4
        
        n = np.sqrt(eps_1_eff)
        n_eff = np.sqrt(eps_r_eff)
        
        d = array([inf, self.substrate_thickness, d1, d_r, inf])
        n = array([self.n_prism, self.n_substrate, n, n_eff, 1])
        result = np.empty(len(theta_list))
        for i in range(theta_list.size):
            result[i] = scale * coh_tmm('p', n, d, theta_list[i], lam_vac)['R']
        return result
        
    def plot(self, plot_Err = True):
        self.fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(self.theta_array/degree, self.data, 'blue', label = 'data')
        ax.plot(self.theta_array/degree, self.fit_curve, 'red', label = 'fit')
        if plot_Err:
            ax.plot(self.theta_array/degree, self.SqErr * 1000, 'green', label = r'Err$^2×1000$')
        ax.set_xlabel(r'$\Theta$ [˚]')
        ax.set_ylabel('Odrazivost')
        ax.legend(loc='lower right')
        ax.set_xlim(self.theta_array.min()/degree-1, self.theta_array.max()/degree+1)
        ax.set_ylim(0.0, 1.1)
        ax.set_title(self.sample_name)
    
# Estimated thickness [nm] and refractive indices of the layer
d = 50
d_r = 3 # Roughness height (for EMA)
n = 0.056253+4.276j
scale = 1
if mode == 'eps':
    eps = n**2
    general_init_guess = [d, eps.real, eps.imag, d_r, scale]
else:
    general_init_guess = [d, n.real, n.imag, d_r, scale]

# Write here the path where you want the fit parameters to be stored.
fitpars_path = '.../fitpars_batch.txt'

# Description
header = "Value +- std deviation"
if mode == 'eps':
    variables = 'd, \pm, e1, \pm, e2, \pm, d_r, \pm, scale, \pm, rmse, mae'
else:
    variables = 'd, \pm, n, \pm, k, \pm, d_r, \pm, scale, \pm, rmse, mae'
    
# Used glass n + jk at 632.8 nm:
BK7 = 1.5151
fused_silica = 1.457
FZU_Glass = 1.509780 + 3e-6j
KIPL_A = 1.511980 - 1e-6j
KIPL_B = 1.499575 + 2e-6j
#KIPL_C = 1.542548 + 0.001275j

#Samples list:
samples = [
    Sample('path to measurement data #1', 'sample name', BK7, 90, fused_silica, 0.5e6, [65.26, 0.11984, 4.31777, 3.11, 1.91787, 3.64897, scale], True),
    Sample('path to measurement data #2', 'sample name', BK7, 90, fused_silica, 0.5e6, [65.26, 0.11984, 4.31777, 3.11, 1.91787, 3.64897, scale], True)
    ]
Results = 'name, ' + str(variables) + '\n'

#multiprocessing of samples:
def process_sample(sample):
    if sample.enable:
        sample.process()
    else:
        print(f'Skipping {sample.sample_name}...')
    return sample

if __name__ == '__main__':
    pool = mp.Pool(processes = 8) #change the number of processes you want to let the script use
    samples = pool.map(process_sample, samples)
    pool.close()
    pool.join()

#If you don't want to use multiprocessing, use just a iteration loop over the list of samples and the command 'sample.process()'

for Sample in samples:
    if Sample.enable:
        Sample.plot(plot_Err=not(output))
        if output:
            Sample.fig.savefig('path to where you want to store your plots/%s_batch.%s' %(Sample.sample_name, form), format=form)
        else:
            plt.figure(Sample.fig.number)
            plt.show()
        Results += Sample.sample_name
        for i in range(len(Sample.fit_pars)):
            Results += ', {}, {}'.format(Sample.fit_pars[i], np.sqrt(np.diag(Sample.fit_covs)[i]))
        Results += ', {}, {}\n'.format(Sample.rmse, Sample.mae)
    else:
        print(f'Skipping {Sample.sample_name}...')

if output:
    with open(fitpars_path, "w") as f:
        f.write(Results)    
else:
    print(Results)
