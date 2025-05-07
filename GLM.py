#%% import
# from curses import raw
import os
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.interpolate 
import scipy.signal
from scipy import linalg
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
import sys
import copy
from numpy.fft import fft as fft
from numpy.fft import ifft as ifft
import scipy.stats
import copy
from scipy.special import rel_entr
import pickle
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon, chi2

import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as smm

import utility_functions as utils

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
#%% Define the linear model




#%% PP_GLM class
class PP_GLM():
    def __init__(self, 
                 dataset=None, 
                 select_trials=None, 
                 membership=None, 
                 condition_ids=None, 
                 nt=None, 
                 ntrial=None,
                 npadding=None, 
                 fps=1000):
        """Initialize PP_GLM 

        Args:
            dataset (Allendataset, optional): you can input a dataset for easier use. Defaults to None.
            select_trials (array of Boole, optional): if you use dataset as input, you can specify running trials to use. Defaults to None.
            nt (int, optional): number of time bins, if don't import dataset directly. Defaults to None.
            ntrial (int, optional): number of trials, if don't import dataset directly. Defaults to None.
        """
        if dataset is None:
            self.nt = nt
            self.ntrial = ntrial
            if select_trials is None:
                self.select_trials = np.full(self.ntrial, True)
            else:
                self.select_trials = select_trials
            if type(self.select_trials[0]) == np.bool_:
                self.ntrial = self.select_trials.sum()
            else:
                self.ntrial = self.select_trials.shape[0]
            self.npadding = npadding
            self.dataset = None
            self.time_line = np.arange(self.nt)*1e-3
            self.fps = fps
        else:
            self.dataset = dataset
            self.nt = self.dataset.nt
            self.fps = self.dataset.fps
            
            if select_trials is None:
                self.select_trials = np.full(dataset.ntrial, True)
            else:
                self.select_trials = select_trials
            if type(select_trials[0]) == np.bool_:
                self.ntrial = self.select_trials.sum()
            else:
                self.ntrial = self.select_trials.shape[0]
            self.membership = membership
            self.condition_ids = condition_ids
            self.npadding = self.dataset.npadding
            self.time_line = self.dataset.time_line
        self.effect_list = []
        self.basis_list = []
        self.basis_name = []
        self.effect_type_list = []
        self.raw_input_list = []
        self.kwargs_list = []
        self.target = None
        self.output = None
        self.no_penalty = []
        self.a = None
        self.intecept = None

    def add_effect(self, effect_type, raw_input=None, use_all=False, apply_no_penalty=False, **kwargs):
        assert effect_type in ['homogeneous_baseline', 
                        'inhomogeneous_baseline', 
                        'trialwise_inhomogeneous_baseline', 
                        'coupling', 
                        'twoway_coupling', 
                        'circular', 
                        'linear',
                        'history', 
                        'varying_linear',
                        'dense_coupling',
                        'refractory',
                        'refractory_additive',
                        'refractory_box',
                        'trial_coef', 
                        'condition_coef', 
                        'interaction'],  "Not supported effect_type!"

        # record for later use
        self.effect_type_list.append(effect_type)
        self.raw_input_list.append(raw_input)
        self.kwargs_list.append(kwargs)
            
        if effect_type == 'homogeneous_baseline':
            X_baseline = np.ones((self.nt*self.ntrial,1))
            self.effect_list.append(X_baseline)
            self.basis_list.append(X_baseline[0:self.nt,:])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'inhomogeneous_baseline':
            X_baseline = inhomo_baseline(ntrial=self.ntrial, 
                                        start=0,
                                        end=self.nt,
                                        dt=1, 
                                        **kwargs)
            self.effect_list.append(X_baseline)
            self.basis_list.append(X_baseline[0:self.nt,:])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'trialwise_inhomogeneous_baseline':
            template = inhomo_baseline(ntrial=1, 
                                        start=0,
                                        end=self.nt,
                                        dt=1, 
                                        **kwargs)
            nbasis = template.shape[1]
            X_trial_coef = np.zeros((self.nt*self.ntrial, self.ntrial*nbasis))
            for itrial in range(self.ntrial):
                X_trial_coef[(itrial*self.nt):((itrial+1)*self.nt), (itrial*nbasis):((itrial+1)*nbasis)] = template
            self.effect_list.append(X_trial_coef)
            self.basis_list.append(np.tile(template, (1, self.ntrial)))
            self.basis_name.append(effect_type)
            
        elif effect_type == 'coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            pillow_basis = make_pillow_basis(**kwargs)
            
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.effect_list.append(X_coupling)
            
            self.basis_list.append(pillow_basis)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'interaction':
            # Interaction term here is an interaction term of coupling effect and current spike count in a window
            # The term can make the coupling effect of two spikes smaller than twice the coupling effect of one spike. 
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            
            tau = kwargs.pop('tau', 100)
            order = kwargs.pop('order', 1)
            
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]
            
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            
            pillow_basis = make_pillow_basis(**kwargs)
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.basis_list.append(pillow_basis)
            
            self.effect_list.append(X_coupling*X_refractory**order)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
                
        elif effect_type == 'dense_coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            num = kwargs.pop('num',10)
            pillow_basis = np.diag(np.ones(num))
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.effect_list.append(X_coupling)
            
            self.basis_list.append(pillow_basis)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'refractory':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            tau = kwargs.pop('tau',10)
            self.tau = tau
            order = kwargs.pop('order', 2)
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]
            
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            X_refractory = X_refractory**order
            # Next line for “Refractory with cutoff”
            # X_refractory = np.minimum(X_refractory, 3)
            self.effect_list.append(X_refractory)
            self.basis_list.append(refractory_spikes.mean(axis=1)[:, np.newaxis])
            self.basis_name.append(effect_type)

        elif effect_type == 'refractory_additive':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            tau = kwargs.pop('tau',10)
            self.tau = tau
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]

            # refractory_spikes = utils.kernel_smoothing(refractory_spikes, 3, window=None)
            refractory_spikes = refractory_spikes[-self.nt:, :]
            refractory_spikes = refractory_spikes.flatten('F')
            refractory_spikes /= tau
            
            # (abandoned) whether to force monotonicity
            # ascend = kwargs.pop('ascend', True)
            
            # Function for refractory
            Lambda_ub = np.max(refractory_spikes)+0.02
            Lambda_lb = 0
            
            num = kwargs.pop('num', 4)
            add_constant_basis = False
            dt = 0.01
            spline_order = 2
            
            early_knots_quan = [0, 0.3, 0.5]
            starting_quan = 0.7
            end_quan = 0.92
            starting_knot = 3
            # The following four lines are for comparison in "More details on f_{rf}.py"
            # early_knots_quan = []
            # starting_quan = 0.0
            # end_quan = 1
            # starting_knot = 2
            early_knots = [np.quantile( refractory_spikes , quan) for quan in early_knots_quan]
            starting_val = np.quantile( refractory_spikes , starting_quan)
            end_val = np.quantile( refractory_spikes , end_quan)
            if end_quan!=1:
                mid_knots = np.linspace(starting_val, end_val, num-2)
                knots = np.hstack((early_knots, mid_knots, [Lambda_ub]))
            else:
                mid_knots = np.linspace(starting_val, end_val, num-1)
                knots = np.hstack((early_knots, mid_knots))
            
            # knots[0]= Lambda_lb
            knots[-1] = Lambda_ub
            knots = np.hstack((np.ones(spline_order) * Lambda_lb,
                    knots,
                    np.ones(spline_order) * Lambda_ub))
            # print(knots)
            f_refractory_basis_vec = make_b_spline_basis_arbitrary_knots(spline_order, knots, dt, add_constant_basis, False)
            f_refractory_basis_vec = f_refractory_basis_vec[:, starting_knot:]   # Get rid of the first three

            f_refractory_xx = np.arange(f_refractory_basis_vec.shape[0])*dt
            self.f_refractory_xx = f_refractory_xx
            
            self.n_f_refractory = f_refractory_basis_vec.shape[1]
            self.f_refractory_basis = []
            for i_basis in range(self.n_f_refractory):
                self.f_refractory_basis.append(scipy.interpolate.interp1d(f_refractory_xx, f_refractory_basis_vec[:,i_basis], kind='cubic'))
            
            X_refractory = np.zeros((self.nt*self.ntrial, self.n_f_refractory))
            for i_basis in range(self.n_f_refractory):
                X_refractory[:,i_basis] = self.f_refractory_basis[i_basis](refractory_spikes)
            
            self.refractory_additive_start = np.sum([effect.shape[1] for effect in self.effect_list])
            self.refractory_spikes = refractory_spikes
            self.effect_list.append(X_refractory)
            self.basis_list.append(f_refractory_basis_vec)
            self.basis_name.append(effect_type)
            
        elif effect_type == 'refractory_box':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            tau = kwargs.pop('tau',10)
            self.tau = tau
            tau = int(np.ceil(tau))
            order = kwargs.pop('order', 2)
            refractory_spikes = np.zeros_like(input_to_couple)
            for t in range(tau, input_to_couple.shape[0]):
                refractory_spikes[t, :] = input_to_couple[t-tau:t, :].sum(axis=0)
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            self.effect_list.append(X_refractory**order)
            self.basis_list.append(refractory_spikes.mean(axis=1)[:, np.newaxis])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'trial_coef':
            X_trial_coef = np.zeros((self.nt*self.ntrial, self.ntrial))
            for itrial in range(self.ntrial):
                X_trial_coef[(itrial*self.nt):((itrial+1)*self.nt), itrial] = 1

            self.trial_coef_start = np.sum([effect.shape[1] for effect in self.effect_list])
            self.effect_list.append(X_trial_coef)
            self.basis_list.append(np.diag(np.ones(self.ntrial)))
            self.basis_name.append(effect_type)

        elif effect_type == 'condition_coef':
            condition_list = self.dataset.presentation_table['stimulus_condition_id']
            condition_ids_np = np.array(self.condition_ids)
            X_condition_coef = np.zeros((self.nt*self.ntrial, len(self.condition_ids)))
            for itrial in range(self.ntrial):
                trial = self.dataset.spike_train.columns[itrial]
                current_condition = condition_list.loc[trial]
                icondition = np.where(condition_ids_np==current_condition)[0][0]
                X_condition_coef[(itrial*self.nt):((itrial+1)*self.nt), icondition] = 1
            X_condition_coef = X_condition_coef[:, X_condition_coef.sum(axis=0)!=0]
            self.effect_list.append(X_condition_coef)
            self.basis_list.append(np.diag( np.ones(X_condition_coef.shape[1]) ))
            self.basis_name.append(effect_type)
            
        elif effect_type == 'twoway_coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
                input_to_couple = input_to_couple[self.npadding:, :]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
                input_to_couple = input_to_couple[:,self.select_trials]
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            pillow_basis = make_pillow_basis(**kwargs)
            X_speed_pos = conv(input_to_couple, pillow_basis, enforce_causality=True, npadding=None)
            X_speed_neg = conv_flip(input_to_couple, pillow_basis, enforce_causality=False, npadding=None)
            X_speed_pos[:,0] += X_speed_neg[:,0]
            X_speed_neg = X_speed_neg[:,1:]
            self.effect_list.append(np.hstack((X_speed_pos,X_speed_neg)))
            
            lbasis = pillow_basis.shape[0]
            temp = np.zeros((2*lbasis+1))
            temp[lbasis+1] = 1
            basis_pos = conv(temp, pillow_basis, enforce_causality=True)
            basis_neg = conv_flip(temp, pillow_basis, enforce_causality=False)
            basis_pos[:,0] += basis_neg[:,0]
            basis_neg = basis_neg[:,1:]
            pillow_basis_twoway = np.hstack((basis_pos, basis_neg))
            self.basis_list.append(pillow_basis_twoway)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input]+" cross-pop")
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'circular':
            assert type(raw_input)==np.ndarray , "Circular effects should be from LFP phase!" 
            raise ValueError("Unfinish!")
        
        elif effect_type == 'linear':
            assert type(raw_input)==np.ndarray , "Linear effects should be from instantaneous speed!" 
            X_linear = raw_input.flatten('F')[:,np.newaxis]
            self.effect_list.append(X_linear)
            self.basis_list.append(np.ones((1,1)))
            self.basis_name.append(effect_type)
            
        elif effect_type == 'varying_linear':
            assert type(raw_input)==np.ndarray , "Varying_linear effects should be from instantaneous speed!" 
            coef_basis = inhomo_baseline(ntrial=1, 
                                         start=0,
                                         end=self.nt,
                                         dt=1, 
                                         **kwargs)
            X_varying_linear = np.zeros((self.ntrial*self.nt, coef_basis.shape[1]))
            for i in range(coef_basis.shape[1]):
                single_coef_basis = coef_basis[:,i]
                single_coef_basis = single_coef_basis[:,np.newaxis]
                X_varying_linear[:,i] = (single_coef_basis*raw_input).flatten('F')

            self.effect_list.append(X_varying_linear)
            self.basis_list.append(coef_basis)
            self.basis_name.append(effect_type)
    
        # if apply no penalty to this effect
        if apply_no_penalty == True:
            no_penalty_start = np.sum([self.basis_list[i].shape[1] for i in range(len(self.basis_list)-1)])
            no_penalty_end = np.sum([self.basis_list[i].shape[1] for i in range(len(self.basis_list))])
            self.no_penalty.append(np.arange(int(no_penalty_start), int(no_penalty_end)))

    def fit(self, target, use_all=False, verbose=True, penalty=1e-10, method='mine', max_spike=None, no_penalty_term_penalty=0, 
            smoothing=0, offset=None):
        self.use_warping = False
        if self.target is None or self.output is None or self.response is None:
            self.target = target
            if type(target) == str:
                # print(f"Assuming output is spike trains from {target}")
                self.output = utils.pooling_pop(self.membership, self.condition_ids, 
                                        self.dataset, target, 0, use_all=use_all)
                self.output = self.output[:,self.select_trials]
            elif type(target) == np.ndarray:
                self.output = target[:,self.select_trials]
            else:
                raise ValueError("target must be either str like \"probeC\" or numpy.ndarray!")
            if self.npadding is not None:
                self.output = self.output[self.npadding:, :]
            self.response = self.output.flatten('F')
        
        self.predictors = np.hstack(self.effect_list)
        if method=='mine':
            self.results = poisson_regression(self.response, self.predictors, 
                                              nt=self.nt,
                                              L2_pen=penalty, no_penalty=self.no_penalty, 
                                              no_penalty_term_penalty=no_penalty_term_penalty, smoothing=smoothing, offset=offset)
        elif method=='additional':
            self.results, self.a, self.intecept = poisson_regression_additional(self.response, self.predictors, L2_pen=penalty, no_penalty=self.no_penalty, 
                                                                                a=self.a, intecept=self.intecept)
        elif method=='pytorch':
            self.results = poisson_regression_pytorch(self.response, self.predictors, L2_pen=penalty, no_penalty=self.no_penalty)
        elif method=='logit':
            assert penalty == 0, "logit regression must use 0 penalty"
            success_fail, max_spike = get_success_fail(self.response, max_spike=max_spike, return_max_spike=True)
            self.results = sm.GLM(success_fail, self.predictors, family=sm.families.Binomial()).fit()
        else:
            assert penalty == 0, "Statistical Model package (SM) must use 0 penalty"
            self.results = sm.GLM(self.response, self.predictors, family=sm.families.Poisson()).fit()
        self.method = method
        self.max_spike = max_spike
        
        if offset is None:
            self.log_lmbd = (self.predictors@self.results.params).reshape((self.nt, self.ntrial), order='F')
        else:
            self.log_lmbd = (self.predictors@self.results.params + offset.squeeze()).reshape((self.nt, self.ntrial), order='F') 
        if self.method == 'additional':
            self.log_lmbd = f_correct(self.log_lmbd, a=self.a, intecept=self.intecept)
            
        self.nll = spike_trains_neg_log_likelihood(self.log_lmbd, self.output, max_spike=self.max_spike)
        self.nll_trialwise = spike_trains_neg_log_likelihood(self.log_lmbd, self.output, trial_wise=True, max_spike=self.max_spike)
        self.aic = 2*self.predictors.shape[1] + 2*self.nll
        self.bic = np.log(self.response.shape[0])*self.predictors.shape[1] + 2*self.nll
        self.filters = self.get_filter(ci=False)
        if verbose:
            print(f"Negative log likelihood is: {self.nll :.2f}")
            print(f"aic/2 is: {self.aic :.2f}")
        return self.results
    
    def temporal_fit(self, target, use_all=False, verbose=True, penalty=1e-10, method='mine', max_spike=None):
        pass
    
    def deviance_test(self, verbose=False):
        # Not really useful with tens of thousands of column
        sat_nll = spike_trains_neg_log_likelihood( np.log(self.output+1e-6), self.output)
        dev = 2 * (self.nll - sat_nll)
        degree_freedom = self.predictors.shape[0] - self.predictors.shape[1]
        pvalue = 1 -  scipy.stats.chi2.cdf(dev, degree_freedom)
        if verbose:
            print(sat_nll, self.nll, degree_freedom)
            print(f"p-value for deviance test is:{pvalue}")
        return pvalue
    
    def get_filter(self, ci=False):
        ### say there are one inhomo baseline and three coupling filters, 
        ### result_filter[2] contains the information for the second coupling filters
        ### if ci==True, result_filter[2][0] is the filter, result_filter[2][1] is the ci
        ### if ci==False, result_filter[2] is the filter
        effect_id_list = np.arange(len(self.basis_name))
        result_filter = []
        for effect_id in effect_id_list:
            if self.effect_type_list[effect_id] == 'trialwise_inhomogeneous_baseline':
                # result_filter.append([])
                continue
            start_col = 0
            for previous_id in range(effect_id):
                start_col += (self.effect_list[previous_id]).shape[1]
            nbasis = (self.effect_list[effect_id]).shape[1]
            end_col = start_col + nbasis
            basis = self.basis_list[effect_id]
            # estimated filter
            coef = self.results.params[start_col:end_col]
            y = (basis@coef[:,np.newaxis]).squeeze()
            # ci
            if ci:
                inv_hessian = self.results.inv_hessian[start_col:end_col, start_col:end_col]
                one_sigma_ci = np.sqrt(np.diag(basis@inv_hessian@basis.T))
                result_filter.append([y,one_sigma_ci])
            else:
                result_filter.append(y)
        return result_filter
    
    def get_filter_output(self, intermediate=True, trial_wise=False, ci=False):
        ### The function is used for both fitting fit_time_warping and for display final plot. 
        ### say there are one inhomo baseline and three coupling filters, 
        ### result_output[2] contains the information for the second coupling filters
        ### if ci==True, result_output[2][0] is the filter, result_output[2][1] is the ci
        ### if ci==False, result_output[2] is the filter
        if ci==True:
            intermediate = False
        # assert intermediate==False or (intermediate==True and ci==False) 
        effect_id_list = np.arange(len(self.basis_name))
        result_output = []
        for effect_id in effect_id_list:
            start_col = 0
            for previous_id in range(effect_id):
                start_col += (self.effect_list[previous_id]).shape[1]
            nbasis = (self.effect_list[effect_id]).shape[1]
            end_col = start_col + nbasis
            
            if intermediate:
                # If for fitting fit_time_warping
                if self.basis_name[effect_id] in ["inhomogeneous_baseline"]:
                    predicters_temp = self.basis_list[effect_id]
                    if trial_wise:
                        predicters_temp = np.tile(predicters_temp, (self.ntrial, 1))
                    coef = self.results.params[start_col:end_col]
                    output = (predicters_temp@coef[:,np.newaxis]).squeeze()
                    if trial_wise:
                        output_all_trial = output
                else:
                    predicters_temp = self.effect_list[effect_id]
                    coef = self.results.params[start_col:end_col]
                    output_all_trial = (predicters_temp@coef[:,np.newaxis]).squeeze()
                    output_mat = output_all_trial.reshape((self.nt, self.ntrial), order='F')
                    output = output_mat.mean(axis=1)
            else:
                # If for final display
                coef = self.results.params[start_col:end_col]
                inv_hessian = self.results.inv_hessian[start_col:end_col, start_col:end_col]
                effect = self.effect_list[effect_id]
                trial_mean = effect.reshape((self.nt, self.ntrial, nbasis), order='F').mean(axis=1)
                output_all_trial = (effect@coef[:,np.newaxis]).squeeze()
                output_mat = output_all_trial.reshape((self.nt, self.ntrial), order='F')
                output = output_mat.mean(axis=1)
                one_sigma_ci_from_coef = np.sqrt(np.diag(trial_mean@inv_hessian@trial_mean.T))
                one_sigma_ci_from_trial = output_mat.std(axis=1)/np.sqrt(self.ntrial)
                one_sigma_ci = one_sigma_ci_from_coef + one_sigma_ci_from_trial
            if ci:
                result_output.append([output,one_sigma_ci])
            else:
                if trial_wise:
                    result_output.append(output_all_trial)
                else:
                    result_output.append(output)
        return result_output
    
    def get_filter_output_merge(self, trial_wise=False, ci=False):
        if trial_wise:
            result_output = []
            raw_result_output_list = self.get_filter_output(trial_wise=True, ci=False)
            for ieffect, raw_result_output in enumerate(raw_result_output_list):
                if self.effect_type_list[ieffect]=='inhomogeneous_baseline': 
                    result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                    ibaseline = ieffect
                elif self.effect_type_list[ieffect]=='coupling':
                    result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                    if utils.PROBE_CORRESPONDING_INVERSE[self.basis_name[ieffect][-2:]] == self.target:
                        iselfeffect = ieffect
                elif self.effect_type_list[ieffect]=='trial_coef':
                    # add refractory to itself's own effect
                    result_output[ibaseline] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
                elif self.effect_type_list[ieffect]=='refractory':
                    result_output[iselfeffect] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
                else:
                    raise ValueError("only support trial_coef and refractory in addition to inhomogeneous_baseline and coupling!")
            return result_output
        else:
            raise ValueError("Unfinished!")
            # result_output = []
            # raw_result_output_list = self.get_filter_output(trial_wise=False, ci=False)
            # for ieffect, raw_result_output in enumerate(raw_result_output_list):
            #     if self.effect_type_list[ieffect]=='inhomogeneous_baseline': 
            #         result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
            #         ibaseline = ieffect
            #     elif self.effect_type_list[ieffect]=='coupling':
            #         result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
            #         if utils.PROBE_CORRESPONDING_INVERSE[self.basis_name[ieffect][-2:]] == self.target:
            #             iselfeffect = ieffect
            #     elif self.effect_type_list[ieffect]=='trial_coef':
            #         # add refractory to itself's own effect
            #         result_output[ibaseline] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            #     elif self.effect_type_list[ieffect]=='refractory':
            #         result_output[iselfeffect] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            #     else:
            #         raise ValueError("only support trial_coef and refractory in addition to inhomogeneous_baseline and coupling!")
            # return result_outputinhomo
        
    def test(self, test_trials, use_all=False, verbose=False):
        if self.dataset is not None:
            self.test_model = PP_GLM(dataset=self.dataset, 
                                    select_trials=test_trials, 
                                    membership=self.membership, 
                                    condition_ids=self.condition_ids)
        else:
            self.test_model = PP_GLM(ntrial=self.ntrial, nt=self.nt, select_trials=test_trials)
        
        if type(self.target) == str:
            # print(f"Assuming output is spike trains from {target}")
            self.test_model.output = utils.pooling_pop(self.test_model.membership, self.test_model.condition_ids, 
                                                       self.test_model.dataset, self.target, 0, use_all=use_all)
            self.test_model.output = self.test_model.output[:,test_trials]
        elif type(self.target) == np.ndarray:
            self.test_model.output = self.target[:,test_trials]
        else:
            raise ValueError("target must be either str like \"probeC\" or numpy.ndarray!")
        if self.npadding is not None:
                self.test_model.output = self.test_model.output[self.npadding:, :]
        for i_effect, effect_type in enumerate(self.effect_type_list):
            raw_input = self.raw_input_list[i_effect]
            kwargs = self.kwargs_list[i_effect]
            self.test_model.add_effect(effect_type, raw_input=raw_input, **kwargs)

        self.test_model.response = self.test_model.output.flatten('F')
        self.test_model.predictors = np.hstack(self.test_model.effect_list)
        self.test_model.results = self.results
        
        if self.use_warping==False:
            self.test_model.log_lmbd = (self.test_model.predictors@self.test_model.results.params).\
                reshape((self.test_model.nt, self.test_model.ntrial), order='F')
            self.test_model.log_lmbd_ci = (self.test_model.predictors@self.test_model.results.bse).\
                reshape((self.test_model.nt, self.test_model.ntrial), order='F')
            self.test_model.nll = spike_trains_neg_log_likelihood(self.test_model.log_lmbd, self.test_model.output)
            self.test_model.nll_trialwise = spike_trains_neg_log_likelihood(self.test_model.log_lmbd, self.test_model.output, trial_wise=True)
            self.test_model.aic = self.test_model.predictors.shape[1] + self.test_model.nll
            return self.test_model.nll
        else:
            inhomo_template = self.test_model.get_filter_output(trial_wise=False, ci=False)[0]
            result_output = self.test_model.get_filter_output(trial_wise=True, ci=False)
            inhomo = result_output[0]
            total_output = np.vstack((result_output)).T
            total_output = total_output.sum(axis=1)
            minus_one_output = total_output - inhomo
            best_shift, nll  = get_best_shift(self.test_model.time_line, inhomo_template, minus_one_output, 
                                            self.test_model.response, self.test_model.nt, max_spike=self.max_spike, warp_interval=self.warp_interval, 
                                            a=self.a, intecept=self.intecept)
            return nll
    
    def get_initial_peaks(self):
        if self.dataset is not None:
            inhomo_model = PP_GLM(dataset=self.dataset, 
                                    select_trials=self.select_trials, 
                                    membership=self.membership, 
                                    condition_ids=self.condition_ids)
        else:
            inhomo_model = PP_GLM(ntrial=self.ntrial, nt=self.nt, select_trials=self.select_trials)
        for i_effect, effect_type in enumerate(self.effect_type_list):
            if effect_type == "inhomogeneous_baseline":
                raw_input = self.raw_input_list[i_effect]
                kwargs = self.kwargs_list[i_effect]
                inhomo_model.add_effect(effect_type, raw_input=raw_input, **kwargs)
                break
        inhomo_model.fit_time_warping_baseline(target=self.target, max_iter=3, warp_interval=self.warp_interval, 
                                                method='mine', penalty=1e-2, verbose=False, use_all=self.use_all)
        self.inhomo_model = inhomo_model
        return inhomo_model.shifts

    def fit_time_warping_baseline(self, target, use_all=False, max_iter=100, penalty=1e-10, warp_interval=[[0, 0.15], [0.15, 0.35]], 
                                  tol=1e-10, method='mine', max_spike=None, fix_shifts=None, initial_shifts=None, verbose=True, 
                                  no_penalty_term_penalty=0, acc_warping=False, smoothing=0, offset=None):
        assert 'inhomogeneous_baseline' in self.effect_type_list, "You must create an inhomogeneous baseline before changing it to time-warp baseline!"
        
        self.use_warping = True
        self.warp_interval = warp_interval
        self.target = target
        self.use_all = use_all
        
        ALPHA = 0.5   # to smooth the optimization process
        BETA = 0.0   # to smooth the optimization process
        THETA = 0.9   # Mean converge

        if offset is not None:
            if offset.shape[1] == len(self.select_trials):
                offset = offset[:, self.select_trials]
            offset = offset.flatten('F')[:, np.newaxis]
        
        # Find the effect index that should be warpped
        i_effect = [i_effect for i_effect,effect_type in enumerate(self.effect_type_list) 
            if effect_type=='inhomogeneous_baseline'][0]
        
        nll_old = np.inf
        X_baseline_original = self.effect_list[i_effect]

        # Initialization
        if initial_shifts is None:
            # self.shifts = np.zeros((self.ntrial, 2*len(warp_interval)))
            pass
        else:
            if initial_shifts=="peaks":
                initial_shifts = self.get_initial_peaks()
            elif type(initial_shifts) == np.ndarray:
                assert initial_shifts.shape == (self.ntrial, 2*len(warp_interval)), "initial_shifts shape wrong!"
                
            else:
                raise ValueError("nitial_shifts can only be None (fit first); \"peaks\" (fit inhomogeneous and get firing rate peaks);"/
                    +"np.array (from existing initial_shifts)")
            self.initial_shifts = initial_shifts
            self.shifts = initial_shifts
            X_baseline_warp = apply_warping_to_predictors(self.time_line, X_baseline_original, self.shifts, self.nt, 
                                                        warp_interval=warp_interval)
            self.effect_list[i_effect] = X_baseline_warp
        
        if fix_shifts is None:
            for iter in range(max_iter):
                # update coef (based on *warped* effect_list[i_effect])
                self.fit(target, use_all=use_all, verbose=False, penalty=penalty, method=method, max_spike=max_spike, 
                         no_penalty_term_penalty=no_penalty_term_penalty, smoothing=smoothing, offset=offset)
                
                # 'inhomo' and 'inhomo_template' are based on 'basis_list', so they are not warped
                
                inhomo_template = self.get_filter_output(trial_wise=False, ci=False)[i_effect]
                result_output = self.get_filter_output(trial_wise=True, ci=False)
                inhomo = result_output[i_effect]
                total_output = np.vstack((result_output)).T
                total_output = total_output.sum(axis=1)
                if offset is not None:
                    total_output += offset.squeeze()
                minus_one_output = total_output - inhomo
                if verbose:
                    print(f"After the {iter} th iteration of fitting: {self.nll}")
                    
                # Update shifts (based on non-warping inhomo baseline)
                if acc_warping is False:
                    best_shift, nll  = get_best_shift(self.time_line, inhomo_template, minus_one_output, 
                                                    self.response, self.nt, max_spike=self.max_spike, warp_interval=warp_interval, 
                                                    a=self.a, intecept=self.intecept)
                else:
                    best_shift, nll  = get_best_shift(self.time_line, inhomo_template, minus_one_output, 
                                                    self.response, self.nt, max_spike=self.max_spike, warp_interval=warp_interval, 
                                                    a=self.a, intecept=self.intecept, previous_shifts=self.shifts)
                if iter==0:
                    self.shifts = best_shift
                else:
                    for i_interval in range(len(warp_interval)):
                        self.shifts[:,2*i_interval+1] = BETA*self.shifts[:,2*i_interval+1] + (1-BETA)*best_shift[:,2*i_interval+1]
                        # self.shifts[:,2*i_interval] = ALPHA*self.shifts[:,2*i_interval] + (1-ALPHA)*best_shift[:,2*i_interval]
                        self.shifts[:,2*i_interval] = self.shifts[:,2*i_interval+1].mean()
                # if iter != max_iter-1:
                    # for i_interval in range(len(warp_interval)):
                        # self.shifts[:,2*i_interval+1] = (self.shifts[:,2*i_interval+1] - self.shifts[:,2*i_interval])*THETA + self.shifts[:,2*i_interval]
                X_baseline_warp = apply_warping_to_predictors(self.time_line, X_baseline_original, self.shifts, self.nt, 
                                                            warp_interval=warp_interval)
                self.effect_list[i_effect] = X_baseline_warp
                
                if verbose:
                    print(f"After the {iter} th iteration of warping: {nll}")
                
                # if not_updating, break
                if nll_old - nll < tol:
                    # Finished fitting
                    pass
                    # break
                nll_old = nll
        else:
            self.shifts = fix_shifts
            # for i_interval in range(len(warp_interval)):
            #     self.shifts[:,2*i_interval+1] = (self.shifts[:,2*i_interval+1] - self.shifts[:,2*i_interval])*THETA + self.shifts[:,2*i_interval]
            X_baseline_warp = apply_warping_to_predictors(self.time_line, X_baseline_original, self.shifts, self.nt, 
                                                        warp_interval=warp_interval)
            self.effect_list[i_effect] = X_baseline_warp
            
        self.fit(target, use_all=use_all, verbose=False, penalty=penalty, method=method, max_spike=max_spike, 
                 no_penalty_term_penalty=no_penalty_term_penalty, smoothing=smoothing, offset=offset)
        # Finished fitting
        if fix_shifts is None and iter == max_iter:
            print("Maximum iteration reach!")
        self.basis_name[i_effect] = 'time_warping_inhomogeneous_baseline'
        # self.inhomo_template = inhomo_template
        # self.nll = nll
        self.use_warping = True

    def fit_individual_history_and_get_offset():
        pass
        
def fit_individual(model, target_probe, coupling_filter_params, verbose=True, penalty=1e-10, method='mine', max_spike=None):
    # print("Starting now!")
    model = copy.deepcopy(model)
    # Find total number of neurons needed and their corresponding trials where they are classified as cross-pop
    cross_pop_list = []
    cross_pop_conditions = {}
    for neuron in model.membership[0].index:
        if model.membership[1].loc[neuron]['probe'] == target_probe:
            for i, member in enumerate(model.membership):
                if member.loc[neuron]['group_id'] == 0:
                    if neuron not in cross_pop_conditions:
                        cross_pop_list.append(neuron)
                        cross_pop_conditions[neuron] = [model.condition_ids[i]]
                    else:
                        cross_pop_conditions[neuron].append(model.condition_ids[i])
    major_cross_pop_list = []
    for neuron in cross_pop_list:
        if len(cross_pop_conditions[neuron])>= 20:
            major_cross_pop_list.append(neuron)
    n_neuron = len(major_cross_pop_list)

    # Get the design matrix for each 
    major_cross_pop_trial_list = []
    design_mat = []
    response_vec = []
    for i, neuron in enumerate(major_cross_pop_list):
        design_col = []
        major_cross_pop_trial_list.append( np.array([model.dataset.presentation_table['stimulus_condition_id'][stimuli_id] in cross_pop_conditions[neuron]
                for stimuli_id in model.dataset.spike_train.columns]) )
        spike_train_ind = np.zeros((model.nt+model.npadding, model.dataset.spike_train.shape[1]))
        for itrial in range(model.dataset.spike_train.shape[1]):
            spike_train_ind[:,itrial] = model.dataset.spike_train.loc[neuron, model.dataset.spike_train.columns[itrial]]
        pillow_basis = make_pillow_basis(**coupling_filter_params)
        spike_train_ind = spike_train_ind[:, np.logical_and(model.select_trials, major_cross_pop_trial_list[-1])]
        X_history = conv(spike_train_ind, pillow_basis, npadding=model.npadding)
        empty = np.zeros_like(X_history)
        design_col = n_neuron*[empty]
        design_col[i] = X_history
        selection = []
        for j, incrosspop in enumerate(model.select_trials):
            if incrosspop:
                if major_cross_pop_trial_list[-1][j]:
                    selection = selection + model.nt*[True]
                else:
                    selection = selection + model.nt*[False]
        design_col = [model.predictors[selection]] + design_col
        design_mat.append(design_col)
        response_vec.append(spike_train_ind[model.npadding:,:].flatten('F')[:, np.newaxis])

    design_mat = np.block(design_mat)
    response_vec = np.vstack(response_vec).flatten('F')
    # print("Start doing poisson regression!")
    # print(X_history.shape)
    # print(design_mat.shape)
    # print(response_vec.shape)
    model.design_mat = design_mat
    model.response_vec = response_vec
    model.results = poisson_regression(response_vec, design_mat, L2_pen=penalty, no_penalty=model.no_penalty)

    model.log_lmbd = (design_mat@model.results.params).reshape((model.nt, -1), order='F')
    model.nll = spike_trains_neg_log_likelihood(model.log_lmbd, response_vec.reshape((model.nt, -1), order='F'))
    model.nll_trialwise = spike_trains_neg_log_likelihood(model.log_lmbd, response_vec.reshape((model.nt, -1), order='F'), trial_wise=True)
    model.aic = design_mat.shape[1] + model.nll
    # model.filters = model.get_filter(ci=False)
    if verbose:
        print(f"Negative log likelihood is: {model.nll :.2f}")
        print(f"aic/2 is: {model.aic :.2f}")
    return model

#%% Binomial GLM 'logit'
def get_link(method):
    if method=='logit':
        link = lambda x: 1/(1+np.exp(-x))
    else:
        link = np.exp
    return link

def get_success_fail(response, return_max_spike=False, max_spike=None):
    success_fail = np.zeros((*response.shape, 2))
    if response.ndim == 1:
        success_fail[:,0] = response
    else:
        success_fail[:,:,0] = response
    if max_spike is None:
        max_spike = int(response.max() * 1) + 1
    if response.ndim == 1:
        success_fail[:,1] = max_spike - response
    else:
        success_fail[:,:,1] = max_spike - response
    if return_max_spike:
        return success_fail, max_spike
    else:
        return success_fail
    
#%% Time-warping baseline
### None MP
def get_best_shift(time_line, inhomo_template, minus_one_output, response, nt, max_spike=None, warp_interval=[[0, 0.15], [0.15, 0.35]], 
                   a=None, intecept=None, previous_shifts=None):
    ntrial = int(len(response)/nt)
    best_shifts = np.zeros((ntrial, 2*len(warp_interval)))
    total_nll = 0
    for itrial in range(ntrial):
        if previous_shifts is None:
            previous_shifts_trial = None
        else:
            previous_shifts_trial = previous_shifts[itrial, :]
        best_shifts_trial, best_nll_trial = get_best_shift_single(time_line, 
                                                inhomo_template, 
                                                minus_one_output[itrial*nt:(itrial+1)*nt], 
                                                response[itrial*nt:(itrial+1)*nt], 
                                                max_spike=max_spike, 
                                                warp_interval=warp_interval,
                                                a=a, 
                                                intecept=intecept,
                                                previous_shifts=previous_shifts_trial)
        best_shifts[itrial, :] = best_shifts_trial
        total_nll += best_nll_trial
    return best_shifts, total_nll

def get_best_shift_single(time_line, inhomo_template, minus_one_output, response, max_spike=None, warp_interval=[[0, 0.15], [0.15, 0.35]],
                          a=None, intecept=None, previous_shifts=None):
    to_return = []
    for i_interval, interval in enumerate(warp_interval):
        if previous_shifts is None or np.sum(previous_shifts)==0:
            search_grid = np.arange(interval[0], interval[1], 0.002)
        else:
            # print(previous_shifts)
            previous = np.round(previous_shifts[2*i_interval+1]/0.002) * 0.002
            search_grid = np.arange(max(interval[0], previous-0.05), 
                                    min(interval[1], previous+0.05), 
                                    0.002)
            # print(i_interval, interval[0], previous_shifts[2*i_interval+1], max(interval[0], previous-0.000),min(interval[1], previous+0.002) )
        peak = time_line[np.sum(time_line<interval[0])+np.argmax(inhomo_template[np.logical_and(time_line>=interval[0], time_line<interval[1])])]
        sources = [interval[0], peak, interval[1]]
        best_nll = np.inf
        
        for moved_peak in search_grid:
            targets = [interval[0], moved_peak, interval[1]]
            warped = linear_time_warping_single(time_line, inhomo_template, sources, targets, verbose=False)
            if a is not None:
                nll = spike_trains_neg_log_likelihood(f_correct(warped+minus_one_output, a, intecept), response, max_spike=max_spike)
            else:
                nll = spike_trains_neg_log_likelihood(warped+minus_one_output, response, max_spike=max_spike)
            if nll <= best_nll:
                best_shift_peak = moved_peak
                best_warped = warped
                best_nll = nll
        to_return.append(peak)
        to_return.append(best_shift_peak)
    return np.array(to_return), best_nll

def apply_warping_to_predictors(time_line, X_baseline_original, shifts, nt, warp_interval=[[0, 0.15], [0.15, 0.35]]):
    ntrial = int(X_baseline_original.shape[0]/nt)
    X_baseline_warp = np.zeros_like(X_baseline_original)
    for itrial in range(ntrial):
        sources = []
        targets = []
        for i_interval, interval in enumerate(warp_interval):
            sources.append([warp_interval[i_interval][0], shifts[itrial, 2*i_interval], warp_interval[i_interval][1]])
            targets.append([warp_interval[i_interval][0], shifts[itrial, 2*i_interval+1], warp_interval[i_interval][1]])
        # print(sources)
        for i_col in range(X_baseline_warp.shape[1]):
            for i_interval, interval in enumerate(warp_interval):
                if i_interval==0:
                    to_warp = X_baseline_original[itrial*nt:(itrial+1)*nt, i_col]
                    warped = linear_time_warping_single(time_line, to_warp, sources[i_interval], targets[i_interval], verbose=False)
                else:
                    to_warp = warped
                    warped = linear_time_warping_single(time_line, to_warp, sources[i_interval], targets[i_interval], verbose=False)
            X_baseline_warp[itrial*nt:(itrial+1)*nt, i_col] = warped
    return X_baseline_warp

def linear_time_warping_single(t, f, sources, targets, verbose=True):
    """Time warping function for the intensity.

    Args:
        sources: Positions of input `f` needed to be shifted.
        targets: New positions of the sources. The rest of curve will be shifted
            linearly in between sources.
    """
    sources = np.array(sources)
    targets = np.array(targets)
    t_interp = t.copy()

    for i in range(1, len(sources)):
        source_left = sources[i-1]
        source_right = sources[i]
        target_left = targets[i-1]
        target_right = targets[i]

        # Linearly stretch the source intervals to the target interverals.
        t_target_index = (t >= target_left) & (t < target_right)
        t_target = t[t_target_index]
        if len(t_target) == 0:
            continue
        t_interp[t_target_index] = ((t_target - target_left) *
            (source_right - source_left) / (target_right - target_left)
            + source_left)
    # Run the linear interporation using the sample points.
    f_warp = np.interp(t_interp, t, f)
    return f_warp
    
#%% Simulation

def simulate_baseline_coupling_refractory(model_list, nepoch=1, verbose=False, offset=0):
    MAX_FIRING_RATE = np.log(1000)
    nneuron = len(model_list)
    taus = np.array([model.tau for model in model_list])
    probe_list = model_list[0].dataset.selected_probes
    
    spikes_rcd = np.zeros((model_list[0].nt, nepoch*model_list[0].ntrial, len(model_list)))
    log_firing_rate_rcd = np.zeros((model_list[0].nt, nepoch*model_list[0].ntrial, len(model_list)))
    peaks_rcd = np.zeros((2, nepoch*model_list[0].ntrial, len(model_list)))
    
    probe2num = {}
    for iprobe, probe in enumerate(probe_list):
        probe2num[probe] = iprobe
    
    # Get three dimension matrix of coupling filters for better computing. 
    max_histories = 1
    max_histories_refractory = 1
    nt = model_list[0].nt
    allowed_effect_type = ['inhomogeneous_baseline', 'coupling', 'trial_coef', 'refractory_additive']
    coupling_mat = np.zeros((max_histories, nneuron, nneuron))
    f_refractory_list = []
    
    for ineuron in range(nneuron):
        assert all(effect_type in allowed_effect_type for effect_type in model_list[ineuron].effect_type_list), \
            "For refractory additive models with time warping only!"
        model = model_list[ineuron]
        for ieffect, effect_type in enumerate(model.effect_type_list):

            if effect_type in ['inhomogeneous_baseline', ]:
                pass

            elif effect_type in ['coupling']:
                nhistories = len(model.filters[ieffect])
                probe_name = utils.PROBE_CORRESPONDING_INVERSE[model.basis_name[ieffect][-2:]]
                iprobe = probe2num[probe_name]
                if nhistories > max_histories:
                    coupling_mat_old = coupling_mat
                    coupling_mat = np.zeros((nhistories, nneuron, nneuron))
                    coupling_mat[-max_histories:, :, :] = coupling_mat_old
                    max_histories = nhistories

                coupling_mat[-nhistories:, iprobe, ineuron] = np.flip(model.filters[ieffect])

            elif effect_type in ['refractory_additive']:
                f_refractory_vec = model.filters[ieffect]
                f_refractory_xx = model.f_refractory_xx

                # Get a longer time line for lambda
                dt = f_refractory_xx[1] - f_refractory_xx[0]
                new_f_refractory_xx = np.arange(f_refractory_xx.shape[0]+50)*dt
                # The end point, which is used to get the parameter k of k*x**2
                last_point = [f_refractory_xx[-1], f_refractory_vec[-1]]
                # quadratic_refractory = new_f_refractory_xx**2*(last_point[1]/last_point[0]**2)/2 + last_point[1]/2
                quadratic_refractory = new_f_refractory_xx**2*(last_point[1]/last_point[0]**2)
                # Get the longer f_refractory_vec in discrete form
                new_f_refractory_vec = copy.deepcopy(quadratic_refractory)
                new_f_refractory_vec[0:len(f_refractory_vec)] = f_refractory_vec
                # Transfer discrete f_refractory_vec to a continuous callable function
                f_refractory = scipy.interpolate.interp1d(new_f_refractory_xx, 
                                                          new_f_refractory_vec, 
                                                          kind='cubic', 
                                                          fill_value="extrapolate")
                f_refractory_list.append(f_refractory)
                if verbose:
                    plt.plot(new_f_refractory_vec)

    for iepoch in tqdm( range(nepoch) ):
        for itrial in range(model_list[0].ntrial):

            baseline_mat = np.zeros((nt, nneuron))
            spikes = np.zeros((nt, nneuron, 1))

            for ineuron in range(nneuron):
                model = model_list[ineuron]
                for ieffect, effect_type in enumerate(model.effect_type_list):

                    if effect_type in ['inhomogeneous_baseline', ]:
                        baseline_mat[:, ineuron] = model.filters[ieffect]
                        baseline_mat[:, ineuron] = apply_warping_to_predictors(model.time_line, 
                                                                                baseline_mat[:, ineuron][:,None], 
                                                                                model.shifts[itrial,:][None,:], 
                                                                                model.nt).squeeze()
                        baseline_mat[:, ineuron] += model.results.params[model.trial_coef_start+itrial] + offset

            log_firing_rate = baseline_mat[:,:,np.newaxis]
            spikes[0,:,0] = np.random.poisson(np.exp(log_firing_rate[0,:,0]))
            recent_spike_sum = copy.deepcopy(spikes[0, :, 0])

            for t in range(1, nt):
                recent_spike_sum *= np.exp(-1000.0/model_list[0].fps/taus)
                nhistories = min(t, max_histories)
                temp_log_firing_rate = (coupling_mat[-nhistories:, :, :] * spikes[(t-nhistories):(t), :, :]).sum(axis=(0, 1))
                refractory = np.zeros(nneuron)
                for ineuron in range(nneuron):
                    refractory[ineuron] = f_refractory_list[ineuron](recent_spike_sum[ineuron]/taus[ineuron])
                log_firing_rate[t,:,0] += temp_log_firing_rate + refractory
                # log_firing_rate[t,:,0] = np.minimum(log_firing_rate[t,:,0], MAX_FIRING_RATE)
                spikes[t,:,0] = np.random.poisson(np.exp(log_firing_rate[t,:,0]))
                recent_spike_sum += spikes[t, :, 0]

            log_firing_rate = log_firing_rate.squeeze()
            spikes = spikes.squeeze()

            log_firing_rate_rcd[:, iepoch*model_list[0].ntrial + itrial, :] = log_firing_rate
            spikes_rcd[:, iepoch*model_list[0].ntrial + itrial, :] = spikes
            for ineuron in range(nneuron):
                peaks_rcd[0, iepoch*model_list[0].ntrial + itrial, ineuron] = \
                    np.nanargmax(utils.kernel_smoothing(np.exp(log_firing_rate[:150, ineuron])[:, np.newaxis], std=40))
                peaks_rcd[1, iepoch*model_list[0].ntrial + itrial, ineuron] = \
                    np.nanargmax(utils.kernel_smoothing(np.exp(log_firing_rate[150:, ineuron])[:, np.newaxis], std=40))
                
    return spikes_rcd, log_firing_rate_rcd, peaks_rcd

def simulate_baseline_coupling(baseline_mat, coupling_mat, ntrial=1):
    MAX_FIRING_RATE = np.log(10000)
    max_histories, _, nneuron = coupling_mat.shape
    nt = baseline_mat.shape[0]
    
    spikes_rcd = np.zeros((nt, nneuron, ntrial))
    for itrial in range(ntrial):
        spikes = np.zeros((nt, nneuron, 1))
        log_firing_rate = copy.deepcopy(baseline_mat[:,:,np.newaxis])
        spikes[0,:,0] = np.random.poisson(np.exp(log_firing_rate[0,:,0]))
        for t in range(1, nt):
            nhistories = min(t, max_histories)
            temp_log_firing_rate = (coupling_mat[-nhistories:, :, :] * spikes[(t-nhistories):(t), :, :]).sum(axis=(0, 1))
            log_firing_rate[t,:,0] += temp_log_firing_rate
            log_firing_rate[t,:,0] = np.minimum(log_firing_rate[t,:,0], MAX_FIRING_RATE)
            spikes[t,:,0] = np.random.poisson(np.exp(log_firing_rate[t,:,0]))
        
        log_firing_rate = log_firing_rate.squeeze()
        spikes = spikes.squeeze()
        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]
        spikes_rcd[:,:,itrial] = spikes
    return spikes_rcd, log_firing_rate


#%% Plotting GLM
def plot_GLM_one_effect(model, effect_id, results=None, title=None, label=None, color=None, use_exp=None, linewidth=1):
    start_col = 0
    for previous_id in range(effect_id):
        start_col += (model.effect_list[previous_id]).shape[1]
    nbasis = (model.effect_list[effect_id]).shape[1]
    end_col = start_col + nbasis
    if results is None:
        results = model.results
    if use_exp is None:
        if model.basis_name[effect_id] in ['inhomogeneous_baseline',
                                        'homogeneous_baseline', 
                                        'time_warping_inhomogeneous_baseline']:
            use_exp = True
        else:
            use_exp = False
    try:
        # try to get standard error from "results", if failed, just ignore standard error
        utils.plot_filter(model.basis_list[effect_id], results.params[start_col:end_col], 
                      results.bse[start_col:end_col], label=label, color=color, exp=use_exp, linewidth=linewidth)
    except:
        utils.plot_filter(model.basis_list[effect_id], results.params[start_col:end_col], 
                np.zeros(end_col-start_col), label=label, color=color, exp=use_exp, linewidth=linewidth)
    plt.title(title)
    plt.legend()
    if model.basis_name[effect_id] == 'twoway_coupling':
        length = int(model.basis_list[effect_id].shape[0]/2)
        plt.xticks([0, length, length*2], [-length, 0, length])
        
    
def plot_GLM_compare(model, effect_id_list=None,  results_list=None, title_list=None, label_list=None, color_list=['r','b'] ):
    if effect_id_list is None:
        effect_id_list = np.arange(len(model.basis_name))
    if title_list is None:
        title_list = [ model.basis_name[i] for i in effect_id_list]
    if results_list is None:
        results_list = [model.results]
    if label_list is None:
        label_list = [ ' ' for i in effect_id_list]
    i_effect = 0
    for effect_id in effect_id_list:
        for i_results in range(len(results_list)):
            plot_GLM_one_effect(model, 
                                effect_id, 
                                results=results_list[i_results], 
                                title=title_list[i_effect], 
                                label=label_list[i_results], 
                                color=color_list[i_results])
        i_effect += 1
        plt.show()

#%% Get predictors for coupling effects
def conv_flip(raw_input, kernel, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of a single trial
        kernel (2d vector): a (nt, nbasis) matrix, which contains multiple basis

    Returns:
        X [type]: [description]
    """
    raw_input = np.flipud(raw_input)
    nbasis = kernel.shape[1]
    return np.flipud(conv(raw_input, kernel, npadding=npadding, enforce_causality=enforce_causality))

def conv(raw_input, kernel, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        raw_input (2d matrix): spike trains of multiple trials
        kernel (2d matrix): a (nt, nbasis) matrix, which contains multiple basis

    Returns:
        X [type]: [description]
    """
    if kernel.ndim == 1:
        kernel = kernel[:,np.newaxis]
    if raw_input.ndim == 1:
        raw_input = raw_input[:,np.newaxis]
    nbasis = kernel.shape[1]
    nt, ntrial = raw_input.shape
    if npadding is not None:
        nt = nt - npadding
    X = np.zeros((nt*ntrial,nbasis))
    for ibasis in range(nbasis):
        X[:,ibasis] = conv_multi_trial(raw_input, kernel[:,ibasis], merge_trial=True, npadding=npadding, enforce_causality=enforce_causality)
    return X

def conv_multi_trial(raw_input, kernel, merge_trial=False, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of a single trial
        kernel (1d vector): one basis

    Returns:
        [type]: [description]
    """
    if raw_input.ndim == 1:
        raw_input = raw_input[:,np.newaxis]
    nt, ntrial = raw_input.shape
    if enforce_causality:
        kernel = np.hstack((np.array([0]), kernel))
    nn = nt + len(kernel) - 1
    G = ifft(fft(raw_input,nn,axis=0)*fft(kernel,nn)[:,np.newaxis],axis=0)
    G = G[0:len(raw_input)].real
    G[np.abs(G)<1e-10] = 0
    if npadding is not None:
        G = G[npadding:,:]
    if merge_trial:
        G = G.flatten('F')
    return G

#%% Make basis for inhomo baseline, coupling filter (Pillow basis), etc.
def inhomo_baseline(ntrial=1, start=0, end=1e3, dt=1, num=10, add_constant_basis=False, 
                    extend_zeros=None, apply_trial=None):
    if extend_zeros is not None:
        end -= extend_zeros
    basis = make_b_spline_basis(
        t_min=start, 
        t_max=end, 
        dt=dt, 
        num_basis=num, 
        add_constant_basis=add_constant_basis, 
        verbose=False)
    if basis.ndim == 1:
        basis = basis[:,np.newaxis]
    nt = basis.shape[0]
    if extend_zeros is not None:
        basis = np.vstack((basis, np.zeros((extend_zeros, num))))
    if apply_trial is None:
        baseline = np.tile(basis, (ntrial, 1))
    else:
        baseline = np.tile(np.zeros(basis.shape), (ntrial, 1))
        for i in range(ntrial):
            if apply_trial[i]:
                baseline[(i*nt):(i*nt+nt)] = basis
    return baseline

def make_pillow_basis(num=10, peaks_min=0, peaks_max=100, nonlinear=0.2, dt=1, verbose=False):
    """ Generating raised cosine basis

    Args:
        num (int, optional): Number of basis. Defaults to 10.
        peaks_min (float, optional): Position of the first basis peak. Defaults to 0.
        peaks_max (float, optional): Position of the last basis peak. Defaults to 100.
        h_nonlin (float, optional): Range from 0 to 1. Determines how nonlinear these basis functions would be. Defaults to 0.2.
        dt (int/float, optional): Length of time bin. Defaults to 1.

    Returns:
        ihbasis: nt by num matrix, each column for each basis
    """    
    
    assert 0<=nonlinear<=1, "h_nonlin should be from 0 to 1"
    nonlinear = nonlinear*peaks_max
    nlin = lambda x: np.log(x+1e-10)
    invnl = lambda x: np.exp(x)-1e-10
    hpeaks = np.array([peaks_min, peaks_max])
    yrnge = nlin(hpeaks+nonlinear)

    db = np.diff(yrnge)[0]/(num-1)
    ctrs = np.linspace(yrnge[0], yrnge[1], num)[None,:]
    mxt = (invnl(yrnge[1]+2*db)-nonlinear).astype(int)
    iht = np.arange(0,mxt,dt)[:,None]
    nt = len(iht)
    ff = lambda x, c, dc: (np.cos(np.maximum(-np.pi,np.minimum(np.pi, (x-c)*np.pi/dc/2)))+1)/2
    ihbasis = ff(np.tile(nlin(iht+nonlinear), (1, num)), np.tile(ctrs, (nt, 1)), db)
    if verbose:
        fontsize = 20
        plt.figure()
        plt.plot(ihbasis, '-')
        plt.ylabel('log firing rate',fontsize=fontsize)
        plt.xlabel('time (ms)', fontsize=fontsize) 
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()
    return ihbasis

def make_b_spline_basis(
    num_basis=10,
    t_max=1000,
    t_min=0,
    add_constant_basis=False,
    dt=1,
    spline_order=2,
    verbose=False):
    """Constructs B-spline basis with knots equal distance.

    Args:
        t_range: [left_end, right_end].
    """
    # construct_b_spline_basis
    num_knots = num_basis-spline_order+1
    knots = np.linspace(t_min, t_max, num_knots)
    knots = np.hstack((np.ones(spline_order) * t_min, 
						knots,
						np.ones(spline_order) * t_max))
    basis_matrix = make_b_spline_basis_arbitrary_knots(
      	spline_order, knots, dt, add_constant_basis, verbose)

    return basis_matrix

def make_b_spline_basis_arbitrary_knots(
        spline_order,
        knots,
        dt,
        add_constant_basis,
        verbose):
    """Constructs B-spline basis."""

    num_basis = len(knots) - spline_order - 1
    num_rows = int(np.round((knots[-1] - knots[0]) / dt))
    t = np.linspace(knots[0], knots[-1], num_rows)
    basis_matrix = np.zeros((len(t), num_basis))
    interpolate_token=[0, 0, spline_order]
    interpolate_token[0] = np.array(knots)

    for i in range(num_basis):
        basis_coefficients = [0] * num_basis
        basis_coefficients[i] = 1.0 
        interpolate_token[1] = basis_coefficients
        y = scipy.interpolate.splev(t, interpolate_token)
        basis_matrix[:, i] = y

    if add_constant_basis:
        basis_matrix = np.hstack((np.ones((len(t), 1)), basis_matrix))
    max_scale = np.max(basis_matrix,axis=0)
    basis_matrix = basis_matrix/max_scale
    if verbose:
        plt.figure()
        plt.plot(t, basis_matrix)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return basis_matrix

#%% Generate spike train
def generate_spike_train(lmbd, random_seed=None):
    """Generate one trial of spike train using firing rate lamdba.

    Args:
        lmbd: The firing rate.

    Returns:
        One spike train.
    """
    if random_seed:
        np.random.seed(random_seed)

    spike_train = np.zeros(len(lmbd))
    for t in range(len(lmbd)):
        num_spikes = np.random.poisson(lmbd[t])
        spike_train[t] = num_spikes
    return spike_train

#%% Calculating log likelihood
def spike_trains_neg_log_likelihood(log_lmbd, spike_trains, trial_wise=False, max_spike=None):
    """Calculates the log-likelihood of a spike train given log firing rate.

    When it calculates the log_likelihood funciton, it assumes that it is a
    function of lambda instead of spikes. So it drops out the terms that are not
    related to the lambda, which is the y! (spikes factorial) term.

    Args:
        log_lmbd: The format can be in two ways.
                timebins 1D array.
                (nt, ntrials) numpy array. Different trials have differnet intensity.
                        In this case, `spike_trains` and `log_lmbd` have matching rows.
        spike_trains: (nt, ntrials) numpy array.
    """
    # Having maximum spikes is just like binomial regression. 
    # Having inf maximum spikes is Poisson
    
    if max_spike is None:
        if spike_trains.ndim == 1:
            spike_trains = spike_trains[:, np.newaxis]
        nt, ntrial= spike_trains.shape
        # Default is Poisson
        if log_lmbd.ndim == 2:    # Trialwise intensity function.
            assert log_lmbd.shape==spike_trains.shape, "matrix doesn't match!"
            nll = - (spike_trains * log_lmbd)
            nll += np.exp(log_lmbd)
            if trial_wise:
                return nll.sum(axis=0)
            else:
                return nll.sum()
        elif log_lmbd.ndim == 1:    # Single intensity for all trials.
            nll = - spike_trains.sum(axis=1) @ log_lmbd
            nll += np.exp(log_lmbd).sum() * ntrial
            return nll
    else:
        # Binomial
        if spike_trains.ndim == 1:
            spike_trains = spike_trains[:, np.newaxis]
        success_fail = get_success_fail(spike_trains, max_spike=max_spike)
        link = get_link('logit')
        lmbd = link(log_lmbd)
        if log_lmbd.ndim == 2:    # Trialwise intensity function.
            nll = -lmbd*success_fail[:,:,0] - (1-lmbd)*success_fail[:,:,1]
            if trial_wise:
                return nll.sum(axis=0)
            else:
                return nll.sum()
        elif log_lmbd.ndim == 1:    # Single intensity for all trials.
            success_fail_sum = success_fail.sum(axis=1)
            nll = -lmbd*success_fail_sum[:,0] - (1-lmbd)*success_fail_sum[:,1]
            return nll.sum()


class poisson_regression_result():
    def __init__(self, params, bse, inv_hessian=None):
        self.params = params
        self.bse = bse
        self.inv_hessian = inv_hessian

#%% Core code for fitting Poisson GLM
def poisson_regression(
        Y,
        X,
        nt=350,
        L2_pen=1e-6,
        max_num_iterations=100, 
        tol=1e-8,
        no_penalty=[],
        offset=None, 
        no_penalty_term_penalty=0, 
        smoothing=0):
    """Fit Poisson GLM.

    The coefficients beta is fitted using Newton's method.
    Args:
        Y: (nt*ntrial, ) numpy vector
        X: (nt*ntrial, num_predictor) numpy array
    """
    Y = Y[:, np.newaxis]
    assert Y.shape[0] == X.shape[0], "Predictors X should match the shape of Y"
    num_predictor = X.shape[1]
    if offset is None:
        offset = np.zeros_like(Y)

    beta = np.zeros((num_predictor, 1))
    penalty_vec = np.ones((num_predictor, 1))
    if (X[:,0] == 1).all():
        # The first column is the constant baseline, set the constant to mean firing rate. 
        beta[0] = np.log(Y.sum()/len(Y))
        penalty_vec[0] = no_penalty_term_penalty
    for no_penalty_term in no_penalty:
        penalty_vec[no_penalty_term] = no_penalty_term_penalty
    penalty_matrix = np.diag(penalty_vec.squeeze())
    ### Second order difference matrix
    X_average = X.reshape(-1, nt, num_predictor).mean(axis=0)
    # X_average = X[:500,:]
    D = np.diag(np.ones(X_average.shape[0]-1), k=-1) + np.diag(-2*np.ones(X_average.shape[0]), k=0) \
        + np.diag(np.ones(X_average.shape[0]-1), k=1)
    D = D[1:-1, :]
    # print(X.shape, D.shape)
    smoothing_penalty_matrix = D@X_average
    log_lmbda_hat = (X @ beta) + offset

    nll = spike_trains_neg_log_likelihood(log_lmbda_hat, Y) + L2_pen * np.linalg.norm(beta*penalty_vec)**2\
            + smoothing * np.linalg.norm(smoothing_penalty_matrix@beta)**2
    nll_old = np.inf
    for iter_index in range(max_num_iterations):
        # Newton's method.
        # g: search direction
        mu = np.exp((X @ beta) + offset)
        grad = - (X.T @ Y) + (X.T @ mu) + 2*L2_pen * penalty_vec * beta \
            + 2*smoothing * smoothing_penalty_matrix.T @ smoothing_penalty_matrix @ beta
        hessian = (X.T) @ (mu * X) + 2*L2_pen * penalty_matrix \
            + 2*smoothing * smoothing_penalty_matrix.T @ smoothing_penalty_matrix
        g = np.linalg.pinv(hessian) @ grad
        lr = 1
        ALPHA = 0.4
        BETA = 0.2
        
        # Backtracking line search.
        while True:
            beta_tmp = beta - lr * g
            log_lmbd_tmp = (X @ beta_tmp) + offset
            nll_left = spike_trains_neg_log_likelihood(log_lmbd_tmp, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2\
                        + smoothing * np.linalg.norm(smoothing_penalty_matrix@beta_tmp)**2
            nll_right = nll - ALPHA * lr * grad.T @ g

            if (nll_left > nll_right or
                    np.isnan(nll_left) or
                    np.isnan(nll_right)):
                lr *= BETA
                # print(f"update learning_rate: {lr}")
            else:
                break
        if iter_index == max_num_iterations - 1:
            print('Warning: Reaches maximum number of iterations.')
            
        # Update beta, negtive log-likelihood.
        beta = beta_tmp
        nll = nll_left
        # print(iter_index, nll)
        # Check convergence.
        if abs(nll - nll_old) < tol:
            break
        nll_old = nll
    # print(nll)
    # print(spike_trains_neg_log_likelihood(log_lmbd_tmp, Y)/nll_left, L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2/nll_left,
    #       smoothing * np.linalg.norm(smoothing_penalty_matrix@beta_tmp)**2/nll_left)
    # Get standard error
    mu = np.exp((X @ beta) + offset)
    hessian = X.T @ (mu * X) + 2*L2_pen * penalty_matrix + 2*smoothing * smoothing_penalty_matrix.T @ smoothing_penalty_matrix
    inv_hessian = np.linalg.pinv(hessian)
    bse = np.sqrt(np.diag(inv_hessian))
    return poisson_regression_result(beta.squeeze(), bse, inv_hessian)

def relu(x):
    return np.maximum(0, x)

def sign(x):
    return (x>=0).astype(float)

def f_correct(t, a=0, intecept=0):
    return t + a*relu(t-intecept)**2

def poisson_regression_additional(
        Y,
        X,
        L2_pen=1e-6,
        max_num_iterations=100, 
        tol=1e-8,
        no_penalty=[],
        offset=None, 
        a=None,
        intecept=None):
    """Fit Poisson GLM.

    The coefficients beta is fitted using Newton's method.
    Args:
        Y: (nt*ntrial, ) numpy vector
        X: (nt*ntrial, num_predictor) numpy array
    """
    Y = Y[:, np.newaxis]
    assert Y.shape[0] == X.shape[0], "Predictors X should match the shape of Y"
    num_predictor = X.shape[1]
    if offset is None:
        offset = np.zeros_like(Y)

    beta = np.zeros((num_predictor, 1))
    penalty_vec = np.ones((num_predictor, 1))
    if (X[:,0] == 1).all():
        # The first column is the constant baseline, set the constant to mean firing rate. 
        beta[0] = np.log(Y.sum()/len(Y))
        penalty_vec[0] = 0
    for no_penalty_term in no_penalty:
        penalty_vec[no_penalty_term] = 0
    penalty_matrix = np.diag(penalty_vec.squeeze())
    
    """ Please see the photo for equations
    Args:
        intecept: intecept
        a: a
        t: raw_log_lmbda_hat
        z: log_lmbda_hat after correction
        expz: lmbda_hat
        J: loss/nll
    """
    if a is None:
        a = -0.15
    if intecept is None:
        intecept = -0.5
    
    t = (X @ beta) + offset
    z = f_correct(t, a, intecept)

    nll = spike_trains_neg_log_likelihood(z, Y) + L2_pen * np.linalg.norm(beta*penalty_vec)**2
    nll_old = np.inf
    
    for alternating_index in range(1):
        
        for iter_index in range(max_num_iterations):
            ### First, update beta
            # Newton's method.
            # g: search direction
            t = (X @ beta) + offset
            z = f_correct(t, a, intecept)
            expz = np.exp(z)
            
            pJ_pz = -Y + expz
            after_relu = relu(t-intecept)
            after_sign = sign(t-intecept)
            pz_pt = 1 + after_relu*2*a
            pJ_pbeta = X.T @ (pJ_pz*pz_pt)
            term1 = pJ_pz*2*a*after_sign
            term2 = pz_pt**2*expz
            ppJ_ppbeta = X.T @ ( (term1+term2) *X)

            grad = pJ_pbeta + 2*L2_pen * penalty_vec * beta
            hessian = ppJ_ppbeta + 2*L2_pen * penalty_matrix
            g = np.linalg.pinv(hessian) @ grad 
            lr = 1
            ALPHA = 0.4
            BETA = 0.2
            
            # Backtracking line search.
            while True:
                beta_tmp = beta - lr * g
                t = (X @ beta_tmp) + offset
                z = f_correct(t, a, intecept)
                nll_left = spike_trains_neg_log_likelihood(z, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2
                z_right = f_correct( (X @ (beta - ALPHA * lr * g)) + offset , a, intecept)
                nll_right = spike_trains_neg_log_likelihood(z_right, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2

                if (nll_left > nll_right or
                        np.isnan(nll_left) or
                        np.isnan(nll_right)):
                    lr *= BETA
                    # print(f"update learning_rate: {lr}")
                else:
                    break
            if iter_index == max_num_iterations - 1:
                print('Warning: Reaches maximum number of iterations.')
                
            # Update beta, negtive log-likelihood.
            beta = beta_tmp
            nll = nll_left
            # print(iter_index, nll)
            # Check convergence.
            if abs(nll - nll_old) < tol:
                break
            nll_old = nll
        # print(f"after beta updating: {nll:.5f}")
        
        
    #     for iter_index in range(max_num_iterations):
    #         ### Then, update a and intecept
    #         a_intecept = np.array([[a], [intecept]])
    #         ppJ_ppa = (pJ_pz.T@after_relu**2).item()
    #         pJ_pa = a*ppJ_ppa
    #         # ppJ_pa_pintecept = -(pJ_pz.T@after_relu*2*a).item()
    #         pJ_pintecept = -(pJ_pz.T@after_relu*2*a).item()
    #         # ppJ_ppintecept = (pJ_pz.T@after_sign*2*a).item()
    #         grad = np.array([[pJ_pa], [pJ_pintecept]])
    #         # hessian = np.array([[ppJ_ppa, ppJ_pa_pintecept], [ppJ_pa_pintecept, ppJ_ppintecept]])
    #         # g = np.linalg.pinv(hessian) @ grad
    #     #     print(a, intecept)
    #     #     print(grad)
    #     #     print(hessian)
    #     #     print(g)
    #         g = - grad
            
    #         lr = 1e-2
    #         ALPHA = 0.4
    #         BETA = 0.2
    #         t = (X @ beta) + offset
            
    #         while True:
    #             a_intecept_tmp = a_intecept - lr * g
    #             z = f_correct(t, a_intecept_tmp[0, 0], a_intecept_tmp[1, 0])
    #             nll_left = spike_trains_neg_log_likelihood(z, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2
                
    #             a_intecept_tmp_right = a_intecept - ALPHA * lr * g
    #             z_right = f_correct(t, a_intecept_tmp_right[0, 0], a_intecept_tmp_right[1, 0])
    #             nll_right = spike_trains_neg_log_likelihood(z_right, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2

    #             if (nll_left > nll_right or
    #                     np.isnan(nll_left) or
    #                     np.isnan(nll_right)):
    #                 lr *= BETA
    #                 print(f"update learning_rate: {lr}")
    #             else:
    #                 break
    #         if iter_index == max_num_iterations - 1:
    #             print('Warning: Reaches maximum number of iterations.')
                
    #         # Update beta, negtive log-likelihood.
    #         a, intecept = a_intecept[0, 0], a_intecept[1, 0]
    #         nll = nll_left
    #         # print(iter_index, nll)
    #         # Check convergence.
    #         if abs(nll - nll_old) < tol:
    #             break
    #         nll_old = nll
    #     print(f"after a and intecept updating: {nll:.5f}")
    
    # print(pJ_pa, g[0,0])
    # print(pJ_pa)
    # for a in np.arange(-0.1-1e-3, -0.1+2*1e-3, 1e-3):
    #     z = f_correct(t, a, intecept)
    #     print(spike_trains_neg_log_likelihood(z, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2)
                
    # print(nll)
    # Get standard error
    # mu = np.exp((X @ beta) + offset)
    # hessian = X.T @ (mu * X) + 2*L2_pen * penalty_matrix
    t = (X @ beta) + offset
    z = f_correct(t, a, intecept)
    expz = np.exp(z)
    pJ_pz = -Y + expz
    pz_pt = 1 + relu(t-intecept)*2*a
    pJ_pbeta = X.T @ (pJ_pz*pz_pt)
    term1 = pJ_pz*2*a*sign(t-intecept)
    term2 = 1 + relu(t-intecept)*2*a*expz
    ppJ_ppbeta = X.T @ ( (term1+term2) *X)
    hessian = ppJ_ppbeta + 2*L2_pen * penalty_matrix
    
    inv_hessian = np.linalg.pinv(hessian)
    bse = np.sqrt(np.diag(inv_hessian))
    return poisson_regression_result(beta.squeeze(), bse, inv_hessian), a, intecept

def poisson_regression_pytorch(
        Y,
        X,
        L2_pen=1e-6,
        max_num_iterations=100, 
        tol=1e-8,
        no_penalty=[],
        offset=None):
    """Fit Poisson GLM.

    The coefficients beta is fitted using Newton's method.
    Args:
        Y: (nt*ntrial, ) numpy vector
        X: (nt*ntrial, num_predictor) numpy array
    """
    USE_ADDITIONAL_LAYER = True
    INITIAL_VALUES_FROM_LINE_SEARCH = False

    # Run line search to get initial values
    if INITIAL_VALUES_FROM_LINE_SEARCH:
        line_search_result = poisson_regression(
            Y,
            X,
            L2_pen=L2_pen,
            max_num_iterations=max_num_iterations, 
            tol=tol,
            no_penalty=no_penalty,
            offset=offset)
        # return line_search_result
        initial_beta = line_search_result.params[:, np.newaxis]
        beta_ts = torch.tensor(initial_beta, requires_grad=True, dtype = torch.float64)
    else:
        beta_ts = torch.zeros((X.shape[1], 1), requires_grad=True, dtype = torch.float64)
        
    # Initialization and transform numpy to tensor    
    Y = Y[:, np.newaxis]
    num_predictor = X.shape[1]
    Y_ts = torch.tensor(Y)
    X_ts = torch.tensor(X)
    if offset is None:
        offset = np.zeros_like(Y)
    offset_ts = torch.tensor(offset)
    
    penalty_vec_ts = torch.ones((num_predictor, 1))
    if (X[:,0] == 1).all():
        # The first column is the constant baseline, set the constant to mean firing rate. 
        beta_ts[0] = np.log(Y.sum()/len(Y))
        penalty_vec_ts[0] = 0
    for no_penalty_term in no_penalty:
        penalty_vec_ts[no_penalty_term] = 0
    penalty_matrix = np.diag(penalty_vec_ts.squeeze())
    penalty_matrix_ts = torch.tensor(penalty_matrix)
    
    # Substitude RHS with:    RHS + a*( ReLu(RHS-b) )^2
    if USE_ADDITIONAL_LAYER:
        intecept = torch.zeros((1, 1), requires_grad=True, dtype = torch.float64)
        coef = torch.zeros((1, 1), requires_grad=True, dtype = torch.float64)
        relu = torch.nn.ReLU()
    
    # Optimization
    
    if not USE_ADDITIONAL_LAYER:
        lr = 1e-3
        # lr = 1e-6
        # optimizer = torch.optim.SGD([beta_ts], lr=lr,momentum=0.0)
        optimizer = torch.optim.Adam([beta_ts], lr=lr)
    else:
        lr = 1e-3
        optimizer = torch.optim.Adam([
                                    {'params': [beta_ts]},
                                    {'params': [intecept], 'lr': 1e-5},
                                    {'params': [coef], 'lr': 1e-6}
                                    ], lr=lr)
    nepoch = 5000
    for epoch in range(nepoch):
        # Get current loss and apply gradient descent
        optimizer.zero_grad()
        # Forward pass and compute loss
        if not USE_ADDITIONAL_LAYER:
            log_lmbda_hat = (X_ts @ beta_ts) + offset_ts
            loss = torch.sum( - (Y_ts * log_lmbda_hat) + torch.exp(log_lmbda_hat) ) \
                    + L2_pen * torch.linalg.norm(beta_ts*penalty_vec_ts)**2
        else:
            raw_log_lmbda_hat = (X_ts @ beta_ts) + offset_ts
            log_lmbda_hat = raw_log_lmbda_hat + coef*(relu(raw_log_lmbda_hat-intecept))**2
            loss = torch.sum( - (Y_ts * log_lmbda_hat) + torch.exp(log_lmbda_hat) ) \
                    + L2_pen * torch.linalg.norm(beta_ts*penalty_vec_ts)**2
        last_loss = loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # print(f"grad {torch.linalg.norm(beta_ts.grad).item():.5f}")
        # print(f"loss: {loss.item():.2f}")
        
        # Evaluation at a new point
        if not USE_ADDITIONAL_LAYER:
            log_lmbda_hat = (X_ts @ beta_ts) + offset_ts
            loss = torch.sum( - (Y_ts * log_lmbda_hat) + torch.exp(log_lmbda_hat) ) \
                    + L2_pen * torch.linalg.norm(beta_ts*penalty_vec_ts)**2
        else:
            raw_log_lmbda_hat = (X_ts @ beta_ts) + offset_ts
            log_lmbda_hat = raw_log_lmbda_hat + coef*(relu(raw_log_lmbda_hat-intecept))**2
            loss = torch.sum( - (Y_ts * log_lmbda_hat) + torch.exp(log_lmbda_hat) ) \
                    + L2_pen * torch.linalg.norm(beta_ts*penalty_vec_ts)**2
        current_loss = loss.item()
        
        if last_loss - current_loss < -1e-5:
            print("loss increased!")
            lr = lr/2
            for g in optimizer.param_groups:
                g['lr'] *= 1/2
        if last_loss - current_loss <=1e-5:
            # break
            pass
        else:
            last_loss = current_loss
    if epoch>=nepoch-2:
        print("ran for full epoch and still not converge!")
    else:
        print(f"Used {epoch} steps to converge. ")

    print(f"loss: {current_loss:.3f}")
    if USE_ADDITIONAL_LAYER:
        print(f"intecept:{intecept.item():.7f}; coef:{coef.item():.7f}")
        
    # Get final results (with standard error)
    beta = beta_ts.detach().numpy()
    mu = np.exp((X @ beta) + offset)
    hessian = X.T @ (mu * X) + 2*L2_pen * penalty_matrix
    inv_hessian = np.linalg.pinv(hessian)
    bse = np.sqrt(np.diag(inv_hessian))
    return poisson_regression_result(beta.squeeze(), bse, inv_hessian)

#%% Excursion test
def get_excursion_statistic_from_dict(record, method='default', time_range=None, calibrate_threshold=2.58):
    statistics_dict = {}
    for key, value in tqdm(record.items(), desc='Processing records'):
        if key[0]>=0 and key[1]>=0:
            statistics_dict[key] = get_excursion_statistic_multi_trial(
                value, method=method, time_range=time_range, calibrate_threshold=calibrate_threshold
            )
    return statistics_dict

def get_excursion_statistic_multi_trial(null_functions, method='default', time_range=None, calibrate_threshold=2.58):
    statistics_list = []
    # print(len(value))
    for functions in null_functions:
        statistics, ROI = get_excursion_statistic_single_trial(
            functions, method=method, time_range=time_range, calibrate_threshold=calibrate_threshold
        )
        statistics_list.append(statistics)
    return statistics_list

def get_excursion_statistic_single_trial(functions, method='default', time_range=None, calibrate_threshold=2.58):
    # assert len(functions)==4, len(functions)
    assert method in ['default', 'max', 'calibrate'], "Only support \'default\', \'max\', and \'calibrate\' "
    func1, func2, std1, std2 = functions
    if time_range is None:
        time_range = [0, len(func1)]
    func = np.abs(func1[time_range[0]:time_range[1]] - func2[time_range[0]:time_range[1]])
    
    # Define "diff" function and "threshold"
    if method in ['default']:
        std = None
        threshold = func.max()/2
    elif method in ['max', 'calibrate']:
        std = np.sqrt(std1**2 + std2**2)
        std[std<=std.mean()] = std.mean()
        func /= std
        threshold = calibrate_threshold if method=='calibrate' else 0
        
    # Get ROIs
    idx = np.where(func >= threshold)[0]
    ROI_list = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
    
    # Get excursion test statistics
    stats_list = []
    for i, ROI in enumerate(ROI_list):
        if method in ['default']:
            stats_list.append( np.sum( func[ROI]) )
        elif method in ['calibrate']:
            stats_list.append( np.sum( func[ROI]-calibrate_threshold) )
        elif method in ['max']:
            stats_list.append( np.max( func[ROI]) )

    max_index = stats_list.index(max(stats_list))
    ROI_list.insert(0, ROI_list.pop(max_index))
    return np.max(stats_list), ROI_list

# def get_excursion_statistic(function1, function2, time_range=None, return_filter=False, std1=None, std2=None):
#     assert ((std1 is not None) and (std2 is not None)) or ((std1 is None) and (std2 is None)), "std1 and std2 all being None means not calibrating."
#     if time_range is None:
#         time_range = [0, len(function1)]
#     if ((std1 is None) and (std2 is None)):
#         std = None
#     else:
#         std = np.sqrt(std1**2 + std2**2)
#         std[std<=std.mean()] = std.mean()
#     func = np.abs(function1[time_range[0]:time_range[1]] - function2[time_range[0]:time_range[1]])
#     ROI = get_ROI(func, std=std)
#     test_statistic = get_excursion_test(func, ROI, std=std)
#     if return_filter is False:
#         return [single_ROI+time_range[0] for single_ROI in ROI], test_statistic
#     else:
#         return [single_ROI+time_range[0] for single_ROI in ROI], test_statistic, [function1, function2, std1, std2]

# def get_excursion_test(func, ROI_list, std=None):
#     stats_list = []
#     if std is None:
#         func_calibrate = func
#     else:
#         func_calibrate = func/std
#     for i, ROI in enumerate(ROI_list):
#         stats_list.append( np.sum( func_calibrate[ROI]) )
#         # stats_list.append( np.sum( func_calibrate[ROI]-2.58) ) # Calibrate
#         # stats_list.append( np.max( func_calibrate[ROI]-0) ) # Max
#     return [np.max(stats_list)]

# def get_ROI(func, std=None):
#     if std is None:
#         threshold = func.max()/2
#         func_calibrate = func
#     else:
#         func_calibrate = func/std
#         # threshold = 2.58 # Calibrate
#         # threshold = 0 # Max
#     idx = np.where(func_calibrate >= threshold)[0]
#     return np.split(idx, np.where(np.diff(idx) != 1)[0]+1)

def merge_dict(d1, d2):
    # d1 is the mother, d2 is the one to add to d1
    d = {}
    for k in d1.keys():
        d[k] = d1[k] + d2[k]
    return d

def merge_dict_multi(d_list):
    merged_d = {}
    for k in d_list[0].keys():
        merged_d[k] = []
        for d in d_list:
            merged_d[k] += d[k]
    return merged_d

def get_statistics_null_excursion(V1, membership, condition_ids, fix_peak_time):
    # The following hyperparameters turned out to be the best
    # num_f_refractory = 4
    # max_iter = 5
    # tau = 15
    # coupling_filter_params = {'peaks_max':50, 'num':6, 'nonlinear':0.3}
    # num_basis_baseline = 30
    # penalty = 3e-1
    
    # num_f_refractory = 4
    # max_iter = 5
    # tau = 15
    # coupling_filter_params = {'peaks_max':26, 'num':3, 'nonlinear':0.9}
    # num_basis_baseline = 20
    # penalty = 5e-1

    num_f_refractory = 4
    max_iter = 5
    tau = 15
    coupling_filter_params = {'peaks_max':20.2, 'num':3, 'nonlinear':0.5}
    num_basis_baseline = 20
    penalty = 5e-1

    # num_f_refractory = 4
    # max_iter = 5
    # tau = 15
    # coupling_filter_params = {'peaks_max':30, 'num':4, 'nonlinear':0.5}
    # num_basis_baseline = 20
    # penalty = 3e-1

    ################ No need to change below
    fake_running_trial_index = copy.deepcopy(V1.running_trial_index)
    np.random.shuffle(fake_running_trial_index)  # don't need to assign shuffled list, it's changed automatically. 
    fake_stationary_trial_index = np.logical_not(fake_running_trial_index)
    
    probe_list = V1.selected_probes
    running_filter = {}
    stationary_filter = {}
    running_output = {}
    stationary_output = {}

    record_filter = {}
    record_output = {}

    for i, target_probe in enumerate(probe_list):
        select_trials = fake_running_trial_index
        model = PP_GLM(dataset=V1, 
                        select_trials=select_trials, 
                        membership=membership, 
                        condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j], apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('refractory_additive', target_probe, tau=tau, num=num_f_refractory, apply_no_penalty=True)
        model.add_effect('trial_coef')
        if fix_peak_time is None:
            model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
        else:
            model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty, 
                                            fix_shifts=fix_peak_time[i][select_trials])
            
        filter_list = model.get_filter(ci=True)
        for j in range(len(model.basis_list)):
            running_filter[i,j-1] = filter_list[j]
        output_list = model.get_filter_output(ci=True)
        for j in range(len(model.basis_list)):
            running_output[i,j-1] = output_list[j]
        
        select_trials = fake_stationary_trial_index
        model = PP_GLM(dataset=V1, 
                        select_trials=select_trials, 
                        membership=membership, 
                        condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j],apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('refractory_additive', target_probe, tau=tau, num=num_f_refractory, apply_no_penalty=True)
        model.add_effect('trial_coef')
        if fix_peak_time is None:
            model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
        else:
            model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty, 
                                            fix_shifts=fix_peak_time[i][select_trials])
        
        filter_list = model.get_filter(ci=True)
        for j in range(len(model.basis_list)):
            stationary_filter[i,j-1] = filter_list[j]
        output_list = model.get_filter_output(ci=True)
        for j in range(len(model.basis_list)):
            stationary_output[i,j-1] = output_list[j]
        
        # for effect filter
        filter_index = i,-1
        function1 = np.exp( running_filter[filter_index][0] )
        function2 = np.exp( stationary_filter[filter_index][0] )
        std1 = function1*np.exp(running_filter[filter_index][1]**2/2)*np.sqrt(np.exp(running_filter[filter_index][1]**2)-1)
        std2 = function1*np.exp(stationary_filter[filter_index][1]**2/2)*np.sqrt(np.exp(stationary_filter[filter_index][1]**2)-1)
        record_filter[filter_index] = [function1, function2, std1, std2]
        for j, input_probe in enumerate(probe_list):
            filter_index = i,j
            function1 = running_filter[filter_index][0]
            function2 = stationary_filter[filter_index][0]
            std1 = running_filter[filter_index][1]
            std2 = stationary_filter[filter_index][1]
            record_filter[filter_index] = [(function1, function2, std1, std2)]
        
        # for effect output
        filter_index = i,-1
        function1 = np.exp( running_output[filter_index][0] )
        function2 = np.exp( stationary_output[filter_index][0] )
        std1 = function1*np.exp(running_output[filter_index][1]**2/2)*np.sqrt(np.exp(running_output[filter_index][1]**2)-1)
        std2 = function1*np.exp(stationary_output[filter_index][1]**2/2)*np.sqrt(np.exp(stationary_output[filter_index][1]**2)-1)
        record_output[filter_index] = [function1, function2, std1, std2]
        for j, input_probe in enumerate(probe_list):
            filter_index = i,j
            function1 = running_output[filter_index][0]
            function2 = stationary_output[filter_index][0]
            std1 = running_output[filter_index][1]
            std2 = stationary_output[filter_index][1]
            record_output[filter_index] = [(function1, function2, std1, std2)]
    
    return record_filter, record_output

### Multiprocess version of null distribution
def get_statistics_null_mp(n_null, V1, membership, condition_ids, fix_peak_time):
    """Get the distribution of test statistics (excursion test) under the null hypothesis. 
    Null hypothesis is that trial-wise running state doesn't affect neural response. So null 
    statistics is sample from random shuffling the running state of each trial. 
    To avoid KiB Swap running of out memory issue, parallel processing has a limitation of total
    tasks at the queue. The number of tasks is PARALLEL_BATCH_SIZE. 

    Args:
        n_null (int): number of samples of the null distributin
        V1 (DataLoader.Allen_dataset): the object containing all data and experimental information
        membership (pandas frame): IPRF result
        condition_ids (list): IPRF result
        probe_list (list): a list like ['probeA', 'probeC']
        num_basis_baseline (int): number of B-spline basis for inhomogeneous baseline

    Returns:
        dict: a dict whose key-value pair denote the null statistics distributino samples of a 
        certain filter. 
    """

    import multiprocessing
    import os
    # PROCESSES = os.cpu_count()-2
    PROCESSES = 5
    PARALLEL_BATCH_SIZE = 5
    nbatch = int(np.ceil(n_null/PARALLEL_BATCH_SIZE))
    print(f"Starting multiprocessing on {sys.platform}. \nCores={PROCESSES}. \nBatch size={PARALLEL_BATCH_SIZE}")
    if sys.platform == 'linux':
        with tqdm(total=n_null) as pbar:
            for ibatch in range(nbatch):
                with multiprocessing.get_context('spawn').Pool(processes = PROCESSES) as pool:               
                    results = [pool.apply_async(get_statistics_null_excursion, (V1, membership, condition_ids, fix_peak_time)) 
                            for i_null in np.arange(PARALLEL_BATCH_SIZE)]
                    pool.close()
                    if ibatch == 0: 
                        # The first batch the first return result will be the very first "statistics_null"
                        record_filter_null, record_output_null = results[0].get()
                        pbar.update(1)
                        for result in results[1:]:
                            record_filter_null_new, record_output_null_new = result.get()
                            record_filter_null = merge_dict(record_filter_null, record_filter_null_new)
                            record_output_null = merge_dict(record_output_null, record_output_null_new)
                            pbar.update(1)
                    else:
                        for result in results:
                            record_filter_null_new, record_output_null_new = result.get()
                            record_filter_null = merge_dict(record_filter_null, record_filter_null_new)
                            record_output_null = merge_dict(record_output_null, record_output_null_new)
                            pbar.update(1)
    else:
        raise ValueError("Only on Linux at the moment. ")
    return record_filter_null, record_output_null



def get_statistics_null_parametric_bootstrap(V1, membership, condition_ids, probe_list, num_basis_baseline):
    pass
    # Fit six model
    # for f in all filters:
        # set f to all 0s
        # for i_bt in range(n_bt):
            # simulated a whole dataset
            # regress and get an estimate of f, f hat
            # calculate and record test statistics
        # get p-value of that filter f
        
    # To-do: simulation; 
    #        test statistics try: sum(abs(f)); excursion on abs(f); KL for positive

def plot_function_with_excursion(V1, stationary_function, running_function, statistics_function=None, 
                               statistics_null_function=None, ROI_function=None, inference=True, 
                               plot_baseline=True, plot_self=False, function_amp=0.15, dpi=300, 
                               colors=['b', 'r'], labels=['Stationary', 'Running'], p_th=0.05,
                               plot_null_distribution=True,):
    transfer_ij = {-1:-1, 4:0, 5:1, 0:2, 1:3, 2:4, 3:5}
    probe_list = V1.selected_probes
    name_list = ['V1', 'LM', 'AL', 'RL', 'AM', 'PM']
    function_length = stationary_function[1,2][0].shape[0]
    trial_length = V1.nt

    if inference:
        pvalue_toplot = {}
        for i in range(len(probe_list)):
            for j in range(0, len(probe_list)):
                function_index = i,j
                pvalue_toplot[function_index] = 1- \
                    np.sum( statistics_function[function_index]>statistics_null_function[function_index]) \
                        /len(statistics_null_function[function_index])

    sns.reset_orig()
    
    plt.subplots(figsize=(6.9,4.6), dpi=dpi)
    utils.use_pdf_plot(**{'axes.linewidth':0.5, 'xtick.labelsize':5, 'ytick.labelsize':5})
    BIGGER_SIZE = 7
    MEDIUM_SIZE = 6
    SMALL_SIZE = 5
    BIGGER_LW = 1
    SMALL_LW = 0.75
    
    for i in range(len(probe_list)):
        for j in range(-1, len(probe_list)):
            i_plot = transfer_ij[i]
            j_plot = transfer_ij[j]
            if i==j and (not plot_self):
                continue
            ax = plt.subplot(6, 7, i*7+j+1+1, frameon=True)
            function_index = i_plot,j_plot
            y, ci = stationary_function[function_index]
            x = np.arange(y.shape[0])
            color = colors[0]
            label = labels[0]

            if j == -1:
                if plot_baseline:
                    plt.plot(x, np.exp(y),label=label, color=color, lw=SMALL_LW)
                    plt.fill_between(x, np.exp((y-2*ci)), np.exp((y+2*ci)), color=color, alpha=.3)
            else:
                # Correct for different number of neurons in different areas
                plt.plot(x, y,label=label, color=color, lw=SMALL_LW)
                plt.fill_between(x, (y-2*ci), (y+2*ci), color=color, alpha=.3)
                
            y, ci = running_function[function_index]
            x = np.arange(y.shape[0])
            color = colors[1]
            label = labels[1]

            if j == -1:
                if plot_baseline:
                    plt.plot(x, np.exp(y),label=label, color=color, lw=SMALL_LW)
                    plt.fill_between(x, np.exp((y-2*ci)), np.exp((y+2*ci)), color=color, alpha=.3)
                    plt.xticks([0, trial_length/2, trial_length])
                    plt.xlim([0, trial_length])
            else:
                plt.plot(x, y,label=label, color=color, lw=SMALL_LW)
                plt.fill_between(x, (y-2*ci), (y+2*ci), color=color, alpha=.3)
                # plt.yticks([-filter_amp, 0, filter_amp])
                plt.yticks([0])
                # if i!=j and i_plot!=0:
                plt.ylim([-function_amp, function_amp])
                plt.yticks(color='w')
                plt.xticks([0, function_length])
                plt.xlim([0, function_length])

            # Add xaxis to know where is above zero and where is below zero. 
            if j != -1:
                plt.axhline(0, ls='-', color='grey', lw=SMALL_LW, alpha=0.5)
            # Only display "time (ms)" at the bottom row. 
            if i == 5:
                plt.xlabel('time (ms)', fontsize=SMALL_SIZE)
            else:
                plt.xticks(color='w')
            # Show amplitute for the most left panels of coupling filters. 
            if j==0 or (j==1 and i==0):
                plt.yticks(color='k')
                plt.text(-0.2, 0.95, f'{function_amp}', fontsize=SMALL_SIZE, transform=ax.transAxes)

            # Add to {area} label
            if j==-1:
                plt.text(-0.8, 0.5, f'To {name_list[i]}', fontsize=MEDIUM_SIZE, transform=ax.transAxes)
            # Add from {area} or "inhomo" label
            if i==0:
                if j==-1:
                    plt.text(-0.15, 1.2, f'Inhomo baseline', fontsize=MEDIUM_SIZE, transform=ax.transAxes)
                else:
                    plt.text(0.2, 1.2, f'From {name_list[j]}', fontsize=MEDIUM_SIZE, transform=ax.transAxes)
            # Mannually add from V1 label
            if i==0 and j==-1:
                ax = plt.subplot(6, 7, 1, frameon=True)
                plt.text(1.4, 1.2, f'From V1', fontsize=MEDIUM_SIZE, transform=ax.transAxes)
                ax = plt.subplot(6, 7, i*7+j+1+1, frameon=True)

            # plt.grid()
            
            if inference and j!=-1 and plot_null_distribution:
                ins = ax.inset_axes([0.7,0.55,0.3,0.3])
                sns.kdeplot(statistics_null_function[function_index], linewidth=BIGGER_LW, ax=ins, fill=True)
                ins.axvline(statistics_function[function_index], linewidth=SMALL_LW, color='r')
                ins.set_xticks([])
                ins.set_yticks([])
                ins.set_ylabel('')

            if inference:
                if j!=-1 and pvalue_toplot[function_index]<=p_th:
                    # change yellow to ROI
                    # stats_list = []
                    # for ii, ROI in enumerate(ROI_function[function_index]):
                    #     function1 = stationaryfunction[function_index][0]
                    #     function2 = running_function[function_index][0]
                    #     diff = np.abs(function1 - function2)
                    #     stats_list.append( np.sum(diff[ROI]) ) 
                    # temp = ROI_filter[function_index][np.argmax(stats_list)]
                    temp = ROI_function[function_index][0]
                    x = np.array([temp.min(), temp.max()])
                    plt.fill_between(x, 
                                     np.array([-function_amp,-function_amp]), 
                                     np.array([function_amp,function_amp]), 
                                     color='yellow', alpha=.5)
        #             ax.patch.set_alpha(0.3)
        #             ax.set_facecolor('yellow')
                    if pvalue_toplot[function_index]>0:
                        if pvalue_toplot[function_index]>=0.01:
                            plt.text(0.6*function_length, 0.75*function_amp, 
                                    f'p={pvalue_toplot[function_index]:.2f}', 
                                    fontsize=SMALL_SIZE)
                        else:
                            plt.text(0.5*function_length, 0.75*function_amp, 
                                    'p='+format_scientific_one_digit_exponent(pvalue_toplot[function_index]), 
                                    fontsize=SMALL_SIZE)
                    else:
                        plt.text(0.5*function_length, 0.75*function_amp, 
                                 'p<'+format_scientific_one_digit_exponent(1/len(statistics_null_function[function_index])), 
                                 fontsize=SMALL_SIZE)
    # plt.tight_layout()

def format_scientific_one_digit_exponent(number):
    """ Formats a number in scientific notation with only one digit after 'e' """
    formatted_number = f"{number:.1e}"
    base, exponent = formatted_number.split('e')
    # Removing leading '+' or '0' from the exponent part
    formatted_exponent = exponent.replace('+0', '+').replace('-0', '-')
    return f"{base}e{formatted_exponent}"

def corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            res_j = C[:, j]
            res_i = C[:, i]
            corr = scipy.stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = scipy.stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr


### 
def getI_real_data(features, dt, nt, npadding, log_fr):
    padding = int(dt*npadding)
    time_line = np.arange(-npadding, nt)*dt
    baseline = log_fr[0]
    log_fr = np.concatenate((np.ones(padding)*baseline,log_fr))
    old_time_line = np.arange(-padding, -padding+len(log_fr))
    stimulus_pre = np.interp(time_line, old_time_line, log_fr)
    
    stimulus = np.zeros((len(time_line), features["bump_center"][0].shape[0]))
    for itrial in range(features["bump_center"][0].shape[0]):
        stimulus[:,itrial] =  linear_time_warping_single(time_line, stimulus_pre, 
            [0, features["bump_pre_center"][0], 150], [0, features["bump_center"][0][itrial], 150])
        stimulus[:,itrial] =  linear_time_warping_single(time_line, stimulus[:,itrial], 
            [150, features["bump_pre_center"][1], 350], [150, features["bump_center"][1][itrial], 350])
    stimulus = np.exp(stimulus)
    I_max = features["bump_amp"]
    I_min = features["baseline"]
    stimulus = (stimulus-stimulus.min())/(stimulus.max()-stimulus.min())*(I_max-I_min) + I_min
    features["I"] = stimulus
    features["I_pre"] = stimulus_pre


def get_bump_centers(params, ntrial):
    if params["external_input_type"] in [
        "two_peaks_with_varying_baseline_slope", "two_peaks_with_varying_gain"
        ]:
        std1, corr1, std2, corr2 = (0, 0, 0, 0)
        use_two_modes = False
    elif params["external_input_type"] in[
        "two_peaks_with_varying_timing", "two_peaks_with_all_varying",
        "two_peaks_with_varying_timing_baseline_slope"
        ]:
        std1, corr1, std2, corr2 = (
            params["std1"], params["corr1"], params["std2"], params["corr2"]
        )
        use_two_modes = params["use_two_modes"]
    else:
        raise ValueError(f"Invalid external input type: {params['external_input_type']}")
    
    trial_info = None
    if use_two_modes:
        # First bump - two modes
        bump_center_mean1a = [40, 60]  # First mode
        bump_center_mean1b = [60, 80]  # Second mode
        bump_center_cov1 = np.array([[std1**2, corr1*std1**2], [corr1*std1**2, std1**2]])
        
        # Second bump - two modes
        bump_center_mean2a = [220, 250]  # First mode
        bump_center_mean2b = [250, 270]  # Second mode
        bump_center_cov2 = np.array([[std2**2, corr2*std2**2], [corr2*std2**2, std2**2]])
        
        # Randomly select which mode to use for each trial
        mode_selector = np.random.random(ntrial) < 0.5
        
        # Generate from appropriate mode for each trial
        bump_centers1 = np.zeros((ntrial, 2))
        bump_centers2 = np.zeros((ntrial, 2))
        
        bump_centers1[mode_selector] = np.random.multivariate_normal(
            bump_center_mean1a, bump_center_cov1, size=np.sum(mode_selector)
        )
        bump_centers1[~mode_selector] = np.random.multivariate_normal(
            bump_center_mean1b, bump_center_cov1, size=np.sum(~mode_selector)
        )
        
        bump_centers2[mode_selector] = np.random.multivariate_normal(
            bump_center_mean2a, bump_center_cov2, size=np.sum(mode_selector)
        )
        bump_centers2[~mode_selector] = np.random.multivariate_normal(
            bump_center_mean2b, bump_center_cov2, size=np.sum(~mode_selector)
        )
        trial_info = mode_selector
    else:
        bump_center_mean1 = [60, 65]
        bump_center_cov1 = np.array([[std1**2, corr1*std1**2], [corr1*std1**2, std1**2]])
        bump_center_mean2 = [260, 265]
        bump_center_cov2 = np.array([[std2**2, corr2*std2**2], [corr2*std2**2, std2**2]])
        bump_centers1 = np.random.multivariate_normal(bump_center_mean1, bump_center_cov1, size=ntrial)
        bump_centers2 = np.random.multivariate_normal(bump_center_mean2, bump_center_cov2, size=ntrial)

    # Apply bounds to both cases
    bump_centers1[bump_centers1>=130] = 130
    bump_centers1[bump_centers1<=10] = 10
    bump_centers2[bump_centers2>=330] = 330
    bump_centers2[bump_centers2<=170] = 170

    return bump_centers1, bump_centers2, trial_info


def EIF_simulator(
        ntrial, 
        nneuron, 
        conn, 
        params = {},
        nneuron_coupling=None,
        shared_background=0.0,
        no_stimulus=False,
        return_I_ext=False,
        return_trial_info=False,
        return_I_syn=False,
    ):

    with open('EIF_params.pickle', 'rb') as handle:
        EIF_params = pickle.load(handle)

    bin_size = 2 # final spikes_rcd bin size in ms
    dt = 0.2 # ms
    ndt = int(1/dt) # number of simulation time bins per ms
    padding = 100 # ms
    nt = (500)*ndt # total number of simulation time bins
    npadding = padding*ndt # number of simulation time bins for padding
    nt_tot = nt + npadding # total number of simulation time bins including padding
    noise_amp = 1.0
    nneuron_part = int(nneuron/2)
    if nneuron_coupling is None:
        nneuron_coupling = nneuron_part

    ### Getting the connectivity matrix
    J_cross = np.zeros((nneuron, nneuron)) # From row i to column j
    J_recurrent = np.zeros((nneuron, nneuron)) # From row i to column j
    # J[0:nneuron_part,nneuron_part:] = conn
    J_cross[0:nneuron_coupling,nneuron_part:nneuron_part+nneuron_coupling] = (
        np.random.lognormal(mean=np.log(conn), sigma=0.4, size=(nneuron_coupling, nneuron_coupling)) 
        if conn!=0 else 0.0
    )
    J_recurrent[0:nneuron_part,0:nneuron_part] = (
        np.random.lognormal(mean=np.log(0.001), sigma=0.4, size=(nneuron_part, nneuron_part)) 
    )
    J_recurrent[nneuron_part:,nneuron_part:] = (
        np.random.lognormal(mean=np.log(0.001), sigma=0.4, size=(nneuron_part, nneuron_part)) 
    )

    ### Getting the bump centers and trial info
    bump_centers1, bump_centers2, trial_info_bump = get_bump_centers(params, ntrial)
    baseline = 0.0
    bump_amp = 0.25

    ### Getting the external input after the bump centers are determined
    source = {"bump_pre_center": [EIF_params["V1_pre_center_p1"],
                                EIF_params["V1_pre_center_p2"]], 
            "bump_center": np.nan, 
            "baseline": baseline, 
            "bump_amp": bump_amp }
    target = {"bump_pre_center": [EIF_params["LM_pre_center_p1"],
                                EIF_params["LM_pre_center_p2"]], 
            "bump_center": np.nan, 
            "baseline": baseline, 
            "bump_amp": bump_amp }
    source["bump_center"] = [bump_centers1[:, 0], bump_centers2[:, 0]]
    target["bump_center"] = [bump_centers1[:, 1], bump_centers2[:, 1]]
    getI_real_data(source, dt, nt, npadding, EIF_params["V1_template"])
    getI_real_data(target, dt, nt, npadding, EIF_params["LM_template"])

    I_ext = 0.0*np.ones((nt_tot, nneuron, ntrial))
    if not no_stimulus:
        I_ext[:,:nneuron_part,:] = np.repeat(source["I"][:,np.newaxis,:],nneuron_part, axis=1)
        I_ext[:,nneuron_part:,:] = np.repeat(target["I"][:,np.newaxis,:],nneuron_part, axis=1)

    trial_info_slope, trial_info_gain = None, None
    ### Getting the external input after some background slope is added
    if params["external_input_type"] in [
        "two_peaks_with_varying_slope", "two_peaks_with_varying_timing_baseline_slope"
        ]:
        slopes = np.random.uniform(params["min_slope"], params["max_slope"], (1, 1, ntrial))
        t = np.arange(nt_tot)[:,np.newaxis,np.newaxis]
        time_line = slopes * 1e-3 * (t - nt/2 - npadding)
        I_ext += (time_line - time_line.min())  # Center around middle timepoint
        trial_info_slope = slopes.flatten()
    if params["external_input_type"] in [
        "two_peaks_with_varying_gain", "two_peaks_with_varying_timing_baseline_slope"
        ]:
        gains = np.random.uniform(params["min_gain"], params["max_gain"], (1, 1, ntrial))
        I_ext += gains
        trial_info_gain = gains.flatten()
    if params["external_input_type"] == "two_peaks_with_all_varying":
        # Generate baseline variations using Gaussian Process
        t = np.arange(nt_tot)

        # Generate GP baseline with RBF kernel
        length_scale = params["gp_time_constant"] * ndt  # Controls smoothness of variations
        amplitude = params["gp_amplitude"]  # Controls magnitude of variations

        # Generate independent GP for each trial using lower resolution and interpolation
        downsample_factor = 50  # Reduce resolution by this factor
        t_low_res = t[::downsample_factor]

        gp_baseline = np.zeros((nt_tot, 2, ntrial))

        # Generate GP at lower resolution
        K_low_res = np.exp(-0.5 * (np.subtract.outer(t_low_res, t_low_res)**2) / length_scale**2)

        # Shape: (2, ntrial, len(t_low_res))
        shared_background = np.clip(shared_background, 0.0, 1.0)
        gp_low_res = amplitude * np.random.multivariate_normal(
            np.zeros(len(t_low_res)), K_low_res, size=(2, ntrial)
        )
        indep_component = np.sqrt(1 - shared_background) * gp_low_res[1, :, :]
        shared_component = np.sqrt(shared_background) * gp_low_res[0, :, :]
        gp_low_res[1, :, :] = indep_component + shared_component

        loading_mat = np.random.uniform(low=0.5, high=1.0, size=(nneuron_part, 2))

        for i in range(2):
            for trial in range(ntrial):
                # Interpolate for each trial separately
                gp_baseline[:, i, trial] = np.interp(t, t_low_res, gp_low_res[i, trial, :])

            # Apply loading matrix
            I_ext[:, i * nneuron_part : (i + 1) * nneuron_part, :] += (
                gp_baseline[:, i:i+1, :] * loading_mat[None, :, i:i+1]
            )

        trial_info_gain = gp_baseline.mean(axis=(0,1)).flatten()
        trial_info_slope = (gp_baseline[-1,:,:] - gp_baseline[0,:,:]).flatten() / (nt_tot * dt)

    if params["external_input_type"] == "two_peaks_with_varying_timing":
        pass

    # Unit: mV or ms
    tau = 15
    EL = -60
    VT = -50
    Vth = -10
    Vre = -65
    DltT = 2
    tau_ref = 1
    tau_d = 5
    tau_r = 1

    nsyn_func = 3*tau_d*ndt
    syn_time_line = np.arange(nsyn_func)
    syn_func = (np.exp(-syn_time_line*dt/tau_d)-np.exp(-syn_time_line*dt/tau_r))[:, np.newaxis]
    ntau_ref = tau_ref*ndt
    spikes_rcd = np.zeros((int(nt_tot/ndt/bin_size), nneuron, ntrial))
    I_syn_rcd = np.zeros((int(nt_tot/ndt/bin_size), nneuron, ntrial))
    I_ext_rcd = np.zeros((int(nt_tot/ndt/bin_size), nneuron, ntrial))
    for itrial in tqdm(range(ntrial)):
        V = Vre*np.ones((nt_tot, nneuron))
        I_syn_cross = np.zeros((nt_tot+nsyn_func+1, nneuron))
        I_syn_recurrent = np.zeros((nt_tot+nsyn_func+1, nneuron))
        I_syn = np.zeros((nt_tot+nsyn_func+1, nneuron))
        spikes = np.zeros((nt_tot, nneuron))

        for t in range(1, nt_tot):
            I_leak = -dt/tau*(V[t-1,:]-EL) + dt/tau*DltT*np.exp((V[t-1,:]-VT)/DltT)
            I_noise = np.random.normal(0, noise_amp**2, nneuron)
            dV = I_leak + I_noise + I_syn[t,:] + I_ext[t,:,itrial]
            V[t,:] = V[t-1,:] + dV*(int(dt/0.1))

            spikes[t,:] = (V[t,:]>=Vth).astype(int)
            I_syn_cross[(t+1):(t+nsyn_func+1),:] += spikes[t:(t+1),:]@J_cross*syn_func
            I_syn_recurrent[(t+1):(t+nsyn_func+1),:] += spikes[t:(t+1),:]@J_recurrent*syn_func
            I_syn[(t+1):(t+nsyn_func+1),:] = I_syn_cross[(t+1):(t+nsyn_func+1),:] + I_syn_recurrent[(t+1):(t+nsyn_func+1),:]

            in_ref = ((spikes[max(0,t-ntau_ref):(t+1),:].sum(axis=0))>0).astype(int)
            V[t,:] += in_ref*(Vre-V[t,:])

            # Sum spikes over ndt time bins and store directly in final array
            if t % (ndt*bin_size) == 0:
                time_idx = t // (ndt*bin_size)
                spikes_rcd[time_idx,:,itrial] = spikes[t-ndt*bin_size+1:t+1,:].sum(axis=0)
                I_syn_rcd[time_idx,:,itrial] = I_syn_cross[t-ndt*bin_size+1:t+1,:].mean(axis=0)
                I_ext_rcd[time_idx,:,itrial] = I_ext[t-ndt*bin_size+1:t+1,:, itrial].mean(axis=0)
            
    res = [spikes_rcd]
    if return_trial_info:
        res.append({"bump": trial_info_bump, "gain": trial_info_gain, "slope": trial_info_slope})
    if return_I_ext:
        res.append(I_ext)
    if return_I_syn:
        res.append(I_syn_rcd)
    return res


    
def get_p(std1, corr1, std2, corr2, ntrial, conn):

    # std1 = 0
    # corr1 = 0.0
    # std2 = 0
    # corr2 = 0.0
    # ntrial = 100
    # conn = 0.008

    ntimes = 1
    nfold = 1
    penalty = 1e-2
    penalty_pop = 1e-5
    num = 20
    num_pop = 50
    num_cp = 3
    peaks_max = 15

    tau = 10
    num_f_refractory = 4

    result = []
    result_pop = []
    
    test_nll_full = []
    test_nll_nest = []
    test_nll_nestest_pop = []
    test_nll_full_pop = []
    test_nll_nest_pop = []
    nll_diff = 0
    nll_diff_pop = 0
    
    spikes_rcd = EIF_simulator(std1, corr1, std2, corr2, ntrial, conn)
    nt, nneuron, ntrial = spikes_rcd.shape
    
    spike_trains_source = []
    spike_trains_target = []
    time_line = np.arange(nt)
    
    for ineuron in range(nneuron):
        if ineuron < int(nneuron/2):
            spike_trains_source.append(spikes_rcd[:,ineuron,:])
        else:
            spike_trains_target.append(spikes_rcd[:,ineuron,:])
    
    # kf = KFold(n_splits=nfold)
    # for ifold, (train_index, test_index) in enumerate(kf.split(np.ones(ntrial))):
    #     if ifold!=0:
    #         break
        # coupling_rcd_once = []
        # training_trials = np.array( [False]*ntrial )
        # training_trials[train_index] = True
        # test_trials = np.logical_not(training_trials)
        
    for ifold in range(1):

        all_trials = np.array( [True]*ntrial )

        ### Single neuron level model
        for output_spike_train in spike_trains_target:

            model_full = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
            model_full.add_effect("inhomogeneous_baseline", num=num, apply_no_penalty=True)
            for input_spike_train in spike_trains_source:
                model_full.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            for input_spike_train in spike_trains_target:
                model_full.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            model_full.fit(target=output_spike_train, method='mine', penalty=penalty, verbose=False)
            # test_nll_full.append( model_full.test(test_trials) )
            test_nll_full.append( model_full.nll )
            
#             coupling_rcd_once.append( np.mean(model_full.get_filter()[1:11], axis=0) )
            
            model_nest = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
            model_nest.add_effect("inhomogeneous_baseline", num=num, apply_no_penalty=True)
            for input_spike_train in spike_trains_target:
                model_nest.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            model_nest.fit(target=output_spike_train, method='mine', penalty=penalty, verbose=False)
            # test_nll_nest.append( model_nest.test(test_trials) )
            test_nll_nest.append( model_nest.nll )
            nll_diff += model_nest.nll - model_full.nll
            

        ### Population level model
        output_spike_train_pool = np.zeros((nt, ntrial))
        for output_spike_train in spike_trains_target:
            output_spike_train_pool += output_spike_train
        input_spike_train_pool = np.zeros((nt, ntrial))
        for input_spike_train in spike_trains_source:
            input_spike_train_pool += input_spike_train
        
        model_full_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_full_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_full_pop.add_effect("coupling", raw_input=input_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        # model_full_pop.fit(target=output_spike_train_pool, method='mine', penalty=penalty_pop, verbose=False)
        model_full_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
                                                 warp_interval=[[0, 0.15],[0.15, 0.35]], 
                                                 method='mine', penalty=penalty_pop, verbose=False,)
                                                #  initial_shifts='peaks')
#         model_full_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
#                                                  warp_interval=[[0, 0.15],[0.15, 0.35]], 
#                                                  method='mine', penalty=penalty_pop, verbose=False, 
#                                                  fix_shifts=best_shifts)
        # test_nll_full_pop.append( model_full_pop.test(test_trials) )
        test_nll_full_pop.append( model_full_pop.nll )

        model_nest_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_nest_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_nest_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_nest_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        # model_nest_pop.fit(target=output_spike_train_pool, method='mine', penalty=penalty_pop, verbose=False)
        model_nest_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
                                                 warp_interval=[[0, 0.15],[0.15, 0.35]], 
                                                 method='mine', penalty=penalty_pop, verbose=False,)
                                                #  initial_shifts='peaks')
#         model_nest_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
#                                                  warp_interval=[[0, 0.15],[0.15, 0.35]], 
#                                                  method='mine', penalty=penalty_pop, verbose=False,
#                                                 fix_shifts=best_shifts)
        # test_nll_nest_pop.append( model_nest_pop.test(test_trials) )
        test_nll_nest_pop.append( model_nest_pop.nll )
        
        nll_diff_pop += model_nest_pop.nll - model_full_pop.nll
        df_diff_pop = model_full_pop.predictors.shape[1] - model_nest_pop.predictors.shape[1]
        
    # return wilcoxon(test_nll_full, test_nll_nest, alternative='less'), \
    #     ((nll_diff_pop, df_diff_pop), chi2.sf(2*nll_diff_pop, df_diff_pop)), \
        
    return fisher_method(test_nll_full, test_nll_nest), \
        ((nll_diff_pop, df_diff_pop), chi2.sf(2*nll_diff_pop, df_diff_pop)), \
        
        # wilcoxon(test_nll_full, test_nll_nest, alternative='less'), \
        # wilcoxon(test_nll_full_pop, test_nll_nest_pop, alternative='less')
        
def entire_lrt(test_nll_full, test_nll_nest):
    stat = 2*(np.sum(test_nll_nest)-np.sum(test_nll_full))
    final_p = chi2.sf(stat, 30*10)
    return (stat, final_p)


def fisher_method(test_nll_full, test_nll_nest):
    ps = []
    for i in range(len(test_nll_full)):
        ps.append(chi2.sf(2*(test_nll_nest[i]-test_nll_full[i]), 30))
    stat = -2*np.sum(np.log(ps))
    final_p = chi2.sf(stat, 2*len(test_nll_full))
    return (np.mean(ps), final_p)
    
def split_lrt(nll_full, nll_nest):
    return ( (nll_full, nll_nest), 1/np.exp(-nll_full+nll_nest) )

def get_ps(ntimes=10, std1=0.0, corr1=0.0, std2=0.0, corr2=0.0, ntrial=100, conn=0.008):
    import multiprocessing
    import os

    PROCESSES = 10
    # ntimes = 10
    result = []
    result_pop = []

    with multiprocessing.get_context('spawn').Pool(processes = PROCESSES) as pool:               
        ress = [pool.apply_async(get_p, (std1, corr1, std2, corr2, ntrial, conn)) 
                    for i_null in np.arange(ntimes)]
        pool.close()
        for res in tqdm(ress):
            result_temp, result_pop_temp = res.get()
            result.append(result_temp)
            result_pop.append(result_pop_temp)
    return result, result_pop


def get_p_three_models(std1, corr1, std2, corr2, ntrial, conn):

    # std1 = 0
    # corr1 = 0.0
    # std2 = 0
    # corr2 = 0.0
    # ntrial = 100
    # conn = 0.008

    ntimes = 1
    nfold = 1
    penalty = 1e-2
    penalty_pop = 1e-5
    num = 20
    num_pop = 40
    num_cp = 3
    peaks_max = 15

    tau = 10
    num_f_refractory = 4

    result = []
    result_pop = []
    
    test_nll_full = []
    test_nll_nest = []
    test_nll_nestest_pop = []
    test_nll_full_pop = []
    test_nll_nest_pop = []
    nll_diff = 0
    
    spikes_rcd = EIF_simulator(std1, corr1, std2, corr2, ntrial, conn)
    nt, nneuron, ntrial = spikes_rcd.shape
    
    spike_trains_source = []
    spike_trains_target = []
    time_line = np.arange(nt)
    
    for ineuron in range(nneuron):
        if ineuron < int(nneuron/2):
            spike_trains_source.append(spikes_rcd[:,ineuron,:])
        else:
            spike_trains_target.append(spikes_rcd[:,ineuron,:])
    
    # kf = KFold(n_splits=nfold)
    # for ifold, (train_index, test_index) in enumerate(kf.split(np.ones(ntrial))):
    #     if ifold!=0:
    #         break
        # coupling_rcd_once = []
        # training_trials = np.array( [False]*ntrial )
        # training_trials[train_index] = True
        # test_trials = np.logical_not(training_trials)
        
    for ifold in range(1):

        all_trials = np.array( [True]*ntrial )

        ### Single neuron level model
        for output_spike_train in spike_trains_target:

            model_full = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
            model_full.add_effect("inhomogeneous_baseline", num=num, apply_no_penalty=True)
            for input_spike_train in spike_trains_source:
                model_full.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            for input_spike_train in spike_trains_target:
                model_full.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            model_full.fit(target=output_spike_train, method='mine', penalty=penalty, verbose=False)
            # test_nll_full.append( model_full.test(test_trials) )
            test_nll_full.append( model_full.nll )
            
#             coupling_rcd_once.append( np.mean(model_full.get_filter()[1:11], axis=0) )
            
            model_nest = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
            model_nest.add_effect("inhomogeneous_baseline", num=num, apply_no_penalty=True)
            for input_spike_train in spike_trains_target:
                model_nest.add_effect("coupling", raw_input=input_spike_train, num=num_cp, peaks_max=peaks_max, nonlinear=1)
            model_nest.fit(target=output_spike_train, method='mine', penalty=penalty, verbose=False)
            # test_nll_nest.append( model_nest.test(test_trials) )
            test_nll_nest.append( model_nest.nll )
            nll_diff += model_nest.nll - model_full.nll
            

        ### Population level model
        output_spike_train_pool = np.zeros((nt, ntrial))
        for output_spike_train in spike_trains_target:
            output_spike_train_pool += output_spike_train
        input_spike_train_pool = np.zeros((nt, ntrial))
        for input_spike_train in spike_trains_source:
            input_spike_train_pool += input_spike_train
        
        model_full_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_full_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_full_pop.add_effect("coupling", raw_input=input_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        model_full_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
                                                 warp_interval=[[0, 0.15],[0.15, 0.35]], 
                                                 method='mine', penalty=penalty_pop, verbose=False,)
        test_nll_full_pop.append( model_full_pop.nll )

        model_nest_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_nest_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_nest_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_nest_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        model_nest_pop.fit_time_warping_baseline(target=output_spike_train_pool, max_iter=5, 
                                                 warp_interval=[[0, 0.15],[0.15, 0.35]], 
                                                 method='mine', penalty=penalty_pop, verbose=False,)
        test_nll_nest_pop.append( model_nest_pop.nll )
        
        nll_diff_pop = model_nest_pop.nll - model_full_pop.nll
        df_diff_pop = model_full_pop.predictors.shape[1] - model_nest_pop.predictors.shape[1]

        ### Pop-GLM but without time warping
        model_full_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_full_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_full_pop.add_effect("coupling", raw_input=input_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_full_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        model_full_pop.fit(target=output_spike_train_pool, method='mine', penalty=penalty_pop, verbose=False)
        test_nll_full_pop.append( model_full_pop.nll )

        model_nest_pop = PP_GLM(ntrial=ntrial, nt=nt, select_trials=all_trials)
        model_nest_pop.add_effect("inhomogeneous_baseline", num=num_pop, apply_no_penalty=True)
        model_nest_pop.add_effect("coupling", raw_input=output_spike_train_pool, num=num_cp, peaks_max=peaks_max, nonlinear=1)
        model_nest_pop.add_effect('refractory_additive', raw_input=output_spike_train_pool, tau=tau, 
                                  num=num_f_refractory, apply_no_penalty=False)
        model_nest_pop.fit(target=output_spike_train_pool, method='mine', penalty=penalty_pop, verbose=False)
        test_nll_nest_pop.append( model_nest_pop.nll )
        
        nll_diff_pop_nowarp = model_nest_pop.nll - model_full_pop.nll
        df_diff_pop_nowarp = model_full_pop.predictors.shape[1] - model_nest_pop.predictors.shape[1]
        
    return fisher_method(test_nll_full, test_nll_nest), \
            ((nll_diff_pop, df_diff_pop), chi2.sf(2*nll_diff_pop, df_diff_pop)), \
            ((nll_diff_pop_nowarp, df_diff_pop_nowarp), chi2.sf(2*nll_diff_pop_nowarp, df_diff_pop_nowarp))

def get_ps_three_models(ntimes=10, std1=0.0, corr1=0.0, std2=0.0, corr2=0.0, ntrial=100, conn=0.008):
    import multiprocessing
    import os

    PROCESSES = 10
    # ntimes = 10
    result = []
    result_pop = []
    result_pop_nowarp = []

    with multiprocessing.get_context('spawn').Pool(processes = PROCESSES) as pool:               
        ress = [pool.apply_async(get_p_three_models, (std1, corr1, std2, corr2, ntrial, conn)) 
                    for i_null in np.arange(ntimes)]
        pool.close()
        for res in tqdm(ress):
            result_temp, result_pop_temp, result_pop_temp_nowarp = res.get()
            result.append(result_temp)
            result_pop.append(result_pop_temp)
            result_pop_nowarp.append(result_pop_temp_nowarp)
    return result, result_pop, result_pop_nowarp


def reduced_rank_regression(X, Y, r):
    # Center the data
    Y_mean = np.repeat(Y.mean(axis=2)[:,:,None], Y.shape[2], axis=2)
    X -= X.mean(axis=2)[:,:,None]
    Y -= Y.mean(axis=2)[:,:,None]

    # Flatten X and Y along the first and third dimensions
    nt, n_features, ntrial = X.shape
    _, n_outputs, _ = Y.shape
    
    X_flat = X.transpose(2, 0, 1).reshape(-1, n_features)
    Y_flat = Y.transpose(2, 0, 1).reshape(-1, n_outputs)
    Y_mean_flat = Y_mean.transpose(2, 0, 1).reshape(-1, n_outputs)

    X, Y = X_flat, Y_flat
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]

    # Fit the Reduced Rank Regression Model
    linreg = LinearRegression().fit(X, Y)
    B_OLS = linreg.coef_.T

    # Perform SVD and construct the reduced rank coefficient matrix
    U, D, Vt = svd(B_OLS, full_matrices=False)
    U_r = U[:, :r]
    D_r = D[:r]
    Vt_r = Vt[:r, :]
    B_rrr = U_r @ np.diag(D_r) @ Vt_r

    # Make predictions and compute RSS for the Reduced Rank Model
    Y_pred = X @ B_rrr + Y_mean_flat
    total_variance = np.var(Y_pred)
    variance_wo_coupling = np.var(Y_mean_flat)
    coupling_percentage = (total_variance - variance_wo_coupling) / total_variance
    return coupling_percentage