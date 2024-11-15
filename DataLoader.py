"""Three data loading adapters that read or generate the same standard of LFP data from Allen insititue, Prof. Teichert, 
and simulation. """

import sys
import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
import pickle
import copy

import utility_functions as utils

class LFP:
    def remove_padding_single(self, npadding):
        self.lfp = self.lfp[:,npadding:-npadding,:]

    def get_power_phase(self, padding_time, lowcut=20, highcut=35):
        self.phase, self.power = utils.get_power_phase(self.lfp, npadding=int(self.fps*padding_time), lowcut=lowcut, highcut=highcut)

    def get_mean_lfp(self):
        self.mean_lfp = self.lfp.mean(axis=2)
        self.mean_lfp = self.mean_lfp[:,:,None]

    def pre_smooth(self, moving_size = None, pooling_size = None):
        if moving_size == None:
            default_moving_dict = {'Allen':1, 'Tobias':1, 'Simulation':1}
            moving_size = default_moving_dict[self.source]
        if pooling_size == None:
            default_pooling_dict = {'Allen':5, 'Tobias':5, 'Simulation':1}
            moving_size = default_pooling_dict[self.source]
        self.lfp_smooth = utils.moving_average(self.lfp, pooling_size, moving_size)
        self.mean_lfp_smooth = utils.moving_average(self.mean_lfp, pooling_size, moving_size)

    def align_lfp(self):
        temp = np.swapaxes(self.lfp, 1,2)    # Temporally convert (electrode, time, trial) -> (electrode, trial, time)
        self.aligned_lfp = self.lfp.reshape(temp.shape[0],-1)
        
    def show(self, trial=0):
        if self.source == 'Simulation':
            plt.figure(figsize=(8, 6))
            plt.subplot(121)
            plt.imshow(self.gt_csd[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('True CSD')
            plt.ylabel('Depth (microns)')
            plt.xlabel('Time (ms)')
            plt.subplot(122)
            plt.imshow(self.lfp[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('LFP (noisy)')
            plt.xlabel('Time (ms)')
            plt.show()
        else:
            plt.figure(figsize=(12, 6))
            plt.imshow(self.lfp[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('LFP')
            plt.xlabel('Time (ms)')
            plt.show()

class Allen_LFP(LFP):
    def __init__(self):
        self.x = None
        self.t = None

class Allen_dataset:
    """ For drifting gratings, there are 30 unknown trials, 15*5*8=600 trials for 8 directions, 5 temporal frequencies, 
    15 iid trials each conditions. """
    # presentation_table is the center of trial info
    def __init__(self, **kwargs):

        self.source = "Allen"
        self.session_id = kwargs.pop('session_id', 791319847)
        self.selected_probes = kwargs.pop('selected_probes', ['probeC'])
        # self.probe_id = kwargs.pop('probe_id', 805008600)
        self.stimulus_name = kwargs.pop('stimulus_name',
                                        'drifting_gratings_contrast')
        self.orientation = kwargs.pop('orientation', None)
        self.temporal_frequency = kwargs.pop('temporal_frequency', None)
        self.contrast = kwargs.pop('contrast', None)
        self.stimulus_condition_id = kwargs.pop('stimulus_condition_id', None)
        self.start_time = kwargs.pop('start_time', 0)
        self.end_time = kwargs.pop('end_time', 0.4)
        self.padding = kwargs.pop('padding', 0.1)
        self.fps = kwargs.pop('fps', 1e3)
        
        assert type(self.selected_probes) in [str,list], "\"probe\" has to be either str or list!"
        if self.selected_probes=='all':
            self.selected_probes = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']
        if type(self.selected_probes) == str:
            self.selected_probes = [self.selected_probes]
        assert set(self.selected_probes).issubset(['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']) 
        
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        if sys.platform == 'linux':
            self.manifest_path = os.path.join('/home/qix/ecephys_cache_dir/', "manifest.json")
        elif sys.platform == 'win32' or 'darwin':
            self.manifest_path = os.path.join('D:/ecephys_cache_dir/', "manifest.json")
        else:
            raise ValueError("Undefined device!")
        self._cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        self._session = self._cache.get_session_data(self.session_id)
        if self.stimulus_name == "All":
            self.presentation_table = self._session.stimulus_presentations
        else:
            if isinstance(self.stimulus_name ,str):
                idx = self._session.stimulus_presentations.stimulus_name == self.stimulus_name
            else:
                idx = self._session.stimulus_presentations.stimulus_name .isin(self.stimulus_name) 
            if self.orientation != None:
                idx = idx & (self._session.stimulus_presentations.orientation.isin(self.orientation))
            if self.temporal_frequency != None:
                idx = idx & (self._session.stimulus_presentations.temporal_frequency.isin(self.temporal_frequency))
            if self.contrast != None:
                idx = idx & (self._session.stimulus_presentations.contrast.isin(self.contrast))
            if self.stimulus_condition_id != None:
                idx = idx & (self._session.stimulus_presentations['stimulus_condition_id'].isin(self.stimulus_condition_id))
            self.presentation_table = self._session.stimulus_presentations [idx]
        self.presentation_times = self.presentation_table.start_time.values
        self.presentation_ids = self.presentation_table.index.values
        self.probes = self._session.probes
        self.ntrial = len(self.presentation_ids)
        self.time_line = np.arange(self.start_time, self.end_time, 1/self.fps)
        self.nt = len(self.time_line)
        if self.padding is None:
            self.npadding = None
            self.time_line_padding = None
        else:
            self.npadding = int(self.padding*self.fps)
            self.time_line_padding = np.arange(self.start_time - self.padding, self.end_time, 1/self.fps)

    def get_trial_metric_per_unit_per_trial(
        self, 
        metric_type='spike_trains', 
        area='visual',
        dt=None, 
        empty_fill=np.nan, 
        verbose=False):

        """ Get spike trains of selected units.
        Args:
            metric_type:
                    'count',
                    'spike_trains' (spike histogram, array of binary of interger counts),
                    'spike_times' (a sequence of spike times)
        """
        if area == 'visual':
            self.selected_units = self._session.units[
                self._session.units['ecephys_structure_acronym'].isin(utils.VISUAL_AREA) &
                self._session.units['probe_description'].isin(self.selected_probes)]
        else:
            self.selected_units = self._session.units[
                self._session.units['probe_description'].isin(self.selected_probes)]
        trial_time_window = [self.start_time - self.padding, self.end_time]
        if dt is None:
            dt = 1/self.fps
        self.unit_ids = self.selected_units.index.values
        spikes_table = self._session.trialwise_spike_times(
                self.presentation_ids, self.unit_ids, trial_time_window)
        num_neurons = len(self.unit_ids)
        num_trials = len(self.presentation_ids)
        metric_table = pd.DataFrame(index=self.unit_ids, columns=self.presentation_ids)
        metric_table.index.name = 'units'

        if metric_type == 'spike_trains':
            time_bins = np.linspace(
                    trial_time_window[0], trial_time_window[1],
                    int((trial_time_window[1] - trial_time_window[0]) / dt) + 1)

        for u, unit_id in enumerate(self.unit_ids):
            if verbose and (u % 40 == 0):
                print('neuron:', u)
            for s, stimulus_presentation_id in enumerate(self.presentation_ids):
                spike_times = spikes_table[
                        (spikes_table['unit_id'] == unit_id) &
                        (spikes_table['stimulus_presentation_id'] ==
                         stimulus_presentation_id)]
                spike_times = spike_times['time_since_stimulus_presentation_onset']
                if metric_type == 'count':
                    metric_value = len(spike_times) if len(spike_times) != 0 else empty_fill
                elif metric_type == 'shift':
                    metric_value = np.mean(spike_times) if len(spike_times) != 0 else empty_fill

                # The spike train is special, since DataFrame does not take array as the
                # entry, we have to use a separate data structure to store the spike
                # trains. The metric table is used to store the index mapping.
                elif metric_type == 'spike_trains':
                    metric_value = np.histogram(spike_times, time_bins)[0]
                elif metric_type == 'spike_times':
                    metric_value = np.array(spike_times)
                else:
                    raise TypeError('Wrong type of metric')
                metric_table.loc[unit_id, stimulus_presentation_id] = metric_value
        # Very important step, change the datatype to numeric, otherwise functions
        # like correlation cannot be performed.
        if metric_type not in ['spike_trains', 'spike_times']:
            metric_table = metric_table.apply(pd.to_numeric, errors='coerce')
        if metric_type == 'spike_trains':
            self.spike_train = metric_table
        if metric_type == 'spike_times':
            self.spike_times = metric_table

    def get_running(self, method="Pillow"):
        running_speed = self._session.running_speed
        running_speed['mean_time'] = (running_speed['start_time']+running_speed['end_time'])/2
        running_speed_toarray_temp = running_speed.set_index(['mean_time'])
        self.running_speed_xarray = running_speed_toarray_temp['velocity'].to_xarray()
        # self.running_speed_xarray = self.running_speed_xarray.set_coords(('mean_time'))

        self.mean_speed = np.zeros(self.ntrial)
        self.min_speed = np.zeros(self.ntrial)
        self.max_speed = np.zeros(self.ntrial)
        self.speed = np.zeros((self.nt, self.ntrial))
        trial_window = np.arange(self.start_time,self.end_time, 1/self.fps)
        
        for i in range(self.ntrial):
            speed_temp = running_speed[np.logical_and(running_speed['mean_time']<self.presentation_times[i]+self.end_time , 
                                self.presentation_times[i]+self.start_time<running_speed['mean_time']).values]['velocity'].values
            self.mean_speed[i] = speed_temp.mean()
            self.min_speed[i] = speed_temp.min()
            self.max_speed[i] = speed_temp.max()

            time_selection = trial_window + self.presentation_times[i]
            self.speed[:,i] = self.running_speed_xarray.sel(mean_time = time_selection, method='nearest')
        if method=="Pillow":
            self.running_trial_index = np.logical_and( self.mean_speed >= 3 , self.min_speed >= 0.5 )
            self.stationary_trial_index = np.logical_and( self.mean_speed < 0.5 , self.max_speed < 3 )
        else:
            self.running_trial_index = self.mean_speed >= 1
            self.stationary_trial_index = self.mean_speed < 1
        self.all_trial_index = np.full(self.ntrial, True)

    def get_pupil_diam(self):
        pupil_table = self._session.get_pupil_data()
        pupil_table["pupil_diam"] = np.sqrt(
            pupil_table["pupil_height"]**2 + pupil_table["pupil_width"]**2
        )
        self.pupil_diam_xarray = pupil_table['pupil_diam'].to_xarray()
        self.pupil_diam_xarray = self.pupil_diam_xarray.rename({'Time (s)': 'time'})

        self.mean_pupil_diam = np.zeros(self.ntrial)
        self.min_pupil_diam = np.zeros(self.ntrial)
        self.max_pupil_diam = np.zeros(self.ntrial)
        self.pupil_diam = np.zeros((self.nt, self.ntrial))
        trial_window = np.arange(self.start_time,self.end_time, 1/self.fps)
        
        for i in range(self.ntrial):
            time_selection = trial_window + self.presentation_times[i]
            self.speed[:,i] = self.pupil_diam_xarray.sel(time = time_selection, method='nearest')
            self.mean_speed[i] = self.speed[:,i].mean()
            self.min_speed[i] = self.speed[:,i].min()
            self.max_speed[i] = self.speed[:,i].max()
        
    def get_lfp(self, **kwargs):
        self.lfp = {}
        for probe_name in self.selected_probes:
            
            temp_obj = Allen_LFP()
            probe_id = self.probes[self.probes['description']==probe_name].index[0]
            
            lfp_data = self._session.get_lfp(probe_id)
            trial_window = np.arange(self.start_time-self.padding,self.end_time+self.padding, 1/self.fps)
            time_selection = np.concatenate([trial_window + t for t in self.presentation_times])
            inds = pd.MultiIndex.from_product((self.presentation_ids, trial_window), 
                                        names=('presentation_id', 'time_from_presentation_onset'))
            ds = lfp_data.sel(time = time_selection, method='nearest').to_dataset(name = 'aligned_lfp')
            ds = ds.assign(time=inds).unstack('time')
            lfp_temp = ds['aligned_lfp'].values     # Three dimensions. e.g. (77, 540, 625). Channels, trials, times
            lfp_temp = np.swapaxes(lfp_temp,1,2)    # Swap time and trial. e.g. (77, 625, 540). Channels, times, trials
            try:
                location = self._session.channels[['probe_vertical_position','probe_horizontal_position']]
                location = location.loc[lfp_data['channel'].values].values
                x = location[:, 0]      # LFP spatial locations , microns
            except:
                location = np.arange(0,lfp_temp.shape[0])*40.0
                x = location

            # temp_obj.t = np.linspace(self.start_time,self.end_time,lfp_temp.shape[1] )[:,None]
            temp_obj.t = trial_window
            temp_obj.x = x[:, None]
            temp_obj.channel = lfp_data["channel"].values
            try:
                temp_obj.structure_acronyms, temp_obj.intervals_lfp = self._session.channel_structure_intervals(temp_obj.channel)
            except:
                temp_obj.structure_acronyms = np.array([np.nan], dtype=object)
                temp_obj.intervals_lfp = np.array([ 0, lfp_temp.shape[0]])
            temp_obj.nx, temp_obj.nt, temp_obj.ntrial, temp_obj.lfp = utils.check_and_get_size(lfp_temp)
            temp_obj.get_mean_lfp()
            self.lfp[probe_name] = temp_obj

    def remove_padding(self, padding_time):
        npadding = int(padding_time*self.fps)
        for key, value in self.lfp.items():
            value = value.remove_padding_single(npadding)
            # self.lfp[key] = value

    def get_spike_train_sparse(self):
        self._units_pd = self._session.units[(self._session.units.probe_id == self.probe_id)]
        self.unit_id_list = self._units_pd.index.values
        self._channel_index = self._units_pd.loc[self.unit_id_list].channel_local_index
        self.st_channel_id_list = []
        self.nunit = len(self.unit_id_list)
        self.spike_train_sparse = []
        for i in range(self.nunit):
            self.st_channel_id_list.append( self._session.channels[(self._session.channels.local_index == self._channel_index.values[i]) & 
                                                (self._session.channels.probe_id == self.probe_id)].index.values[0] )
            spike_times = self._session.spike_times[self.unit_id_list[i]]
            self.spike_train_sparse.append([ spike_times[ np.logical_and(spike_times>t+self.start_time, spike_times<t+self.end_time) ]-t 
                                     for t in self.presentation_times ])
            
        self.st_channel_id_list = np.array(self.st_channel_id_list)
        
    def get_spike_train(self):
        # (nunit, nt, ntrial)
        from scipy.sparse import csr_matrix
        self.get_spike_train_sparse()

        self.spike_train = np.zeros((self.nunit, self.nt, self.ntrial))
        for i in range(self.nunit):
            row = np.array([])
            col = np.array([])
            data = np.array([])
            for trial in range(self.ntrial):
                temp = self.spike_train_sparse[i][trial]
                nspike = len(temp)
                row = np.hstack( (row, np.array(temp)*self.fps) )
                col = np.hstack( (col, trial*np.ones(nspike)) )
                data = np.hstack( (data, np.ones(nspike)) )
            
            self.spike_train[i,:,:] = csr_matrix((data, (row, col)), shape=(self.nt, self.ntrial)).toarray()
        self.spike_count = self.spike_train.sum(axis=(1))
        
    def get_pooled_spike_train(self):
        pooled_spike_train = []
        for i in range(self.nunit):
            pooled_spike_train.append( np.concatenate(self.spike_train_sparse[i]) )
        return pooled_spike_train
    
    def get_fr(self):
        raise ValueError ("need to be in rCSD folder! ")
        # import smoothing_spline
        # fit_model = smoothing_spline.SmoothingSpline()
        # time_line = np.arange(self.start_time,self.end_time,1.0/self.fps)
        # eta_smooth_tuning = 1e-10
        # f_basis, f_Omega = fit_model.construct_basis_omega(
        #     time_line, knots=15, verbose=False)
        
        # self.fr = np.zeros((self.nunit, self.nt, self.ntrial))
        # for i in range(self.nunit):
        #     for trial in range(self.ntrial):
        #         temp_spike_train = self.spike_train[i,:,trial]
        #         # log_lambda_hat, beta = fit_model.poisson_regression(temp_spike_train[None,:], f_basis)
        #         log_lambda_hat, (beta, beta_baseline, log_lambda_offset, hessian, hessian_baseline, nll) \
        #             = fit_model.poisson_regression_smoothing_spline(
        #                 temp_spike_train[None,:], time_line, basis=f_basis, Omega=f_Omega, constant_fit=False)
        #         self.fr[i, :, trial] = np.exp(log_lambda_hat)

    def get_kernel_fr(self, bandwidth=0.03):
        self.bandwidth = bandwidth
        from sklearn.neighbors import KernelDensity

        emp_fr_X = np.arange(self.start_time,self.end_time, 1/self.fps)
        emp_fr_X = emp_fr_X[:,None]
        emp_fr = np.zeros((self.nunit, emp_fr_X.shape[0], self.ntrial))
        for i in range(self.nunit):
            for trial in range(self.ntrial):
                points = np.array(self.spike_train_sparse[i][trial])
                
                if len(points)>0:
                    points = points[:,None]
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
                    logdensity = kde.score_samples( emp_fr_X )
                    emp_fr[i,:, trial] = np.exp(logdensity)
                    total_spike = points.shape[0]
                    emp_fr[i,:, trial] = emp_fr[i,:, trial]*total_spike
        self.kernel_fr = emp_fr
        return emp_fr

    def get_psth(self, bandwidth=0.02):
        self.bandwidth = bandwidth
        from sklearn.neighbors import KernelDensity
        pooled_spike_train = self.get_pooled_spike_train() 
        emp_fr_X = np.arange(self.start_time,self.end_time, 1/self.fps)
        emp_fr_X = emp_fr_X[:,None]
        emp_fr = np.zeros((self.nunit, emp_fr_X.shape[0]))
        for i in range(self.nunit):
            points = np.array(pooled_spike_train[i])
            points = points[:,None]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
            logdensity = kde.score_samples( emp_fr_X )
            emp_fr[i,:] = np.exp(logdensity)
            total_spike = points.shape[0]
            emp_fr[i,:] = emp_fr[i,:]*total_spike/self.ntrial
        self.emp_fr = emp_fr
        return emp_fr
    
    def print_session_info(self):
        """Print a list of information for the session."""
        if self.session is None:
            print('session is None')
            return

        print(self.session._metadata)
        print('num units:', self.session.num_units)

class Tobias_LFP(LFP):
    def __init__(self, **kwargs):
        self.source = "Tobias"
        self.dataset_id = kwargs.pop('dataset_id', 'Walter_20160512')
        self.probe_id = kwargs.pop('probe_id', 805008600)
        self.start_time = kwargs.pop('start_time', 0)
        self.end_time = kwargs.pop('end_time', self.start_time+500)
        self.get_Tobias()
        
    def get_lfp(self):
        import scipy.io
        mat = scipy.io.loadmat("D:/LFP data/"+self.dataset_id+".mat")
        self.lfp = mat['data']
        self.lfp = self.lfp[:, self.start_time: self.end_time]
        # Set up spatial locations , temporal locations
        self.t = np. linspace(0, self.nt, self.nt)[:, None]       # time points , milliseconds
        self.x = np. linspace(0, 2300, 24)[:, None]     # LFP spatial locations , microns
        self.nx, self.nt, self.ntrial, self.lfp = utils.check_and_get_size(self.lfp)
        self.get_mean_lfp()
        self.intervals_lfp = np.array([0, np.nonzero(self.x > 1000)[0][0],
                                np.nonzero(self.x > 1500)[0][0], self.nx])
        self.structure_acronyms = np.array(['Superficial', 'Medium', 'Deep'])
    
# class Simulation_LFP(LFP):
#     def __init__(self, **kwargs):
#         """To be done later. Need to include: gaussian bumps; the cases from jupyter notebooks"""
#         if sys.platform == 'linux':
#             sys.path.append("/home/qix/rCSD")
#         else:
#             sys.path.append("D:/Github/rCSD")
#         import ground_true_csd_bank
#         from forward_models import b_fwd_1d, fwd_model_1d
        
#         self.source = "Simulation"
#         self.noise_amp = kwargs.pop('noise_amp', 0.03)
#         if 'gt_csd' in kwargs:
#             print("not finished! Can't allow user self design gt csd at the moment! ")
#             raise ValueError("Unfinished!")
#             # self.gt_csd = kwargs['gt_csd']
#             # self.generate_lfp(noise_amp)
#         else:
#             self.gt_csd_id = kwargs.pop('gt_csd_id', 0)
#             self.get_csd()
#             self.get_lfp()

#     def get_csd(self):

#         self.R = 150
#         self.nx = 24
#         self.nz = 5*23
#         self.x = np. linspace(0, 2300, self.nx)[:, None]
#         self.z = np. linspace(0+10, 2300-10, self.nz)[:, None]
#         self.gt_csd, self.t, self.nt = ground_true_csd_bank.csd_simple_templete(self.gt_csd_id, self.z)
#         self.ntrial = self.gt_csd.shape[2]
        
#     def get_lfp(self):
#         lfp_noiseless = fwd_model_1d(self.gt_csd, self.z, self.x, self.R)
#         noise = self.noise_amp * np.abs(lfp_noiseless).max()*np.random.randn(self.nx, self.nt, self.ntrial)
#         self.lfp = lfp_noiseless + noise
#         self.nx, self.nt, self.ntrial, self.lfp = utils.check_and_get_size(self.lfp)
#         self.get_mean_lfp()
#         self.intervals_lfp = np.array([0, np.nonzero(self.x > 1000)[0][0],
#                                 np.nonzero(self.x > 1500)[0][0], self.nx])
#         self.structure_acronyms = np.array(['Superficial', 'Medium', 'Deep'])

