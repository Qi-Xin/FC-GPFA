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
import logging
import utility_functions as utils
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from collections import defaultdict
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


class Allen_dataloader_multi_session():
    def __init__(self, session_ids, **kwargs):
        """
        Args:
            session_ids (list): List of session IDs to load
            train_ratio (float): Ratio of data to use for training (default: 0.7)
            val_ratio (float): Ratio of data to use for validation (default: 0.1)
            batch_size (int): Number of trials per batch (default: 32)
            shuffle (bool): Whether to shuffle the data (default: True)
            **kwargs: Additional arguments passed to Allen_dataset
        """
        self.session_ids = session_ids if isinstance(session_ids, list) else [session_ids]
        self.train_ratio = kwargs.pop('train_ratio', 0.7)
        self.val_ratio = kwargs.pop('val_ratio', 0.1)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.shuffle = kwargs.pop('shuffle', True)
        self.common_kwargs = kwargs

        logger = logging.getLogger(__name__)
        logger.info(f"Total number of sessions: {len(self.session_ids)}")
        logger.info(f"Train ratio: {self.train_ratio}")
        logger.info(f"Val ratio: {self.val_ratio}")
        logger.info(f"Test ratio: {1-self.train_ratio-self.val_ratio}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Initialize session info
        self._initialize_sessions()
        
        # Split data into train/val/test
        self._split_data()
        
        # Initialize iterators
        self.current_session = None
        self.current_batch_idx = 0
        self.current_split = 'train'

    def _initialize_sessions(self):
        """Initialize metadata for all sessions"""
        self.sessions = {}
        self.total_trials = 0
        self.session_trial_counts = []
        self.session_trial_indices = []
        
        for session_id in self.session_ids:
            # Get trial count for this session
            self.sessions[session_id] = Allen_dataset(session_id=session_id, **self.common_kwargs)
            n_trials = len(self.sessions[session_id].presentation_ids)
            
            self.session_trial_counts.append(n_trials)
            self.session_trial_indices.append((self.total_trials, self.total_trials + n_trials))
            self.total_trials += n_trials

    def _split_data(self):
        """Split trials into train/val/test sets"""
        all_indices = np.arange(self.total_trials)
        if self.shuffle:
            np.random.shuffle(all_indices)
            
        # Calculate split points
        train_size = int(self.total_trials * self.train_ratio)
        val_size = int(self.total_trials * self.val_ratio)
        
        self.train_indices = all_indices[:train_size]
        self.val_indices = all_indices[train_size:train_size + val_size]
        self.test_indices = all_indices[train_size + val_size:]
        
        # Create batch indices
        self.train_batches = self._create_batches(self.train_indices)
        self.val_batches = self._create_batches(self.val_indices)
        self.test_batches = self._create_batches(self.test_indices)

    def _create_batches(self, indices):
        """Create batches from indices"""
        n_samples = len(indices)
        n_batches = n_samples // self.batch_size
        batches = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batches.append(indices[start_idx:end_idx])
            
        # Handle remaining samples
        if n_samples % self.batch_size != 0:
            batches.append(indices[n_batches * self.batch_size:])
            
        return batches

    def _get_session_for_trial(self, trial_idx):
        """Get session ID for a given trial index"""
        for i, (start, end) in enumerate(self.session_trial_indices):
            if start <= trial_idx < end:
                return self.session_ids[i], trial_idx - start
        raise ValueError(f"Invalid trial index: {trial_idx}")

    def _load_batch(self, batch_indices, include_behavior=False):
        """Load a batch of trials"""
        # Group trials by session
        session_trials = defaultdict(list)
        for trial_idx in batch_indices:
            session_id, local_idx = self._get_session_for_trial(trial_idx)
            session_trials[session_id].append(local_idx)

        # Load data for each session
        batch_data = []
        for session_id, local_indices in session_trials.items():
            current_session = self.sessions[session_id]

            # Extract trials for this session
            for local_idx in local_indices:
                trial_data = {
                    'spike_train': current_session.get_trial_metric_per_unit_per_trial(
                        selected_trials=[local_idx]
                    ),
                    'session_id': session_id,
                    'trial_idx': local_idx
                }
                batch_data.append(trial_data)

        return batch_data

    def get_batch(self, split='train'):
        """Get next batch for specified split"""
        if split == 'train':
            batches = self.train_batches
        elif split == 'val':
            batches = self.val_batches
        elif split == 'test':
            batches = self.test_batches
        else:
            raise ValueError(f"Invalid split: {split}")

        if self.current_batch_idx >= len(batches):
            self.current_batch_idx = 0
            if self.shuffle and split == 'train':
                np.random.shuffle(self.train_batches)
            return None

        batch = self._load_batch(batches[self.current_batch_idx])
        self.current_batch_idx += 1
        return batch

    def reset(self, split='train'):
        """Reset batch iterator for specified split"""
        self.current_batch_idx = 0
        self.current_split = split
        if self.shuffle and split == 'train':
            np.random.shuffle(self.train_batches)


def combine_stimulus_presentations(stimulus_presentations, time_window=0.49):
    """ Combine or split stimulus presentations so each trial is at least time_window length
    Combined trials must have the same stimulus_name and are consecutive. """
    # stimulus_presentations = stimulus_presentations.sort_values(by="start_time") # it's already sorted
    combined_stimulus_presentations = []
    for i, row in stimulus_presentations.iterrows():
        if (
            combined_stimulus_presentations
            and combined_stimulus_presentations[-1]["stimulus_name"] != row["stimulus_name"]
            and (combined_stimulus_presentations[-1]["stop_time"] - 
                 combined_stimulus_presentations[-1]["start_time"]) < time_window
        ):
            # If the last combined stimulus presentation is not the same as the current one, and 
            # the last combined stimulus presentation is less than time_window away from the current one,
            # then we pop the last combined stimulus presentation.
            combined_stimulus_presentations.pop()
        if (
            not combined_stimulus_presentations
            or combined_stimulus_presentations[-1]["stimulus_name"] != row["stimulus_name"]
            or (combined_stimulus_presentations[-1]["stop_time"] - 
                 combined_stimulus_presentations[-1]["start_time"]) >= time_window
        ):
            # If the last combined stimulus presentation is not the same as the current one, or 
            # there is no combined stimulus presentation yet, we append the current one.
            combined_stimulus_presentations.append(row)
        else:
            # If the last combined stimulus presentation is the same as the current one,
            # we combine them and update the stop time.
            combined_stimulus_presentations[-1]["stop_time"] = row["stop_time"]            
    return pd.DataFrame(combined_stimulus_presentations)


def get_fake_stimulus_presentations(presentation_table, time_window=0.5, 
                                    interval_minimum=0.05, interval_maximum=0.1):
    """ Get random trials that are the same length of time_window. 
     Inter trial interval is a uniform distribution. """

    # Get the last stop time from presentation_table, or start at 0
    experiment_start_time = presentation_table['start_time'].min()
    experiment_stop_time = presentation_table['stop_time'].max()
    
    # Generate start times with random intervals until we reach experiment_stop_time
    start_times = []
    current_time = experiment_start_time
    
    while current_time + time_window <= experiment_stop_time:
        # Add random interval between 0 and 0.1
        interval = np.random.uniform(interval_minimum, interval_maximum)
        current_time += interval
        
        # Only add if there's room for full trial
        if current_time + time_window <= experiment_stop_time:
            start_times.append(current_time)
            current_time += time_window
    
    # Create DataFrame with stimulus_presentation_id as index
    fake_stimulus_presentations = pd.DataFrame({
        'start_time': start_times,
        'stop_time': [start + time_window for start in start_times]
    }, index=pd.RangeIndex(len(start_times), name='stimulus_presentation_id'))
    
    return fake_stimulus_presentations


class Allen_dataset:
    """ For drifting gratings, there are 30 unknown trials, 15*5*8=600 trials for 8 directions, 5 temporal frequencies, 
    15 iid trials each conditions. """
    # presentation_table is the center of trial info
    def __init__(self, verbose=False, **kwargs):

        self.source = "Allen"
        self.session_id = kwargs.pop('session_id', 791319847)
        self.selected_probes = kwargs.pop('selected_probes', 'all')
        self.align_stimulus_onset = kwargs.pop('align_stimulus_onset', True)
        self.merge_trials = kwargs.pop('merge_trials', False)
        self.stimulus_name = kwargs.pop('stimulus_name', "all")
        self.orientation = kwargs.pop('orientation', None)
        self.temporal_frequency = kwargs.pop('temporal_frequency', None)
        self.contrast = kwargs.pop('contrast', None)
        self.stimulus_condition_id = kwargs.pop('stimulus_condition_id', None)
        self.start_time = kwargs.pop('start_time', 0)
        self.end_time = kwargs.pop('end_time', 0.4)
        self.padding = kwargs.pop('padding', 0.1)
        self.fps = kwargs.pop('fps', 1e3)
        self.area = kwargs.pop('area', 'visual')

        self.padding = kwargs.get('padding', 0.1)
        self.fps = kwargs.get('fps', 1e2)
        if verbose:
            logger = logging.getLogger(__name__)
            logger.info(f"Align stimulus: {self.align_stimulus}")
            logger.info(f"Trial length: {self.end_time - self.start_time}")
            logger.info(f"Padding: {self.padding}")
            logger.info(f"FPS: {self.fps}")
            logger.info(f"Area: {self.area}")
        
        assert type(self.selected_probes) in [str,list], "\"probe\" has to be either str or list!"
        if self.selected_probes=='all':
            self.selected_probes = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']
        if type(self.selected_probes) == str:
            self.selected_probes = [self.selected_probes]
        assert set(self.selected_probes).issubset(['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']) 
        
        if sys.platform == 'linux':
            self.manifest_path = os.path.join('/home/qix/ecephys_cache_dir/', "manifest.json")
        elif sys.platform == 'win32' or 'darwin':
            self.manifest_path = os.path.join('D:/ecephys_cache_dir/', "manifest.json")
        else:
            raise ValueError("Undefined device!")

        self._cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        self._session = self._cache.get_session_data(self.session_id)

        # Get stimulus presentation table (Select trials)
        if self.align_stimulus_onset:
            if self.stimulus_name == "all":
                self.presentation_table = self._session.stimulus_presentations
            else:
                if isinstance(self.stimulus_name ,str):
                    idx = self._session.stimulus_presentations.stimulus_name == self.stimulus_name
                else:
                    idx = self._session.stimulus_presentations.stimulus_name.isin(self.stimulus_name) 
                if self.orientation != None:
                    idx = idx & (self._session.stimulus_presentations.orientation.isin(self.orientation))
                if self.temporal_frequency != None:
                    idx = idx & (self._session.stimulus_presentations.temporal_frequency.isin(self.temporal_frequency))
                if self.contrast != None:
                    idx = idx & (self._session.stimulus_presentations.contrast.isin(self.contrast))
                if self.stimulus_condition_id != None:
                    idx = idx & (self._session.stimulus_presentations['stimulus_condition_id'].isin(self.stimulus_condition_id))
                self.presentation_table = self._session.stimulus_presentations[idx]
            if self.merge_trials:
                self.presentation_table = combine_stimulus_presentations(
                    self.presentation_table, 
                    time_window=self.end_time - self.start_time + self.padding
                )
        else:
            # The trials are just random say 0.5 sec long sections in the session. 
            self.presentation_table = get_fake_stimulus_presentations(self._session.stimulus_presentations, time_window=0.5)
        
        # Get units
        if self.area == 'visual':
            self.selected_units = self._session.units[
                self._session.units['ecephys_structure_acronym'].isin(utils.VISUAL_AREA) &
                self._session.units['probe_description'].isin(self.selected_probes)]
        else:
            self.selected_units = self._session.units[
                self._session.units['probe_description'].isin(self.selected_probes)]
        self.unit_ids = self.selected_units.index.values

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

    def get_spike_table(self, selected_presentation_ids):
        """ Get spike times for selected trials.

        Args:
            selected_trials (array-like, optional): Indices of trials to get spikes for. 
                If None, gets spikes for all trials. Default: None.
                It's 0-indexed and not the id of the trial in the presentation_ids.

        Returns:
            pd.DataFrame: DataFrame containing spike times with columns:
                - stimulus_presentation_id (int): ID of the stimulus presentation
                - unit_id (int): ID of the unit that spiked
                - time_since_stimulus_presentation_onset (float): Time since stimulus onset in seconds
                Index is spike_time (float): Absolute time of spike in seconds
        """
        trial_time_window = [self.start_time - self.padding, self.end_time]
        presentation_start_times = np.array(self.presentation_table.loc[self.presentation_ids]['start_time'])
        presentation_end_times = np.array(self.presentation_table.loc[self.presentation_ids]['stop_time'])
        
        presentation_ids = []
        unit_ids = []
        spike_times = []

        for unit_id in self.unit_ids:
            unit_spike_times = self._session.spike_times[unit_id]

            for s, stimulus_presentation_id in enumerate(selected_presentation_ids):
                trial_start_time = presentation_start_times[s] + trial_time_window[0] 
                trial_end_time = presentation_start_times[s] + trial_time_window[1]

                trial_start_index = np.searchsorted(unit_spike_times, trial_start_time)
                trial_end_index = np.searchsorted(unit_spike_times, trial_end_time)

                trial_unit_spike_times = unit_spike_times[trial_start_index:trial_end_index]
                if len(trial_unit_spike_times) == 0:
                    continue

                unit_ids.append(np.zeros([trial_unit_spike_times.size]) + unit_id)
                presentation_ids.append(np.zeros([trial_unit_spike_times.size]) + stimulus_presentation_id)
                spike_times.append(trial_unit_spike_times)

        spike_df = pd.DataFrame({
            'stimulus_presentation_id': np.concatenate(presentation_ids).astype(int),
            'unit_id': np.concatenate(unit_ids).astype(int)
        }, index=pd.Index(np.concatenate(spike_times), name='spike_time'))

        onset_times = self.presentation_table.loc[self.presentation_ids]["start_time"]
        spikes_table = spike_df.join(onset_times, on=["stimulus_presentation_id"])
        spikes_table["time_since_stimulus_presentation_onset"] = spikes_table.index - spikes_table["start_time"]
        spikes_table.sort_values('spike_time', axis=0, inplace=True)
        spikes_table.drop(columns=["start_time"], inplace=True)
        return spikes_table

    def get_trial_metric_per_unit_per_trial(
        self, 
        selected_trials=None, # None for all trials. This is not the index of the trial in the presentation_ids.
        metric_type='spike_trains', 
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

        trial_time_window = [self.start_time - self.padding, self.end_time]
        if selected_trials is not None:
            selected_presentation_ids = self.presentation_ids[selected_trials]
        else:
            selected_presentation_ids = self.presentation_ids
        if dt is None:
            dt = 1/self.fps
        # spikes_table = self._session.trialwise_spike_times(
        #         self.presentation_ids, self.unit_ids, trial_time_window)
        spikes_table = self.get_spike_table(selected_presentation_ids=selected_presentation_ids)
        num_neurons = len(self.unit_ids)
        num_trials = len(self.presentation_ids)
        metric_table = pd.DataFrame(index=self.unit_ids, columns=selected_presentation_ids)
        metric_table.index.name = 'units'

        if metric_type == 'spike_trains':
            time_bins = np.linspace(
                    trial_time_window[0], trial_time_window[1],
                    int((trial_time_window[1] - trial_time_window[0]) / dt) + 1)

        for u, unit_id in enumerate(self.unit_ids):
            if verbose and (u % 40 == 0):
                print('neuron:', u)
            for s, stimulus_presentation_id in enumerate(selected_presentation_ids):
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
        return metric_table

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
            self.pupil_diam[:,i] = self.pupil_diam_xarray.sel(time = time_selection, method='nearest')
            self.mean_pupil_diam[i] = self.pupil_diam[:,i].mean()
            self.min_pupil_diam[i] = self.pupil_diam[:,i].min()
            self.max_pupil_diam[i] = self.pupil_diam[:,i].max()
        
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

