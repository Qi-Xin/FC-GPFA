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
from tqdm import tqdm
import torch
import socket


class BatchIterator:
    """Custom iterator for Allen_dataloader_multi_session."""
    def __init__(self, dataloader, split):
        self.dataloader = dataloader
        self.split = split
        self.current_batch_idx = 0  # Local batch index

    def __iter__(self):
        self.current_batch_idx = 0
        return self

    def __next__(self):
        if self.current_batch_idx >= len(getattr(self.dataloader, f"{self.split}_batches")):
            raise StopIteration
        batch = self.dataloader.get_batch(current_batch_idx=self.current_batch_idx,
                                          split=self.split)
        self.current_batch_idx += 1
        return batch
    
    def __len__(self):
        """Return the number of batches in the specified split."""
        return len(getattr(self.dataloader, f"{self.split}_batches"))


class Simple_dataloader_from_spikes():
    def __init__(self, 
                 spikes,  
                 npadding=0, 
                 train_ratio=0.7,
                 val_ratio=0.1,
                 batch_size=32,
                 verbose=True):
        """
        Input data is a list of 3D numpy array of shape (nt, nneuron, ntrial).
        It should be high resolution data with bin size of 1 ms or so. 
        """
        self.spikes = spikes
        self.npadding = npadding
        self.batch_size = batch_size
        self.nt, _, self.ntrial = self.spikes[0].shape
        self.nneuron_list = [spike.shape[1] for spike in self.spikes]
        self.npadding = npadding
        
        # Get tokenized spike trains by merging time intervals
        self.spikes_full = np.concatenate(self.spikes, axis=1)

        # Change from tnm to mtn 
        self.spikes_full = torch.tensor(self.spikes_full).float()
        
        # Splitting data into train and test sets
        indices = list(range(self.ntrial))
        utils.set_seed(1)
        np.random.shuffle(indices)
        split1, split2 = int(train_ratio*self.ntrial), int((train_ratio+val_ratio)*self.ntrial)
        self.train_indices = indices[:split1]
        self.val_indices = indices[split1:split2]
        self.test_indices = indices[split2:]

        # Create batches for each split
        self._create_batches('train', self.train_indices)
        self._create_batches('val', self.val_indices)
        self._create_batches('test', self.test_indices)

        # Initialize BatchIterators
        self.train_loader = BatchIterator(self, split='train')
        self.val_loader = BatchIterator(self, split='val')
        self.test_loader = BatchIterator(self, split='test')

        self.session_ids = ["0"]

    def _create_batches(self, split, indices):
        """Create batches for the given split"""
        n_trials = len(indices)
        n_batches = (n_trials + self.batch_size - 1) // self.batch_size
        
        # Shuffle indices if training split
        if split == 'train':
            np.random.shuffle(indices)
            
        # Create batches
        batches = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_trials)
            batch_indices = indices[start_idx:end_idx]
            batches.append(batch_indices)
            
        setattr(self, f'{split}_batches', batches)
        
    def get_batch(self, current_batch_idx, split):
        """Get a batch of data for the given split and batch index"""
        batch_indices = getattr(self, f'{split}_batches')[current_batch_idx]
        return {
            "spike_trains": self.spikes_full[:, :, batch_indices],
            "session_id": "0",
            "nneuron_list": self.nneuron_list,
            "batch_indices": batch_indices
        }
    
    def change_batch_size(self, new_batch_size, verbose=True):
        self.batch_size = new_batch_size
        self._create_batches('train', self.train_indices)
        self._create_batches('val', self.val_indices)
        self._create_batches('test', self.test_indices)


class Allen_dataloader_multi_session():
    def __init__(self, session_ids, verbose=True, **kwargs):
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
        self.train_ratio = kwargs.get('train_ratio', 0.7)
        self.val_ratio = kwargs.get('val_ratio', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.shuffle = kwargs.get('shuffle', True)
        self.common_kwargs = kwargs
        
        # Initialize session info
        self._initialize_sessions()
        
        # Split data into train/val/test
        self._split_data()
        
        # Initialize BatchIterators
        self.train_loader = BatchIterator(self, split='train')
        self.val_loader = BatchIterator(self, split='val')
        self.test_loader = BatchIterator(self, split='test')
        
        if verbose:
            print(f"Total sessions: {len(self.session_ids)}, "
                f"Batch size: {self.batch_size}, "
                f"Train set size: {len(self.train_loader)}, "
                f"Val set size: {len(self.val_loader)}, "
                f"Test set size: {len(self.test_loader)}")

    def _initialize_sessions(self):
        """Initialize metadata for all sessions"""
        self.sessions = {}
        self.total_trials = 0
        self.session_trial_counts = []
        self.session_trial_indices = []
        
        logger = logging.getLogger(__name__)
        logger.critical("Start loading data")

        for session_id in tqdm(self.session_ids):
            # Get trial count for this session
            self.sessions[session_id] = Allen_dataset(session_id=session_id, **self.common_kwargs)
            n_trials = len(self.sessions[session_id].presentation_ids)
            
            self.session_trial_counts.append(n_trials)
            self.session_trial_indices.append((self.total_trials, self.total_trials + n_trials))
            self.total_trials += n_trials

    def _split_data(self):
        """Split trials into train/val/test sets. 
        Keep all trials in a batch to be from the same session"""
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []
        for i, session_id in enumerate(self.session_ids):
            # Ensure all trials in a batch are from the same session
            # Also ensure trials under the same stimulus type are 
            # uniformly distributed in train/val/test
            condition2trials_dict = (
                self.sessions[session_id].get_condition2trials_dict()
            )
            train_trials = []
            val_trials = []
            test_trials = []
            trials_same_condition_list = list(condition2trials_dict.values())
            for i_trials_same_condition in range(len(trials_same_condition_list)):
                # Make local indices to global indices
                trials_same_condition = (
                    trials_same_condition_list[i_trials_same_condition].copy()
                )
                trials_same_condition = trials_same_condition + self.session_trial_indices[i][0]
                trials_same_condition = trials_same_condition.tolist()
                if self.shuffle:
                    np.random.shuffle(trials_same_condition)
                ntrials = len(trials_same_condition)
                if ntrials == 15:
                    train_trials += trials_same_condition[:9]
                    val_trials += trials_same_condition[9:11]
                    test_trials += trials_same_condition[11:]
                else:
                    train_size = int(ntrials*self.train_ratio)
                    val_size = int(ntrials*self.val_ratio)
                    train_trials += trials_same_condition[:train_size]
                    val_trials += trials_same_condition[train_size:train_size+val_size]
                    test_trials += trials_same_condition[train_size+val_size:]

            ### Making trials in a batch next to each other can shorten the search time 
            ### for faster converting spike times to spike trains.
            # if self.shuffle:
            #     np.random.shuffle(train_trials)
            train_trials = np.sort(train_trials)
            val_trials = np.sort(val_trials)
            test_trials = np.sort(test_trials)
            
            train_batches = self._create_batches(train_trials)
            val_batches = self._create_batches(val_trials)
            test_batches = self._create_batches(test_trials)
        
            self.train_batches += train_batches
            self.val_batches += val_batches
            self.test_batches += test_batches
        
        if self.shuffle:
            # not like all batches are from session 1 first, then all from session 2, etc.
            np.random.shuffle(self.train_batches)



    def _create_batches(self, indices):
        """Create batches from indices"""
        n_samples = len(indices)
        n_batches = n_samples // self.batch_size
        batches = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batches.append(np.array(indices[start_idx:end_idx]))
            
        # Handle remaining samples
        if n_samples % self.batch_size != 0:
            batches.append(np.array(indices[n_batches * self.batch_size:]))
            
        return batches

    def _get_session_for_trial(self, trial_idx):
        """Get session ID for a given trial index"""
        for i, (start, end) in enumerate(self.session_trial_indices):
            if start <= trial_idx < end:
                return self.session_ids[i], start
        raise ValueError(f"Invalid trial index: {trial_idx}")

    def _load_batch(self, batch_indices, include_behavior=False):
        """Load a batch of trials"""
        
        # Get session ID and local trial indices for each trial in the batch
        session_id, session_idx_start = self._get_session_for_trial(batch_indices[0])
        local_idx = batch_indices - session_idx_start
        current_session = self.sessions[session_id]
        batch_data = current_session.get_trial_spike_trains(selected_trials=local_idx)
        batch_data['spike_trains'] = torch.tensor(batch_data['spike_trains']).float()
        batch_data['session_id'] = str(session_id)
        batch_data['nneuron_list'] = current_session.nneuron_list
        
        if include_behavior:
            # Load behavior data
            pass
        
        return batch_data

    def get_batch(self, current_batch_idx, split, include_behavior=False):
        """Get next batch for specified split"""
        if split == 'train':
            batches = self.train_batches
        elif split == 'val':
            batches = self.val_batches
        elif split == 'test':
            batches = self.test_batches
        else:
            raise ValueError(f"Invalid split: {split}")

        batch = self._load_batch(batches[current_batch_idx], 
                                 include_behavior=include_behavior)
        return batch
    
    def change_batch_size(self, new_batch_size, verbose=True):
        self.batch_size = new_batch_size

        # Split data into train/val/test
        self._split_data()

        # Initialize BatchIterators
        self.train_loader = BatchIterator(self, split='train')
        self.val_loader = BatchIterator(self, split='val')
        self.test_loader = BatchIterator(self, split='test')
        
        if verbose:
            print(f"Total sessions: {len(self.session_ids)}, "
                f"Batch size: {self.batch_size}, "
                f"Train set size: {len(self.train_loader)}, "
                f"Val set size: {len(self.val_loader)}, "
                f"Test set size: {len(self.test_loader)}")


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
        self.session_id = kwargs.get('session_id', 791319847)
        self.selected_probes = kwargs.get('selected_probes', 'all')
        self.align_stimulus_onset = kwargs.get('align_stimulus_onset', True)
        self.merge_trials = kwargs.get('merge_trials', False)
        self.stimulus_name = kwargs.get('stimulus_name', "all")
        self.orientation = kwargs.get('orientation', None)
        self.temporal_frequency = kwargs.get('temporal_frequency', None)
        self.contrast = kwargs.get('contrast', None)
        self.stimulus_condition_id = kwargs.get('stimulus_condition_id', None)
        self.start_time = kwargs.get('start_time', 0)
        self.end_time = kwargs.get('end_time', 0.4)
        self.padding = kwargs.get('padding', 0.1)
        self.fps = kwargs.get('fps', 1e3)
        self.area = kwargs.get('area', 'cortex')

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
            hostname = socket.gethostname()
            if hostname[:8] == "ghidorah":
                path_prefix = '/home'
            elif hostname[:6] == "wright":
                path_prefix = '/home/export'
            elif hostname[:3] in ["n01", "n02", "n03"]:
                path_prefix = '/home/export'
            else:
                raise ValueError(f"Unknown host: {hostname}")
            self.manifest_path = os.path.join(path_prefix+'/qix/ecephys_cache_dir/', "manifest.json")
        elif sys.platform == 'win32' or 'darwin':
            self.manifest_path = os.path.join('D:/ecephys_cache_dir/', "manifest.json")
        else:
            raise ValueError("Undefined device!")

        self._cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        # V1._cache.get_session_table() to get the session_table
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
        self.presentation_table.reset_index(drop=True, inplace=True)

        # Get units
        self.nneuron_list = []
        self.unit_ids = []
        for probe in self.selected_probes:
            if self.area == 'cortex':
                selected_units = self._session.units[
                    self._session.units['ecephys_structure_acronym'].isin(utils.VISUAL_AREA) &
                    self._session.units['probe_description'].isin([probe])
                ]
            else:
                selected_units = self._session.units[
                    self._session.units['probe_description'].isin([probe])]
            self.nneuron_list.append(len(selected_units))
            self.unit_ids.append(selected_units.index.values)
        self.unit_ids = np.concatenate(self.unit_ids)

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
    
    # def get_spike_table(self, selected_presentation_ids):
    #     """Optimized spike table generation."""
    #     if type(selected_presentation_ids) is not np.ndarray:
    #         selected_presentation_ids = np.array(selected_presentation_ids)
    #     trial_time_window = [self.start_time - self.padding, self.end_time]
    #     presentation_start_times = np.array(self.presentation_table.loc[self.presentation_ids]['start_time'])

    #     spikes = []
    #     for unit_id in self.unit_ids:
    #         unit_spike_times = self._session.spike_times[unit_id]

    #         # Filter spikes once for all selected trials
    #         trial_start_times = presentation_start_times[selected_presentation_ids] + trial_time_window[0]
    #         trial_end_times = presentation_start_times[selected_presentation_ids] + trial_time_window[1]

    #         spike_indices = np.searchsorted(unit_spike_times, [trial_start_times.min(), trial_end_times.max()])
    #         filtered_spike_times = unit_spike_times[spike_indices[0]:spike_indices[1]]

    #         # Match spikes to trials
    #         trial_indices = np.searchsorted(trial_start_times, filtered_spike_times, side='right') - 1
    #         valid_mask = (filtered_spike_times >= trial_start_times[trial_indices]) & \
    #                      (filtered_spike_times <= trial_end_times[trial_indices])

    #         spikes.append(pd.DataFrame({
    #             'stimulus_presentation_id': selected_presentation_ids[trial_indices[valid_mask]],
    #             'unit_id': unit_id,
    #             'spike_time': filtered_spike_times[valid_mask],
    #         }))

    #     spike_df = pd.concat(spikes, ignore_index=True)
    #     spike_df['time_since_stimulus_presentation_onset'] = \
    #         spike_df['spike_time'] - self.presentation_table.loc[spike_df['stimulus_presentation_id'], 'start_time'].values
    #     spike_df.sort_values('spike_time', inplace=True)
    #     return spike_df
    
    def get_spike_table(self, selected_presentation_ids):
        """
        Return a spike dataframe that includes spikes from [start_time - padding, end_time]
        for the selected trials.
        """
        unit_ids = self.unit_ids
        trial_time_window = [self.start_time - self.padding, self.end_time]
        presentation_start_times = np.array(self.presentation_table.loc[self.presentation_ids]['start_time'])

        # Map from selected trial index to start and end times
        trial_start_times = presentation_start_times[selected_presentation_ids] + trial_time_window[0]
        trial_end_times = presentation_start_times[selected_presentation_ids] + trial_time_window[1]

        # Ensure trials are sorted so searchsorted behaves correctly
        sort_idx = np.argsort(trial_start_times)
        trial_start_times = trial_start_times[sort_idx]
        trial_end_times = trial_end_times[sort_idx]
        selected_presentation_ids = np.array(selected_presentation_ids)[sort_idx]

        all_spike_dfs = []

        for unit_id in unit_ids:
            unit_spike_times = self._session.spike_times[unit_id]

            # Global spike time window for efficiency
            global_start = trial_start_times.min()
            global_end = trial_end_times.max()
            spike_start_idx = np.searchsorted(unit_spike_times, global_start, side='left')
            spike_end_idx = np.searchsorted(unit_spike_times, global_end, side='right')
            filtered_spike_times = unit_spike_times[spike_start_idx:spike_end_idx]

            # Assign each spike to a trial (might be outside trial bounds)
            trial_indices = np.searchsorted(trial_start_times, filtered_spike_times, side='right') - 1

            # Guard against negative indices
            valid_mask = (trial_indices >= 0) & \
                        (filtered_spike_times >= trial_start_times[trial_indices]) & \
                        (filtered_spike_times <= trial_end_times[trial_indices])

            spike_df = pd.DataFrame({
                'stimulus_presentation_id': selected_presentation_ids[trial_indices[valid_mask]],
                'time_since_stimulus_presentation_onset':
                    filtered_spike_times[valid_mask] - trial_start_times[trial_indices[valid_mask]] - self.padding,
                'time': filtered_spike_times[valid_mask],
                'unit_id': unit_id
            })

            all_spike_dfs.append(spike_df)

        return pd.concat(all_spike_dfs, ignore_index=True)

    
    def get_trial_spike_trains(self, selected_trials=None, dt=None):
        """
        Compute spike trains as a 3D NumPy array of shape (nt, num_neurons, num_trials).
        
        Args:
            selected_trials (array-like, optional): Indices of selected trials. If None, all trials are used.
            dt (float, optional): Time bin width in seconds. If None, defaults to 1/self.fps.
            
        Returns:
            np.ndarray: Spike trains with shape (nt, num_neurons, num_trials).
        """
        # Set trial time window
        trial_time_window = [self.start_time - self.padding, self.end_time]
        selected_presentation_ids = (
            self.presentation_ids[selected_trials]
            if selected_trials is not None
            else self.presentation_ids
        )
        dt = dt if dt is not None else 1 / self.fps
        
        # Generate time bins
        time_bins = np.arange(trial_time_window[0], trial_time_window[1] + dt, dt)
        nt = len(time_bins) - 1  # Number of time bins
        num_neurons = len(self.unit_ids)
        num_trials = len(selected_presentation_ids)
        
        # Initialize 3D array for spike trains
        spike_train_array = np.zeros((nt, num_neurons, num_trials), dtype=int)

        # Precompute neuron and trial mappings
        neuron_idx_map = {unit_id: idx for idx, unit_id in enumerate(self.unit_ids)}
        trial_idx_map = {stim_id: idx for idx, stim_id in enumerate(selected_presentation_ids)}
        
        # Get spike table
        spikes_table = self.get_spike_table(selected_presentation_ids=selected_presentation_ids)
        
        # Group spikes by neuron and trial
        grouped_spikes = spikes_table.groupby(['unit_id', 'stimulus_presentation_id'])
        
        # Fill spike train array
        for (unit_id, stim_id), group in grouped_spikes:
            neuron_idx = neuron_idx_map[unit_id]
            trial_idx = trial_idx_map[stim_id]
            spike_times = group['time_since_stimulus_presentation_onset'].values
            spike_train_array[:, neuron_idx, trial_idx] = np.histogram(spike_times, bins=time_bins)[0]

        # Save results
        self.spike_train = spike_train_array
        self.trial_index_map = trial_idx_map
        self.neuron_index_map = neuron_idx_map

        return {"spike_trains": spike_train_array, 
                "presentation_ids": selected_presentation_ids, 
                "neuron_id": self.unit_ids}

    def get_condition2trials_dict(self):
        """Get trials for each condition"""
        # Create a new column for grouping
        self.presentation_table['group_key'] = self.presentation_table.apply(custom_group_key, axis=1)
        # Group by this new key and get indices
        self.condition2trials_dict = (
            self.presentation_table.groupby('group_key').apply(lambda x: x.index.values).to_dict()
        )
        return self.condition2trials_dict

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

def custom_group_key(row):
    if row['stimulus_name'] == 'drifting_gratings':
        return ('drifting_gratings', row['stimulus_condition_id'])
    else:
        return (row['stimulus_name'], None)