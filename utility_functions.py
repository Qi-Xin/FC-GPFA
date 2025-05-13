import numpy as np
import scipy
import matplotlib. pyplot as plt
from scipy import signal 

import networkx as nx
import pickle
import seaborn as sns
import scipy.interpolate
import copy
import pandas as pd
import scipy.signal
import torch
import random
import os

##### Useful information about Allen brain areas

'''
primary visual cortex (V1), 
lateromedial area (LM), 
laterointermediate area (LI), 
anterolateral area (AL), 
rostrolateral area (RL), 
anteromedial area (AM), 
posteromedial area (PM)
'''

"""
Total areas
'APN', 'LP', 'MB', 'DG', 'CA1', 'VISrl', nan, 'TH', 'LGd', 'CA3', 'VIS', 'CA2',
'ProS', 'VISp', 'POL', 'VISpm', 'PPT', 'OP', 'NOT', 'HPF', 'SUB', 'VISam', 'ZI',
'LGv', 'VISal', 'VISl', 'SGN', 'SCig', 'MGm', 'MGv', 'VPM', 'grey', 'Eth',
'VPL', 'IGL', 'PP', 'PIL', 'PO', 'VISmma', 'POST', 'SCop', 'SCsg', 'SCzo',
'SCiw', 'IntG', 'MGd', 'MRN', 'LD', 'VISmmp', 'CP', 'VISli', 'PRE', 'RPF', 'LT',
'PF', 'PoT', 'VL', 'RT'

Areas names from data:
units = cache.get_units()
units['ecephys_structure_acronym'].unique()


MB: Midbrain.
APN: Anterior Pretectal area. The pretectal area, or pretectum, is a midbrain
    structure composed of seven nuclei and comprises part of the subcortical
    visual system. 
PPT: Posterior pretectal nucleus
NOT: Nucleus of the optic tract.
AT: Anterior tegmental nucleus.
DT: Dorsal terminal nucleus of the accessory optic tract.
LT: Lateral terminal nucleus of the accessory optic tract.
SC: Superior colliculus
SCig: Superior colliculus, motor related, intermediate gray layer.
SCop: Superior colliculus, optic layer.
SCsg: Superior colliculus, superficial gray layer.
SCzo: Superior colliculus, zonal layer.
SCiw: Superior colliculus, motor related, intermediate white layer.
MRN: Midbrain reticular nucleus.
RPF: Retroparafascicular nucleus.
OP: Olivary pretectal nucleus.


HPF: Hippocampal formation.
CA CA1 CA2 CA3
DG: The dentate gyrus (DG), is part of the hippocampal formation in the temporal
    lobe of the brain that includes the hippocampus, and the subiculum.
ProS: Prosubiculum. Hippocapal formation.
SUB: Subiculum. 
POST: Postsubiculum.
PRE: Presubiculum.


VIS: 
VISam: Anteromedial area .
VISpm: Posteromedial area.
VISp: Primary visual cortex.
VISl: Lateromedial area.
VISal: Anterolateral area.
VISrl: Rostrolateral area.
VISli: Laterointermediate area.
VISmmp: Mediomedial posterior visual area.
VISmma: Mediomedial anterior visual area.

TH: Thalamus.
LGd: The dorsolateral geniculate nucleus is the main division of the lateral
    geniculate body. The majority of input to the dLGN comes from the retina. It
    is laminated and shows retinotopic organization
LGv: The ventrolateral geniculate nucleus has been found to be relatively large
    in several species such as lizards, rodents, cows, cats, and primates
IGL: Intergeniculate leaflet of the lateral geniculate complex. A distinctive
    subdivision of the lateral geniculate complex in some rodents that
    participates in the regulation of circadian rhythm.
POL: Posterior limiting nucleus of the thalamus.
PO: Posterior complex of the thalamus.
SGN: Suprageniculate nucleus.
MGm: Medial geniculate complex, medial part.
MGv: Medial geniculate complex, ventral part.
MGd: Medial geniculate complex, dorsal part.
VPM: Ventral posteromedial nucleus of the thalamus.
Eth: Ethmoid nucleus of the thalamus.
VPL: Ventral posterolateral nucleus of the thalamus.
PP: Peripeduncular nucleus.
PIL: Posterior intralaminar thalamic nucleus.
IntG: Intermediate geniculate nucleus.
LD: Lateral dorsal nucleus of thalamus.
RT: Reticular nucleus of the thalamus.
PF: Parafascicular nucleus.
PoT: Posterior triangular thalamic nucleus.
LP: Lateral posterior nucleus of the thalamus.


Grey matter.
grey: Grey matter.
ZI: The zona incerta is a horizontally elongated region of gray matter in the
    subthalamus below the thalamus.

Striutum.
CP: Caudoputamen.

Ventricle.
VL: lateral ventricle.

The brain region names come from
Allen Institute 2017 - Allen Mouse Common Coordinate Framework v3.pdf

Also check: https://allensdk.readthedocs.io/en/v2.1.0/_static/
    examples/nb/ecephys_data_access.html

name: the probe name is assigned based on the location of the probe on the 
recording rig. This is useful to keep in mind because probes with the same name 
are always targeted to the same cortical region and enter the brain from the 
same angle (probeA = AM, probeB = PM, probeC = V1, probeD = LM, probeE = AL, 
probeF = RL). However, the targeting is not always accurate, so the actual 
recorded region may be different.
"""

PROBE_CORRESPONDING = {'probeA':'AM', 'probeB':'PM', 'probeC':'V1', 'probeD':'LM', 'probeE':'AL', 'probeF':'RL'}
PROBE_CORRESPONDING_INVERSE = {'AM': 'probeA', 'PM': 'probeB', 'V1': 'probeC', 'LM': 'probeD', 'AL':'probeE', 'RL':'probeF'}
##### Colors for plot
# Orange.
MIDBRAIN = ['APN', 'MB', 'AT', 'DT', 'PPT', 'NOT', 'LT', 'OP',
            'SC', 'SCig', 'SCiw', 'SCzo', 'SCsg', 'SCop', 'MRN', 'RPF']

# Blue.
HIPPOCAMPUS_AREA = ['HPF', 'CA', 'DG', 'CA1', 'CA2', 'CA3', 'ProS', 'SUB',
                    'POST', 'PRE']

# Red
THALAMUS_AREA = ['TH', 'LGd', 'LGv', 'LP', 'IGL', 'PO', 'POL', 'SGN',
                 'MGv', 'MGm', 'MGd', 'VPM', 'Eth', 'VPL', 'PP', 'PIL', 'IntG',
                 'LD', 'RT', 'PF', 'PoT']

# Green.
VISUAL_AREA = ['VIS', 'VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl',
               'VISmmp', 'VISmma', 'VISli']

from matplotlib import rcParams, rcParamsDefault
def use_pdf_plot(**kwargs):
    sns.reset_orig()
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIG_SIZE = 7
    # rcParams['font.size'] = SMALL_SIZE
    rcParams['lines.linewidth'] = 1
    rcParams['axes.linewidth'] = 0.8  # Adjust as needed.
    rcParams['axes.labelsize'] = SMALL_SIZE # axis label
    rcParams['axes.titlesize'] = MEDIUM_SIZE # Title
    rcParams['figure.titlesize'] = MEDIUM_SIZE # Suptitle
    rcParams['legend.fontsize'] = SMALL_SIZE # legend
    rcParams['xtick.labelsize'] = SMALL_SIZE # tick label
    rcParams['ytick.labelsize'] = SMALL_SIZE # tick label
    rcParams['xtick.major.size'] = 1.5  # length of x-axis major ticks
    rcParams['ytick.major.size'] = 1.5  # length of y-axis major ticks
    rcParams['xtick.major.pad'] = 1  # Adjust as needed
    rcParams['ytick.major.pad'] = 1  # Adjust as needed

    for key, value in kwargs.items():
        rcParams[key] = value
        
    rcParams['xtick.major.width'] = rcParams['axes.linewidth']
    rcParams['ytick.major.width'] = rcParams['axes.linewidth']

def use_default_plot():
    sns.reset_orig()
    rcParams.update(rcParamsDefault)
    
def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # CPU
    
    # For GPU operations
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (important for multi-GPU setups)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def change_temporal_resolution(spikes, num_merge):
    return [change_temporal_resolution_single(spikes[i], num_merge) for i in range(len(spikes))]

def change_temporal_resolution_single(spike_train, num_merge):
    nt, nneuron, ntrial = spike_train.shape
    if isinstance(spike_train, torch.Tensor):
        new_spike_train = torch.zeros((nt//num_merge, nneuron, ntrial), device=spike_train.device)
        for t in range(nt//num_merge):
            new_spike_train[t,:,:] = torch.sum(spike_train[t*num_merge:(t+1)*num_merge,:,:], dim=0)
    else:
        new_spike_train = np.zeros((nt//num_merge, nneuron, ntrial))
        for t in range(nt//num_merge):
            new_spike_train[t,:,:] = np.sum(spike_train[t*num_merge:(t+1)*num_merge,:,:], axis=0)
    return new_spike_train

def second_order_diff_matrix(T):
    D = np.diag(np.ones(T - 1), k=-1) \
      + np.diag(-2 * np.ones(T), k=0) \
      + np.diag(np.ones(T - 1), k=1)
    return D[1:-1,:]  # shape: (T-2, T)

def color_by_brain_area(ccf_structure, colortype='normal'):
    """Assign a color for a brain area."""
    if ccf_structure in VISUAL_AREA:
        color = 'tab:green'
        if colortype == 'dark':
            color = 'darkgreen'
        elif colortype =='light':
            color = 'lime'
        elif colortype =='rgby':
            color = [0.30196078, 0.68627451, 0.29019608, 1.]
    elif ccf_structure in HIPPOCAMPUS_AREA:
        color = 'tab:blue'
        if colortype == 'dark':
            color = 'darkblue'
        elif colortype =='light':
            color = 'lightblue'
        elif colortype =='rgby':
            color = [0.21568627, 0.49411765, 0.72156863, 1.]
    elif ccf_structure in THALAMUS_AREA:
        color = 'tab:red'
        if colortype == 'dark':
            color = 'darkred'
        elif colortype =='light':
            color = 'lightcoral'
        elif colortype =='rgby':
            color = [0.89411765, 0.10196078, 0.10980392, 1.]
    elif ccf_structure in MIDBRAIN:
        color = 'tab:orange'
        if colortype == 'dark':
            color = 'darkorange'
        elif colortype =='light':
            color = 'gold'
        elif colortype =='rgby':
            color = [1., 0.49803922, 0.,1.]
    else:
        color = 'tab:gray'
        if colortype == 'dark':
            color = 'dimgray'
        elif colortype =='light':
            color = 'lightgray'
        elif colortype =='rgby':
            color = [.5, .5, .5, 1.0]
    return color

##### Turn np.bool array to np.int array
def get_index_array(select_trials):
    return np.where(select_trials==True)[0]

##### Smoothing
def kernel_smoothing(raw, std, window=None):
    if window is None:
        window = int(2*std)
    smoothed = np.zeros_like(raw).astype(float)
    for icol in range(raw.shape[1]):
        hrly = pd.Series(raw[:,icol])
        smoothed[:,icol] = hrly.rolling(window=window, win_type='gaussian', center=True, min_periods=1).mean(std=std)
    return smoothed

##### Spike trains
def bin_spike_times(
        spike_times,
        bin_width,
        len_trial):
    """Convert spike times to spike bins, spike times list to spike bins matrix.

    spike times outside the time range will not be counted in the bin at the
    end. A time bin is left-closed right-open [t, t+delta).
    e.g. t = [0,1,2,3,4], y = [0, 0.1, 0.2, 1.1, 5, 6]
    output: [3, 1, 0, 0]

    Args:
        spike_times: The format can be list, np.ndarray.
    """
    bins = np.arange(0, len_trial+bin_width, bin_width)

    if len(spike_times) == 0:
        return np.zeros(len(bins)-1), bins[:-1]

    # multiple spike_times.
    elif isinstance(spike_times[0], list) or isinstance(spike_times[0], np.ndarray):
        num_trials = len(spike_times)
        num_bins = len(bins) - 1
        spike_hist = np.zeros((num_trials, num_bins))
        for r in range(num_trials):
            spike_hist[r], _ = np.histogram(spike_times[r], bins)

    # single spike_times.
    else:
        spike_hist, _ = np.histogram(spike_times, bins)
    return spike_hist, bins[:-1]

def pooling_pop(membership, condition_ids, dataset, probe_name, group_id, use_all=False):
    """
    'spike train' is now a numpy array
    """
    spike_train = dataset.spike_train
    condition_list = dataset.presentation_table['stimulus_condition_id']
    nt, ntrial = spike_train.shape[0], spike_train.shape[2]
    pooled_spike_train = np.zeros((nt, ntrial))
    
    for itrial in range(ntrial):
        try:
            trial = dataset.trial_index_map(itrial)
            current_condition = condition_list.loc[trial]
            current_membership = membership[np.where(condition_ids==current_condition)[0][0]]
            idx = current_membership[(current_membership['probe']==probe_name) \
                & (current_membership['group_id']==group_id)].index.values
            done = True
        except:
            done = False
        if use_all==True or done==False or idx.sum() == 0:
            # if don't get any group information in 'membership', just use all neurons 
            idx = dataset._session.units[
                dataset._session.units['ecephys_structure_acronym'].isin(VISUAL_AREA) &
                dataset._session.units['probe_description'].isin([probe_name])].index.values
            # idx = dataset._session.units[dataset._session.units['probe_description']==probe_name]
        local_neuron_idx = []
        for id in idx:
            local_neuron_idx.append(dataset.neuron_index_map[id])
        new_df =  spike_train[:, local_neuron_idx, :]
        for iunit in range(new_df.shape[1]):
            pooled_spike_train[:,itrial] += new_df[:, iunit, itrial]
    return pooled_spike_train
    

##### Filter plot
def plot_ci(input, x=None, label=None, color='b', exp=False, center=0, linewidth=1):
    y = input[0]
    if center != 0:
        center = y[0]
    ci = input[1]
    if x is None:
        x = np.arange(len(y))
    if exp:
        plt.plot(x, np.exp(y.squeeze()),label=label, color=color, linewidth=linewidth)
        plt.fill_between(x, np.exp((y-2*ci)), np.exp((y+2*ci)), color=color, alpha=.3)
    else:
        plt.plot(x, y.squeeze()-center,label=label, color=color, linewidth=linewidth)
        plt.fill_between(x, (y-2*ci)-center, (y+2*ci)-center, color=color, alpha=.3)


def plot_filter(basis, coef, se, label=None, color='b', exp=False, linewidth=1):
    x = np.arange(basis.shape[0])
    y = (basis@coef[:,np.newaxis]).squeeze()
    ci = (basis@se[:,np.newaxis]).squeeze()
    if exp:
        plt.plot(x, np.exp(y.squeeze()),label=label, color=color, linewidth=linewidth)
        plt.fill_between(x, np.exp((y-ci)), np.exp((y+ci)), color=color, alpha=.3)
    else:
        plt.plot(x, y.squeeze(),label=label, color=color, linewidth=linewidth)
        plt.fill_between(x, (y-2*ci), (y+2*ci), color=color, alpha=.3)
    # print(f"peaks are:{scipy.signal.find_peaks(y)}")
    
def plot_filter_output(basis, coef, se, label=None, color='b',exp=False):
    x = np.arange(basis.shape[0])
    y = (basis@coef[:,np.newaxis]).squeeze()
    ci = (basis@se[:,np.newaxis]).squeeze()
    if exp:
        plt.plot(x, np.exp(y.squeeze()),label=label, color=color)
        plt.fill_between(x, np.exp((y-ci)), np.exp((y+ci)), color=color, alpha=.3)
    else:
        plt.plot(x, y.squeeze(),label=label, color=color)
        plt.fill_between(x, (y-2*ci), (y+2*ci), color=color, alpha=.3)
    # print(f"peaks are:{scipy.signal.find_peaks(y)}")
    
##### PSTH plot
def plot_PSTH():
    pass


##### CCG related
def cross_corr(
        y1,
        y2,
        index_range=None,
        type='max'):
    """Calculates the cross correlation and lags with normalization.

    The definition of the discrete cross correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html
    The `y1` takes the first place, and the `y2` takes the second place. So when
    lag is negtive, it means the `log_lmbd` is on the left of `spike_trains`.

    Args:
        index_range: two entries list. [min_index, max_index]. If the index_range is
                beyond the range of the array, it will
                automatically be clipped to the bounds.
        type:
                'max': single max value.
                'full': get the whole correlation and corresponding lags.

    Returns:
        max_corr: Maximum correlation without normalization.
        lag: The lag in terms of the index.
    """

    # plt.figure()
    # plt.plot(y1, '.')
    # plt.plot(y2, '.')
    # plt.show()

    if len(y1) != len(y2):
        raise ValueError('The lengths of the inputs should be the same.')

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = signal.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = signal.correlate(
            np.ones(len(y1)), np.ones(len(y1)), mode='same')
    if y1_auto_corr != 0 and y2_auto_corr != 0:
        corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    if index_range is None and type == 'max':
        max_corr = np.max(corr)
        argmax_corr = np.argmax(corr)
        return max_corr, argmax_corr - shift
    elif index_range is None and type == 'full':
        return corr, np.arange(len(corr)) - shift

    index_range = np.array(index_range).astype(int)
    shifted_index_range = index_range + shift
    if index_range[0] + shift < 0:
        index_range[0] -= index_range[0] + shift
        shifted_index_range[0] = 0
    if index_range[1] + shift >= len(y1):
        index_range[1] -= index_range[1] + shift - len(y1) + 1
        shifted_index_range[1] = len(y1) - 1

    index_range_mask = np.array(
            range(index_range[0], index_range[1] + 1))
    shifted_index_range_mask = np.array(
            range(shifted_index_range[0], shifted_index_range[1] + 1))

    if type == 'max':
        max_corr = np.max(corr[shifted_index_range_mask])
        argmax_corr = np.argmax(corr[shifted_index_range_mask])
        lag = index_range_mask[argmax_corr]
        return max_corr, lag
    elif type == 'full':
        return corr[shifted_index_range_mask], index_range_mask

def cross_prod(
        y1,
        y2,
        index_range=None):
    """Calculates the cross correlation and lags without normalization.

    Args:
        index_range: two entries list. [min_index, max_index]. If the index_range is
                beyond the range of the array, it will
                automatically be clipped to the bounds.
    """
    if y1.shape != y2.shape:
        raise ValueError('The lengths of the inputs should be the same.')
    if len(y1.shape) == 1:
        num_bins = len(y1)
    elif len(y1.shape) == 2:
        num_bins = y1.shape[1]

    corr = scipy.signal.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = scipy.signal.correlate(
            np.ones(num_bins), np.ones(num_bins), mode='same')
    corr = corr / unbiased_sample_size
    shift = num_bins // 2

    if index_range is None:
        return corr, np.arange(num_bins) - shift

    index_range = np.array(index_range).astype(int)
    shifted_index_range = index_range + shift
    index_range_mask = np.array(
            range(index_range[0], index_range[1] + 1))
    shifted_index_range_mask = np.array(
            range(shifted_index_range[0], shifted_index_range[1] + 1))
    return corr[shifted_index_range_mask], index_range_mask

def array_shift(x, shift, zero_pad=False):
    """Shift the array.

    Args:
        shift: Negtive to shift left, positive to shift right.
    """
    x = np.array(x)
    shift = np.array(shift)
    if len(x.shape) > 2:
        raise ValueError('x can only be an array of a matrix.')

    # If `shift` is a scalar.
    if len(shift.shape) == 0:
        # If x is 1D array, shift along axis=0, if x is 2D matrix, shift along rows.
        x = np.roll(x, shift, axis=len(x.shape)-1)
        # pad zeros to the new positions.
        if zero_pad and len(x.shape) == 1 and shift > 0:
            x[:shift] = 0
        elif zero_pad and len(x.shape) == 1 and shift < 0:
            x[shift:] = 0
        elif zero_pad and len(x.shape) > 1 and shift > 0:
            x[:, :shift] = 0
        elif zero_pad and len(x.shape) > 1 and shift > 0:
            x[:, shift:] = 0
    # Shift matrix rows independently.
    elif len(shift.shape) == 1: 
        if len(shift) != x.shape[0]:
            raise ValueError('length of shift should be equal to rows of x.')
        for row, s in enumerate(shift):
            x[row] = np.roll(x[row], s)
            # pad zeros to the new positions.
            if zero_pad and s > 0:
                x[row, :s] = 0
            elif zero_pad and s < 0:
                x[row, s:] = 0
    else:
        raise ValueError('shift can be a scalar or a vector for each row in x.')
    return x

##### Correlation calculation

def fisher_transform(rho):
    """Fisher transformation for correlation.

    z = 0.5 * log((1 + rho) / (1 - rho))
    """
    rho = np.array(rho)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    return z

def marginal_corr_from_cov(cov):
    """Calculates marginal correlation matrix from covariance matrix.

    Args:
        cov: N x N matrix.
    """
    cov_diag_sqrt = np.sqrt(np.diag(cov))
    corr = cov / np.outer(cov_diag_sqrt, cov_diag_sqrt)

    return corr

def partial_corr_from_cov(cov):
    """Calculates partial correlation matrix from covariance matrix.

    Args:
        cov: N x N matrix.
    """
    theta = np.linalg.inv(cov)
    theta_diag_sqrt = np.sqrt(np.diag(theta))
    corr = - theta / np.outer(theta_diag_sqrt, theta_diag_sqrt)

    return corr

def xcorr(x, y,verbose=False):
    """Cross correlation coefficient.

    The lag centers at 0 if two arrays have equal length.

    References:
    https://www.mathworks.com/help/signal/ug/
        confidence-intervals-for-sample-autocorrelation.html
    """
    length = len(x)
    x = x - np.mean(x)
    y = y - np.mean(y)
    sigma = np.sqrt(np.dot(x, x) * np.dot(y, y))
    xcorr = np.correlate(x, y, mode='same') / sigma
    lag = np.arange(length) - length // 2

    # 95% CI, 0.025 on each side.
    alpha = scipy.stats.norm.ppf(0.975)
    CI_level = alpha / np.sqrt(length)

    if verbose:
        plt.figure()
        plt.plot(lag, xcorr)
        plt.axhline(y=CI_level, ls=':')
        plt.axhline(y=-CI_level, ls=':')

    return lag, xcorr, CI_level



##### Network plot
def plot_networkx_graph(G):
    """Plot networkx graph."""
    if nx.is_directed(G):
        print('Directed')
        directed = True
    else:
        print('Un-directed')
        directed = False
        # cliques = nx.find_cliques(G)
        # print(list(cliques))

    print(f'num_nodes {G.number_of_nodes()}    num_edges {G.number_of_edges()}')
    plt.figure(figsize=[11, 4])
    plt.subplot(121)
    # pos = nx.spring_layout(graph, scale=2)
    # pos = nx.drawing.random_layout(graph)
    pos=nx.circular_layout(G)
    # nx.draw(G, pos=pos)

    if len(nx.get_node_attributes(G, 'weight')) > 0:
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights,
                        width=2, edge_cmap=plt.cm.jet)
    else:
        nx.draw(G, pos, node_color='b', width=2, edge_cmap=plt.cm.jet)
    plt.subplot(122)
    adj_mat = nx.to_numpy_matrix(G)
    sns.heatmap(adj_mat)
    plt.show()

def plot_networkx_adj(G):
    """Plot networkx graph."""
    if nx.is_directed(G):
        print('Directed')
        directed = True
    else:
        print('Un-directed')
        directed = False
        cliques = nx.find_cliques(G)

    print(f'num_nodes {G.number_of_nodes()}    num_edges {G.number_of_edges()}')
    plt.figure(figsize=[5, 4])
    adj_mat = nx.to_numpy_matrix(G)
    sns.heatmap(adj_mat)
    plt.show()





##### Frequency domain analysis 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butterworth_bandpass_filter(x, lowcut, highcut, fs, order=4):
    """Butter bandpass filter.

    Args:
        x: Input signal.
        fs: Sampling frequency.
        order: The order of the Butterworth filter.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, x)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def low_pass_filter(raw_signal, fs, cutoff=100.):
    b, a = butter_lowpass(cutoff, fs, order=3)
    signal_filt = signal.filtfilt(b, a, raw_signal, axis=1)
    return signal_filt

def get_phase(raw_signal, fs, lowcut=8., highcut=12., npadding=0):
    """Get instantaneous phase of a time series / multiple time series

    Args:
        raw_signal (np array): (nx,nt)
        fs (float): sampling rate
        lowcut (float, optional): lower threshold. Defaults to 8..
        hightcut (float, optional): higher threshold. Defaults to 12..

    Returns:
        np array: instantaneous phase at each time point
    """
    only_one_row = False
    if raw_signal.ndim == 1:
        only_one_row = True
        raw_signal = raw_signal[None,:]
    b, a = butter_bandpass(lowcut, highcut, fs=fs, order=3)
    signal_filt = signal.filtfilt(b, a, raw_signal, axis=1)
    
    analytic_signal = signal.hilbert(signal_filt,axis=1)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    # instantaneous_phase_conti = np.unwrap(instantaneous_phase)
    # instantaneous_frequency = (np.diff(instantaneous_phase_conti) /
    #                         (2.0*np.pi) * fs)
    power = amplitude_envelope.mean(axis=1)
    instantaneous_power = amplitude_envelope
    instantaneous_phase = instantaneous_phase[:,npadding:-npadding]
    instantaneous_power = instantaneous_power[:,npadding:-npadding]
    if only_one_row:
        instantaneous_phase = instantaneous_phase.squeeze()
        instantaneous_power = instantaneous_power.squeeze()
        power = power.squeeze().item()
    return instantaneous_phase, instantaneous_power

def get_power_phase(data, npadding, lowcut, highcut):
    data = copy.copy(data)
    nx, nt, ntrial = data.shape
    phase = np.zeros((nx, nt-2*npadding, ntrial))
    power = np.zeros((nx, nt-2*npadding, ntrial))

    for itrial in range(ntrial):
        raw_signal = data[:,:,itrial]
        instantaneous_phase, instantaneous_power = get_phase(raw_signal, 500, lowcut=lowcut, highcut=highcut, 
                                                                   npadding=npadding)
        phase[:,:,itrial] = instantaneous_phase
        power[:,:,itrial] = instantaneous_power
    return phase, power

def get_power_spectrum(
        x,
        fs,
        output_figure_path=None,
        show_figure=False):
    """Gets the power spectrum."""
    num_per_segment = 2 ** 12
    f, Pxx_den = signal.welch(x, fs, nperseg=1024)

    plt.figure()
    # plt.semilogy(f, Pxx_den)
    plt.plot(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    if output_figure_path:
        plt.savefig(output_figure_path)
        print('Save figure to: ', output_figure_path)
    if show_figure:
        plt.show()
    plt.close()

    return f, Pxx_den

def get_spectrogram(
    x,
    fs,
    time_offset=0,
    output_figure_path=None,
    show_figure=True):
    """Get the spectrum along time.

    Args:
        x: Input signal.
        fs: Sampling frequency.
    """
    # `nfft` > `nperseg` means apply zero padding to make the spectrum look
    # smoother, but it does not provide extra informaiton. `noverlap` is the
    # overlap between adjacent sliding windows, the larger, the more overlap.
    # num_per_segment = 2 ** 8
    num_per_segment = 250
    f, t, Sxx = signal.spectrogram(
        x, fs,
        nperseg=num_per_segment,
        noverlap=num_per_segment // 50 * 49,
        nfft=num_per_segment * 8)
    t = np.array(t) + time_offset
    # Used to debug the positions of the sliding window.
    # print(np.array(t))

    plt.figure(figsize=(10, 8))
    # plt.pcolormesh(t, f, np.log(Sxx))  # The log scale plot.
    # plt.pcolormesh(t, f, Sxx, vmax=np.max(Sxx) / 10)
    plt.pcolormesh(t, f, Sxx, vmax=200)
    plt.ylim(0, 100)
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    if output_figure_path:
        plt.savefig(output_figure_path)
        print('Save figure to: ', output_figure_path)
    if show_figure:
        plt.show()
    plt.close()


##### LFP/CSD utils

def check_and_get_size(lfp):
    if lfp.ndim == 3:
        nx, nt, ntrial = lfp.shape
    elif lfp.ndim == 2:
        nx, nt = lfp.shape
        ntrial = 1
        lfp = lfp[:,:,None]     # Convert single trial LFP data to three dimensional
    else:
        raise AssertionError("LFP data must be either two dimensional (single trial) or three dimensional (multiple trials)!")
    return nx, nt, ntrial, lfp

def moving_average_single_row(x, pooling_size=1, moving_size=1):
    assert np.mod(moving_size,2) == 1, "Moving average kernel width must be an odd number!"
    assert np.mod(len(x),pooling_size) == 0, "Pooling parameter must be exactly divisible towards length!"
    """ Doing pooling """
    x_temp = x.reshape(-1,pooling_size)
    x_temp = x_temp.mean(axis=1)
    """ Doing moving average """
    weight_correct = np.convolve(np.ones(len(x_temp)), np.ones(moving_size), 'same') / moving_size
    x_smooth = np.convolve(x_temp, np.ones(moving_size), 'same') / moving_size  /weight_correct  # 
    return x_smooth

def moving_average(lfp, pooling_size=1, moving_size=1):
    nx, nt, ntrial, lfp = check_and_get_size(lfp)
    lfp_smooth = np.zeros((nx, nt, ntrial))
    for itrial in range(ntrial):
        for i in range(nx):
            temp = moving_average_single_row(lfp[i, :, itrial], pooling_size, moving_size)
            lfp_smooth[None, :, None] = temp[None, :, None]
    return lfp_smooth
    
def pooling(data, merge):
    new_data = np.zeros((data.shape[0], int(data.shape[1]/merge), data.shape[2]))
    for i in range(merge):
        new_data += data[:, i::merge, :]
    new_data = new_data/merge
    return new_data

def normalize(x):
    return x/np.max(np.abs(x), axis=(0, 1))

def normalize_var(x):
    return x/np.std(x)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Add correct label for each figure
def add_label_csd(the_title,the_yticks):
    yticks_num = 5
    #plt.colorbar()
    plt.yticks( np.linspace(0,the_yticks.shape[0],yticks_num),
               np.linspace(the_yticks.min(),the_yticks.max(),yticks_num).astype(int) )
    plt.xlabel('Time (frame)')
    plt.ylabel('Depth (micron)')
    plt.gca().set_title(the_title)
    plt.gca().xaxis.tick_bottom()

def plot_im (arr , v1 , v2):
    plt.xticks([]) 
    plt.yticks([])
    p = plt. imshow (arr , vmin =-np. nanmax (np. abs ( arr )), vmax =np. nanmax (np. abs ( arr )),
                     cmap='bwr', extent =[ np. min (v1), np. max (v1), np. max (v2), np. min (v2)])
    return p 

def plot_im_new (arr,v1,v2,vmin,vmax,yticks):
    if yticks == 0:
        plt.yticks([])
    p = plt. imshow (arr , vmin = vmin, vmax = vmax,
                     cmap='bwr',extent =[ np. min (v1), np. max (v1), np. max (v2), np. min (v2)])
    return p 

def sort_grid (x):
    xsrt = x[x[:, 1]. argsort ()]
    xsrt = xsrt [ xsrt [:, 0]. argsort ( kind ='mergesort ')]
    return xsrt

def expand_grid (x1 ,x2):
    """
    Creates ( len (x1)* len (x2), 2) array of points from two vectors .
    : param x1: vector 1, (nx1 , 1)
    : param x2: vector 2, (nx2 , 1)
    : return : ( nx1 *nx2 , 2) points
    """
    lc = [(a, b) for a in x1 for b in x2]
    return np. squeeze (np. array (lc))

def reduce_grid (x):
    """
    Undoes expand_grid to take (nx , 2) array to two vectors containing
    unique values of each col .
    : param x: (nx , 2) points
    : return : x1 , x2 each vectors
    """
    x1 = np. unique (x [: ,0])
    x2 = np. unique (x [: ,1])
    return x1 , x2

def mykron (A, B):
    """
    Efficient Kronecker product .
    """
    a1 , a2 = A. shape
    b1 , b2 = B. shape
    C = np. reshape (A[:, np. newaxis , :, np. newaxis ] * B[np. newaxis , :, np. newaxis , :], (a1*b1 , a2*b2))
    return C

def comp_eig_D (Ks , Kt , sig2n ):
    """
    Computes eigvecs and diagonal D for inversion of kron (Ks , Kt) + sig2n*I
    : param Ks: spatial covariance
    : param Kt: temporal covariance
    : param sig2n : noise variance
    : return : eigvec (Ks), eigvec (Kt), Dvec
    """
    nx = Ks. shape [0]
    nt = Kt. shape [0]
    evals_t , evec_t = scipy . linalg . eigh (Kt)
    evals_s , evec_s = scipy . linalg . eigh (Ks)
    Dvec = np. repeat ( evals_s , nt) * np. tile ( evals_t , nx) + sig2n *np. ones(nx*nt)
    return evec_s , evec_t , Dvec

def has_key_word(name, key_word_list):
    return any(key_word in name for key_word in key_word_list)


def loss_function_per_trial(log_firing_rate, spikes):
    # nt x nneuron x ntrial
    poisson_loss = (torch.exp(log_firing_rate) - spikes * log_firing_rate).mean(axis=(0,1))
    return poisson_loss


def find_middle_fit_trial(firing_rate_test, spike_train_test, left_and_right_trials=0):
    # nt x nneuron x ntrial
    loss_per_trial = loss_function_per_trial(
        firing_rate_test, 
        spike_train_test,
    )

    middle_fit_trial_idx = torch.argsort(loss_per_trial)[
        len(loss_per_trial)//2 - left_and_right_trials:
        len(loss_per_trial)//2 + left_and_right_trials+1
    ].numpy()
    return middle_fit_trial_idx


def centralize_factor(factor):
    # nt x nfactor
    return factor - factor.mean(axis=0)


def plot_single_factor_loading_horizontal(ax, gt, ft, title="", sort=True, neuron_to_plot_in_red=None):
    """
    Horizontal version: X = loading value, Y = neuron.
    Ground truth = circles, Fitted = squares.
    
    Parameters:
        ax: matplotlib Axes
        gt: np.ndarray of shape (n_neurons, n_factors) or None for ground truth
        ft: np.ndarray of shape (n_neurons, n_factors) for fitted values
        title: title
        sort: whether to sort neurons by ground truth (only if gt is provided)
    """

    neurons = np.arange(len(ft))
    
    if gt is not None:
        if sort:
            sort_idx = np.argsort(gt)
            gt = gt[sort_idx]
            ft = ft[sort_idx]
        
        # Plot ground truth
        ax.scatter(gt, neurons, marker='o', s=3, label='Ground truth', edgecolors='none', color='tab:green', alpha=0.5)
        # Plot fitted values
        ax.scatter(ft, neurons, marker='s', s=3, label='Fitted', edgecolors='none', color='tab:blue', alpha=0.8)
    else:
        colors = plt.cm.tab10.colors
        if ft.ndim == 1:
            ft = ft[:,None]
        if sort:
            sort_idx = np.argsort(ft[:,0])
            ft = ft[sort_idx, :]
            if neuron_to_plot_in_red is not None:
                neuron_to_plot_in_red = np.where(np.isin(sort_idx, neuron_to_plot_in_red))[0]
        
        mask = np.ones(len(neurons), dtype=bool)
        if neuron_to_plot_in_red is not None:
            mask[neuron_to_plot_in_red] = False
            
        for i in range(ft.shape[1]):
            if neuron_to_plot_in_red is not None:
                ax.scatter(ft[~mask,i], neurons[~mask], alpha=1, marker='s', s=3, 
                          label=f'Fitted {i} (highlighted)', edgecolors='none', color='red')
            ax.scatter(ft[mask,i], neurons[mask], alpha=0.5, marker='s', s=3, 
                      label=f'Fitted {i}', edgecolors='none', color=colors[i])

    # ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel(r'Sorted neuron index')
    ax.set_xlabel(r'Weight')
    ax.set_ylim(-1, len(ft))
    ax.set_yticks([])

    # handles, labels = ax.get_legend_handles_labels()
    # if handles:
    #     ax.legend(handles, labels, loc='lower right', frameon=False, fontsize=6)
