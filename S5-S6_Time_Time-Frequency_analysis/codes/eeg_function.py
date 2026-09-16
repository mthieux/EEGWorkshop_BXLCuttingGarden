
from missoutlier import detect_outliers_miss
import numpy as np

# Function to drop trials from epochs based on peak, mean, and slope criteria

def drop_trials(epochs, chs='all', do_peak=0, do_mean=0, do_slope=0, T1=0, T2=0):
    if chs!='all': # iterate over channel select or take the first one
        chs = [i for i,x in enumerate(epochs.info['ch_names']) if x in chs]
    else:
        chs = list(np.arange(len(epochs.info['ch_names'])))

    # Create a copy of epochs
    new_epochs = epochs.copy()
    # Create a list to keep track of trial status (1 for good trials, 0 for rejected trials)
    list_trials = np.ones(len(new_epochs))
    mean_rejection = 0
    slope_rejection = 0
    peak_rejection = 0
    for j in chs:


        if do_peak == 1:
            # Calculate time indices based on T1 and T2 values
            time1 = np.where(new_epochs.times < T1)[0][-1]
            time2 = np.where(new_epochs.times < T2)[0][-1]

            # Calculate peak-to-peak values for each trial and channel
            peak_to_peak = np.array([max(new_epochs._data[i][j][time1:time2]) - min(new_epochs._data[i][j][time1:time2]) for i in range(len(new_epochs._data))], dtype=np.dtype(float))
            
            peak_to_peak = detect_outliers_miss(peak_to_peak, drop=False, na_rm=True, silent=True)

            list_trials[np.isnan(peak_to_peak)] = 0
            peak_rejection =  sum(np.isnan(peak_to_peak)) / len(peak_to_peak) * 100

        if do_slope == 1:
            # Calculate time indices based on T1 and T2 values
            time1 = np.where(new_epochs.times < T1)[0][-1]
            time2 = np.where(new_epochs.times < T2)[0][-1]

            x = new_epochs.times[time1:time2]
            # Calculate slopes for each trial and channel
            slopes_trials = np.array([np.polyfit(x, new_epochs._data[i, j, time1:time2], 1)[0] for i in range(len(new_epochs._data))], dtype=np.dtype(float))

            slopes_trials = detect_outliers_miss(slopes_trials, drop=False, na_rm=True, silent=True)

            list_trials[np.isnan(slopes_trials)] = 0
            slope_rejection =  sum(np.isnan(slopes_trials)) / len(slopes_trials) * 100

        if do_mean == 1:
            # Calculate time indices based on T1 and T2 values
            time1 = np.where(new_epochs.times < T1)[0][-1]
            time2 = np.where(new_epochs.times < T2)[0][-1]
            # Calculate mean values for each trial and channel
            mean_trials = np.array([np.mean(new_epochs._data[i][j][time1:time2]) for i in range(len(new_epochs._data))], dtype=np.dtype(float))

            mean_trials = detect_outliers_miss(mean_trials, drop=False, na_rm=True, silent=True)

            list_trials[np.isnan(mean_trials)] = 0
            mean_rejection =  sum(np.isnan(mean_trials)) / len(mean_trials) * 100


    # Drop the trials marked for removal and return the modified epochs object
    new_epochs.drop(np.where(list_trials == 0)[0])
    return new_epochs, list_trials, peak_rejection, mean_rejection, slope_rejection


def single_trial_normalisation(power, tmin_baseline, tmax_baseline, tmin_power, tmax_power):
    imin_baseline = np.where(power.times<=tmin_baseline)[0][-1]
    imax_baseline = np.where(power.times<=tmax_baseline)[0][-1]
    imin = np.where(power.times<=tmin_power)[0][-1]
    imax = np.where(power.times<=tmax_power)[0][-1]
    for n_trial in range(len(power)):
        data= power._data[n_trial,...]
        mean = np.mean(data[...,imin:imax],axis=-1,keepdims=True)
        std = np.std(data[...,imin:imax],axis=-1,keepdims=True)
        data-=mean
        data /= std


    power_avg = power.average()
    mean_avg = np.mean(power_avg._data[...,imin_baseline:imax_baseline],axis=-1,keepdims=True)
    std_avg = np.std(power_avg._data[...,imin_baseline:imax_baseline],axis=-1,keepdims=True)
    
    for n_trial in range(len(power)):
        data= power._data[n_trial,...]
        data-=mean_avg
        data /= std_avg
    return power