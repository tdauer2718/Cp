import numpy as np
from scipy.optimize import curve_fit
from linecache import getline
import matplotlib.pyplot as plt
import os, sys, glob, errno
from pathlib import Path
import pandas as pd
from scipy import interpolate

def func(x, a, tau, y0):
    return a * np.exp(-x/tau) + y0

def load_decay(decay_file):
    with open(decay_file, "r") as f:
        num_lines = sum([1 for line in f])
    decay = []
    for i in range(3, num_lines+1):
        decay.append(float(getline(decay_file, i)))
    return decay

def plot_decay(decay_file):
    time_per_point = float(str.split(getline(decay_file, 2))[4])
    decay = load_decay(decay_file)
    times = [i*time_per_point for i in range(len(decay))]
    plt.plot(times, decay)
    plt.show()

def fit(decay_file, tau_est, delay, num_taus=2, plot_option="N", show_plot_option="N"):
    time_per_point = float(str.split(getline(decay_file, 2))[4])
    decay = load_decay(decay_file)
    times = [i*time_per_point for i in range(len(decay))]
    start_fit_point = int(delay/time_per_point)
    stop_fit_point = min(start_fit_point + int(num_taus*tau_est/time_per_point), len(decay))
    y0 = decay[-1]
    if decay[0] < decay[-1]:
        a = -200
    else:
        a = 200
    popt, pcov = curve_fit(func, times[start_fit_point:stop_fit_point], decay[start_fit_point:stop_fit_point], [a, tau_est, y0])
    tau = popt[1]
    if plot_option=="Y":
        plt.plot(times, decay)
        plt.plot(times[start_fit_point:stop_fit_point], [func(i, *popt) for i in times[start_fit_point:stop_fit_point]], "r--")
        plt.xlabel("time (s)")
        if show_plot_option=="Y":
            plt.show()
    return tau

def running_mean(tau_list, s=50):
    avg_list = []
    avg = np.mean(tau_list[:s])
    avg_list.append(avg)
    for i in range(1, len(tau_list)-s):
        avg += (tau_list[i+s-1] - tau_list[i-1])/s
        avg_list.append(avg)
    return avg_list

def smart_fit(decay_file, frac=0.3, delay_manual=0, show_plot_option='N'):
    time_per_point = float(str.split(getline(decay_file, 2))[4])
    decay = load_decay(decay_file)
    times = [i*time_per_point for i in range(len(decay))]
    total_time = times[-1]
    yi, yf = decay[0], decay[-1]
    
    if yi < yf: # If the decay is inverted
        # Reflect the decay about its initial baseline
        for i in range(len(decay)):
            decay[i] = yi - (decay[i]-yi)
        yf = decay[-1]
    
    # Find the first point that's frac the way down the curve
    i = 0
    avg_list = running_mean(decay, s=50)
    ai, af = avg_list[0], avg_list[-1]
    while ai - avg_list[i] < frac*np.abs(ai - af):
    #while yi-decay[i] < frac*(yi-yf):
        i += 1
    start_fit_point = i
    
    if delay_manual > 0:
        start_fit_point = int(delay_manual/time_per_point)
    
    a = 200
    y0 = yf
    tau_est = total_time/5
    #stop_fit_point = start_fit_point + int(num_taus*tau_est/time_per_point)
    stop_fit_point = len(decay)
    times_for_fit = times[start_fit_point:stop_fit_point]
    decay_for_fit = decay[start_fit_point:stop_fit_point]
    
    try:
        popt, pcov = curve_fit(func, times_for_fit, decay_for_fit, [a, tau_est, y0])
        a, tau, y0 = popt[0], popt[1], popt[2]
        fit_pred = [func(t, a, tau, y0) for t in times_for_fit]
        MSE = np.square(np.subtract(decay_for_fit, fit_pred)).mean()
        scaled_RMSE = 100*np.sqrt(MSE)/np.abs(yi - yf)

        if show_plot_option == 'Y':
            plt.plot(times, decay)
            plt.plot(times_for_fit, [func(i, *popt) for i in times_for_fit], "r--")
            plt.xlabel("time (s)")
            plt.show()

        return tau, scaled_RMSE
    except RuntimeError:
        # Very occasionally the fitting routine gives an error; I still want to move to the
        # next temperature and continue the series.
        return 0


# files = path.glob('*.txt')
# decay_files = []

# for input_file in files:
#     if ('Average Decay T0' in input_file.name) is True:
#         decay_files.append(str(input_file))
# #     except IOError as exc:
# #         if exc.errno != errno.ESIDIR:
# #             raise

# T0_list = []
# tau_list = []
# run_list = []
# for decay_file in decay_files:
#     run_list.append(' '.join(str.split(decay_file.replace('/',' ').replace('.',' '))[4:-1]))
#     tau = smart_fit(decay_file,frac=0.25, show_plot_option='Y')
#     tau_list.append(tau)
#     T0 = float(str.split(decay_file)[5].replace('p','.'))
#     T0_list.append(T0)
#     print('tau = %.3g s' % tau)
#     print(decay_file)