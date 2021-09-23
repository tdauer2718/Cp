import os
from pathlib import Path
from linecache import getline
from tau_functions import fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def find_thermometers(sample):
    ''' 
    Inputs:     `sample`, the sample name
    Outputs:    `thermometers`, a list of names of thermometers that were used in the measurements
    '''
    # Find the directories in which the data is stored for each thermometer
    dirs = [x[0].split("/") for x in os.walk(Path(f"{sample} data/To use/"))]
    # Store the thermometer names in a list
    thermometer_dirs = [x for x in dirs if len(x) == 3]
    thermometers = [x[2] for x in thermometer_dirs]
    return thermometers


def get_files(sample, measurement_type, thermometer):
    '''
    Inputs:     `sample`, the sample name
                `measurement_type`, a string chosen from ["Tau", "Kappa", "Rth", "Average Decay"]
                `thermometer`, a string corresponding to the thermometer being used
    Outputs:    `measurement_files`, a list of the files for this sample, measurement_type, and thermometer
    '''
    # Make a list of files for measurement_type for this thermometer
    path = Path(f"{sample} data/To use/{thermometer}")
    files = path.glob("*.txt")
    measurement_files = []
    for input_file in files:
        if measurement_type in input_file.name:
            measurement_files.append(str(input_file))
    return measurement_files


def create_dict(sample, measurement_type, thermometers, Rth_fits="undefined"):
    '''
    Inputs:     `sample`, the sample name
                `measurement_type`, a string chosen from ["Tau", "Kappa", "Rth"]
                `thermometers`, a list of the thermometers being used
                `Rth_fits`, relevant if measurement_type =="Kappa" only;
                                        a list of T vs. Rth fit polynomials for each thermometer
    Outputs:
    '''
    result_dict = {}

    for thermometer in thermometers:

        # Initialize the dictionary entry
        result_dict[thermometer] = {}
        # For each thermometer, there will be a dictionary of DataFrames, one for each run

        # Make a list of tau files for this thermometer
        measurement_files = get_files(sample, measurement_type, thermometer)

        if measurement_type == "Kappa":
            poly = Rth_fits[thermometer]
        
        # Extract T and measurement_type values from these files and make a DataFrame for each run
        # The DataFrame's first column is T and the second is the measurement value (Rth, Kappa, or Tau)
        for measurement_file in measurement_files: # each file corresponds to a different run
            file_str = measurement_file.split("/")[-1][:-4]
            run = file_str.replace(f"{measurement_type} ", "")
            if measurement_type == "Tau":
                dT = float(getline(measurement_file, 1).split("= ")[-1])
            with open(measurement_file, "r") as f:
                num_lines = sum([1 for line in f])
            T_list, data_list = [], []
            if measurement_type == "Kappa":
                point_list = []
            # counter = 0
            for i in range(3, num_lines+1):
                line = str.split(getline(measurement_file,i))
                if measurement_type == "Kappa":
                    T0, P, R, dR = [float(line[i]) for i in range(4)]
                    if P == 0: # if this is a line where power=0
                        point = 0
                        Rth = R
                        if "Pt" in thermometer:
                            T_i = poly(Rth)
                        else:
                            T_i = np.exp(poly(np.log(Rth)))
                        continue
                    R = Rth + dR
                    if "Pt" in thermometer:
                        dT = poly(R) - T_i
                    else:
                        dT = np.exp(poly(np.log(R))) - T_i
                    T = T0 + 0.5*dT
                    kappa = P/dT
                    data_point = kappa
                    point += 1
                else:
                    T = float(line[0])
                    if measurement_type == "Tau":
                        T += 0.5*dT
                    data_point = float(line[1])
                if data_point > 0:
                    T_list.append(T)
                    data_list.append(data_point)
                    if measurement_type == "Kappa":
                        point_list.append(point)
            # T_list should already be sorted by increasing temperature, but make sure anyway:
            if measurement_type == "Kappa":
                T_list, data_list, point_list = zip(*sorted(zip(T_list, data_list, point_list)))
            else:
                T_list, data_list = zip(*sorted(zip(T_list, data_list)))
            # Create a DataFrame
            if measurement_type == "Kappa":
                data_table = pd.DataFrame(np.array([T_list, data_list, point_list]).T, columns=["T", measurement_type, "point"])
                data_table = data_table.astype({"point": int})
            else:
                data_table = pd.DataFrame(np.array([T_list, data_list]).T, columns=["T", measurement_type])
            # if measurement_type == "Kappa":
            #     data_table.num_series = num_series
            if measurement_type == "Tau":
                data_table.dT = dT
            # Add to the dictionary
            result_dict[thermometer][run] = data_table

        data_table = combine_data(result_dict, thermometer)
        result_dict[f"All {thermometer}"] = data_table
    
    concat_list = [result_dict[f"All {thermometer}"] for thermometer in thermometers]
    result_dict["All Data"] = pd.concat(concat_list, ignore_index=True).sort_values(by="T")
    result_dict["All Data"].reset_index(drop=True, inplace=True)

    return result_dict


def combine_data(result_dict, thermometer):
    '''
    Inputs:     `result_dict`, a dictionary of data generated by create_dict
                `thermometer`, the thermometer we're interested in
    Outputs:    `data_table`, a single table combining the data in result_dict for a given thermometer
    '''
    first_run = list(result_dict[thermometer].keys())[0]
    columns = list(result_dict[thermometer][first_run].columns)
    data_table = pd.DataFrame(columns=columns)
    for run in result_dict[thermometer].keys():
        df = result_dict[thermometer][run]
        # df = result_dict[thermometer][run].append({'run': run}, ignore_index=True)
        data_table = pd.concat([data_table, df], ignore_index=True)
    data_table.sort_values(by="T", inplace=True)
    data_table.reset_index(inplace=True, drop=True)
    return data_table


# This function is useful for getting the thermometer names from a result_dict, instead of
# needing to pass a list of thermometers to a function that uses result_dict, e.g. for plot_data() below
def thermometers_from_dict(result_dict):
    '''
    Inputs:     `result_dict` for one of the measurement types
    Outputs:    a list of thermometers from that result_dict
    '''
    possible_thermometers = ["LT thermometer", "HT thermometer", "Pt thermometer"]
    keys = result_dict.keys()
    return list(set(possible_thermometers) & set(keys))

def plot_with_opt(x_data, y_data, label, marker="o", plot_type="loglog"):
    if plot_type == "linear":
        plot_type = "plot"
    getattr(plt, plot_type)(x_data, y_data, marker, label=label)
    
def plot_df(df, measurement, plot_type="loglog", fit_df=False):
    plt.figure(figsize=(10,8))
    if plot_type == "linear":
        plot_type = "plot"
    if type(fit_df) == pd.core.frame.DataFrame:
        getattr(plt, plot_type)(df["T"], df[measurement], "o", label="Data")
        getattr(plt, plot_type)(fit_df["T"], fit_df[measurement], "--", label="Fit")
        plt.legend()
    else:
        getattr(plt, plot_type)(df["T"], df[measurement], "o")
    plt.xlabel("T")
    plt.ylabel(measurement)
    plt.show()

'''
def plot_params(thermometers):
    param_dict = {}
    for thermometer in thermometers:
        param_dict[thermometer] = {}
        print(f"For the {thermometer}:")
        input("Lower x limit?")
        input("Upper x limit?")
        input("Lower y limit?")
        input("Upper y limit?")
        x_lims = (int(i) for i in line.split())
        y_lims = (int(i) for i in line.split())
        param_dict[thermometer]["x_lims"] = x_lims
        param_dict[thermometer]["y_lims"] = y_lims
    return param_dict
'''

def plot_data(result_dict, plot_type="linear", combine_thermometers=False, fit_dict=False, T_cuts_dict=False, axes=False):
    '''
    Inputs:     `result_dict`
                `fit_dict`, which is False if the user doesn't want it included, and the dictionary
                            fit_dict otherwise
                `T_cuts_dict`, which is False if the user doesn't want it included, and the dictionary
                                T_cuts_dict otherwise
    Outputs:    Returns nothing, but plots the data in result_dict
    '''
    data_types = ["Tau", "Kappa", "Rth", "T"]
    units = ["s", "W/K", "ohms", "K"]
    thermometers = thermometers_from_dict(result_dict)
    first_run = list(result_dict[thermometers[0]].keys())[0]
    columns = list(result_dict[thermometers[0]][first_run].columns)
    if fit_dict:
        data_for_fits = {}
        if columns[1] == "Rth":
            columns[0], columns[1] = columns[1], columns[0] 
            # in this case, plot T(Rth) instead of Rth(T)
    x_unit, y_unit = [units[data_types.index(columns[i])] for i in range(2)]
    for thermometer in thermometers:
        plt.figure(figsize=(10,8))
        for run in result_dict[thermometer].keys():
            # Add stuff for the kappa points...
            x_data = result_dict[thermometer][run][columns[0]]
            y_data = result_dict[thermometer][run][columns[1]]
            plot_with_opt(x_data, y_data, label=run, plot_type=plot_type)
        if fit_dict:
            poly = fit_dict[thermometer]
            x_data_all = combine_data(result_dict, thermometer)[columns[0]]
            y_fit = poly(x_data_all)
            if "Rth" in columns:
                if "Pt" in thermometer:
                    y_fit = poly(x_data_all)
                else:
                    y_fit = np.exp(poly(np.log(x_data_all)))
            data_for_fits[thermometer] = pd.DataFrame(np.array([x_data_all, y_fit]).T, columns=columns)
            if T_cuts_dict:
                # Cut off the data using the appropriate temperatures
                T_cuts = T_cuts_dict[thermometer]
                data_for_fits[thermometer] = data_for_fits[thermometer][data_for_fits[thermometer]["T"] > T_cuts[0]]
                data_for_fits[thermometer] = data_for_fits[thermometer][data_for_fits[thermometer]["T"] < T_cuts[1]]
                x_data_all = data_for_fits[thermometer][columns[0]]
                y_fit = data_for_fits[thermometer][columns[1]]
            plot_with_opt(x_data_all, y_fit, marker="--", label="fit", plot_type=plot_type)
        plt.title(thermometer)
        plt.xlabel(columns[0] + f" ({x_unit})")
        plt.ylabel(columns[1] + f" ({y_unit})")
        plt.legend()
        plt.show()
    # This code is identical in large part to the above, violating the DRY principle...
    # My ideas to encapsulate this taking into account both cases are a bit kludgey
    # at the moment, though.
    # -----------> Come back to this later
    if combine_thermometers:
        plt.figure(figsize=(10,8))
        for thermometer in thermometers:
            thermometer_label = thermometer.split()[0]
            for run in result_dict[thermometer].keys():
                x_data = result_dict[thermometer][run][columns[0]]
                y_data = result_dict[thermometer][run][columns[1]]
                run = thermometer_label + " " + run
                plot_with_opt(x_data, y_data, label=run, plot_type=plot_type)
            if fit_dict:
                x_data_all = data_for_fits[thermometer][columns[0]]
                y_fit = data_for_fits[thermometer][columns[1]]
                plot_with_opt(x_data_all, y_fit, marker="--", label=f"{thermometer} fit", plot_type=plot_type)
        plt.title("All thermometers")
        plt.xlabel(columns[0] + f" ({x_unit})")
        plt.ylabel(columns[1] + f" ({y_unit})")
        plt.legend()
        plt.show()
    
def get_T_cuts(thermometers):
    T_cuts_dict = {}
    for thermometer in thermometers:
        T_cuts_dict[thermometer] = [0, 1000]
        choices = ["lower", "upper"]
        for i in range(2):
            entry = input(f"What is the ** {choices[i]} ** cutoff for T (in K) for the ** {thermometer} **? Press Enter if none.")
            if entry != "":
                T_cuts_dict[thermometer][i] = float(entry)
    return T_cuts_dict

def cut_data(result_dict, thermometers):
    '''
    Prompts the user for T_low and T_high for each thermometer as cutoff temperatures.
    '''
    T_cuts_dict = get_T_cuts(thermometers)
    for thermometer in thermometers:
        T_cuts = T_cuts_dict[thermometer]
        df = result_dict[thermometer]
        keys = list(df.keys())
        for run in keys:
            # Drop points less than the lower T cutoff
            df[run].drop(df[run][df[run]["T"] < T_cuts[0]].index, inplace=True)
            # Drop points greater than the upper T cutoff
            df[run].drop(df[run][df[run]["T"] > T_cuts[1]].index, inplace=True)
            df[run].reset_index(inplace=True, drop=True)
        df = result_dict[f"All {thermometer}"]
        # Drop points less than the lower T cutoff
        df.drop(df[df["T"] < T_cuts[0]].index, inplace=True)
        # Drop points greater than the upper T cutoff
        df.drop(df[df["T"] > T_cuts[1]].index, inplace=True)
        df.reset_index(inplace=True, drop=True)

def num_Kappa_points(kappa_dict):
    thermometers = thermometers_from_dict(kappa_dict)
    result_dict = {}
    for thermometer in thermometers:
        df = kappa_dict[f"All {thermometer}"]
        num_points = int(df["point"].max())
        result_dict[thermometer] = num_points
    return result_dict

def drop_runs_helper(measurement_dict, drop_list=False):
    thermometers = thermometers_from_dict(measurement_dict)
    for thermometer in thermometers:
        if drop_list:
            to_drop = drop_list[thermometer]
        else:
            to_drop = []
            while True:
                run = input(f"Enter the runs you want to drop for the {thermometer}, and then press Enter.\n If no more runs to drop, just press Enter.")
                if run == "":
                    break
                to_drop.append(run)
        if len(to_drop):
            for run in to_drop:
                if run in measurement_dict[thermometer]:
                    del measurement_dict[thermometer][run]
                    print(f"{thermometer} {run} deleted")
                else:
                    print(f"{run} does not exist for the {thermometer}")
            measurement_dict[f"All {thermometer}"] = combine_data(measurement_dict, thermometer)
    concat_list = [measurement_dict[f"All {thermometer}"] for thermometer in thermometers]
    measurement_dict["All Data"] = pd.concat(concat_list, ignore_index=True).sort_values(by="T")
    measurement_dict["All Data"].reset_index(drop=True, inplace=True)

def ask_for_degrees_helper(thermometers):
    result_dict = {}
    for thermometer in thermometers:
        degree_str = input(f"What degree polynomial do you want to use for the {thermometer} fit? \n Default is 2.")
        if degree_str == "":
            degree = 2
        else:
            degree = int(degree_str)
        result_dict[thermometer] = degree
    return result_dict

def spline_fit(ref_data, eval_on, measurement, s):
    '''
    Inputs:
    This takes a data table `ref_data` that has a "T" column and a value column of type
    `measurement`. It interpolates this data and then applies
    this interpolation function to the temperatures in the table `eval_on`.
    Outputs: `data_interpolated`: DataFrame with T values from the `eval_on` data
                                   and data resulting from the interpolation
                                   function applied at these temperatures.
    '''
    data_interpolated = []
    interp_func = interpolate.splrep(ref_data["T"], ref_data[measurement], s=s)
    data_interpolated = interpolate.splev(eval_on["T"], interp_func)
    return data_interpolated

def find_outliers(input_df, measurement, T_cut):
    '''
    Interactive removal of outliers based on how well they match
    a spline fit to the data. More details are in the main analysis notebook.
    '''
    df = input_df.loc[input_df["T"] < T_cut]
    outlier_indices_input = []
    if measurement == "Kappa":
        s = 1e-11
    elif measurement == "Tau":
        s = 1e-2
    while True:
        data_interpolated = spline_fit(df, df, measurement, s=s)
        fit_df = pd.DataFrame(np.array([df["T"], list(data_interpolated)]).T, columns=
                                    ["T", measurement])
        prompt_str = "Enter the plot type. Default value is the last "
        prompt_str += "value entered, starting with \"loglog\"."
        plot_type = input(prompt_str)
        if plot_type == "":
            plot_type = "loglog"
    #     plot_df(df, measurement, plot_type=plot_type, fit_df=fit_df)
    #     prompt_str = "Enter the lower cutoff temperature in K."
    #     prompt_str += "Default is 0 K."
    #     T_low = float(input(prompt_str))
        plot_df(df, measurement, plot_type=plot_type, fit_df=fit_df)
        prompt_str = "Does s need to be changed? Its current value is "
        prompt_str += f"{s}. Press Enter if it doesn't need to be changed."
        change_s = input(prompt_str)
        if change_s != "":
            s = float(change_s)
            continue
        prompt_str = "Finished with outlier removal? Enter "
        prompt_str += "\"y\" or \"n\"; the default is \"n\"."
        finished = input(prompt_str)
        if finished == "y":
            break
        if measurement == "Tau":
            thresh = 0.3
        else:
            thresh = 0.1
        prompt_str = f"Enter the outlier threshold. The default value is {thresh}."
        thresh_str = input(prompt_str)
        if thresh_str != "":
            thresh = float(thresh_str)
        outlier_indices, outlier_Ts, outlier_measurements = [], [], []
        for i in range(len(df)):
            err = np.abs((df[measurement][i] - fit_df[measurement][i])/fit_df[measurement][i])
            if err > thresh:
                outlier_Ts.append(df["T"][i])
                outlier_measurements.append(df[measurement][i])
        for i in range(len(outlier_Ts)):
            outlier_indices.append(df.loc[df["T"] == outlier_Ts[i]].index[0])
            outlier_indices_input.append(input_df.loc[input_df["T"] == outlier_Ts[i]].index[0])
        plt.figure(figsize=(10,8))
        if plot_type == "linear":
            foo = "plot"
        else:
            foo = plot_type
        getattr(plt, foo)(df["T"], df[measurement], "bo", label="Data")
        getattr(plt, foo)(outlier_Ts, outlier_measurements, "ro", label="Outliers")
        getattr(plt, foo)(fit_df["T"], fit_df[measurement], "--", label="Fit")
        plt.xlabel("T")
        plt.ylabel(measurement)
        plt.legend()
        plt.show()
        accept = input("Throw out these points? Answer \"y\" or \"n\"; default is \"y\".")
        if accept == "n":
            # In this case, go to the top of the loop and let the user
            # enter a different threshold
            continue
        # Otherwise, throw out the points
        df = df.drop(outlier_indices)
        df.reset_index(inplace=True, drop=True)
        # Then return to the top of the loop. Note that the spline interpolation
        # is done again at the top of the loop, now with some points removed.
    return outlier_indices_input