from linecache import getline
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from helper_functions import find_thermometers, create_dict, get_T_cuts, plot_data, cut_data
from helper_functions import num_Kappa_points, spline_fit, plot_df, get_files
from helper_functions import drop_runs_helper, ask_for_degrees_helper, find_outliers
from Rth_functions import get_Rth_polys
from kappa_functions import plot_kappa_points, select_points
from tau_functions import smart_fit

class Data:

    def __init__(self, sample, Cu_thickness, sample_thickness):
        self.sample = sample
        self.Cu_thickness = Cu_thickness
        self.sample_thickness = sample_thickness
        self.thermometers = find_thermometers(sample)
        self.Rth_fits = "undefined"
        self.Tau_fits = {}
    
    def get_Taus(self):
        self.Tau_data = create_dict(self.sample, "Tau", self.thermometers)
        self.orig_Tau_data = deepcopy(self.Tau_data)

    def get_Rths(self):
        self.Rth_data = create_dict(self.sample, "Rth", self.thermometers)
        self.orig_Rth_data = deepcopy(self.Rth_data)

    def do_Rth_fits(self, cut=False, manual=False, ask_for_degrees=False, plot_type="loglog"):
        T_cuts_dict = False
        degree_dict = False
        if cut:
            if manual:
                # Prompt the user for T_cut values
                T_cuts_dict = get_T_cuts(self.thermometers)
            else:
                # The typical cut temperatures are well-known from previous
                # measurements
                T_cuts_dict = {}
                T_cuts_dict["LT thermometer"] = [2, 7]
                T_cuts_dict["HT thermometer"] = [7, 30]
        if ask_for_degrees:
            degree_dict = ask_for_degrees_helper(self.thermometers)
        self.Rth_fits = get_Rth_polys(self.Rth_data, T_cuts_dict, degree_dict)
        self.Rth_fits_cuts = T_cuts_dict
        self.plot_dict("Rth", plot_type=plot_type, fits=True, T_cuts=True)
    
    def get_Kappas(self):
        if self.Rth_fits == "undefined":
            print("You need to find the T vs Rth fit polynomial for each thermometer first by using the \"do_Rth_fits()\" method.")
            return
        self.Kappa_data = create_dict(self.sample, "Kappa", self.thermometers, 
                                      Rth_fits=self.Rth_fits)
        self.num_Kappa_points = num_Kappa_points(self.Kappa_data)
        self.plot_dict("Kappa points", plot_type="loglog")
        select_points(self.Kappa_data, self.num_Kappa_points)
        self.orig_Kappa_data = deepcopy(self.Kappa_data) 
        # Note this is after some points may have been cut
        
    def get_Taus(self):
        '''
        Gets the values of Tau from files for each run; these files have values
        obtained from fits done by the program that runs the experiment.
        Sometimes these fits need to be examined more closely, and that is
        the purpose of the get_Tau_fits() method below.
        '''
        self.Tau_data = create_dict(self.sample, "Tau", self.thermometers)
        self.orig_Tau_data = deepcopy(self.Tau_data)
    
    def get_Tau_fits(self, thermometer, RMSE_cutoff=1.0):
        '''
        Fits the Tau decays, shows the user the corresponding plots,
        and removes the poor decays based on a root-mean-square
        error criterion. Replaces old data in self.Tau_dict[thermometer]
        with this data.
        '''
        print("RMSE cutoff is", RMSE_cutoff)
        decay_filenames = get_files(self.sample, "Average Decay", thermometer)
        decay_list = []
        T0_list = []
        for filename in decay_filenames:
            T0 = float(filename.split("/")[-1].split()[-2].replace("p", "."))
            T0_list.append(T0)
        T0_list, decay_filenames = zip(*sorted(zip(T0_list, decay_filenames)))
        T_list, run_list, tau_list = [], [], []
        for i, filename in enumerate(decay_filenames):
            dT = float(getline(filename, 1).split("= ")[-1])
            T = (1 + dT/2)*T0_list[i]
            line1 = getline(filename, 1).split()[:4]
            run = line1[3] + " " + line1[0] + " " + line1[1]
            tau, scaled_RMSE = smart_fit(filename, frac=0.3, delay_manual=0, show_plot_option='Y')
            print(f"{run}: T = {T:.3g} K, Tau = {tau:.3g} s with scaled RMSE {scaled_RMSE:.3g}")
            if scaled_RMSE < RMSE_cutoff:
                T_list.append(T)
                run_list.append(run)
                tau_list.append(tau)
            else:
                print_str = "\x1b[31m The scaled RMSE is above the cutoff, so this decay "
                print_str += "won't be included.\x1b[0m"
                print(print_str)
            print("\n\n\n\n")
        fits_df = pd.DataFrame(np.array([T_list, tau_list, run_list]).T, columns=["T", "Tau", "Run"])
        self.Tau_fits[thermometer] = fits_df
        # Now change self.Tau_data to reflect the values of Tau obtained from these fits
        thermometer_dict = getattr(self, "Tau_data")[thermometer]
        thermometer_dict = {}
        runs = fits_df["Run"].unique()
        for run in runs:
            thermometer_dict[run] = []
        for i in range(len(fits_df)):
            row = fits_df.loc[i]
            thermometer_dict[row["Run"]].append([row["T"], row["Tau"]])
        for run in runs:
            thermometer_dict[run] = pd.DataFrame(np.array(thermometer_dict[run]), columns=["T", "Tau"])
            
    def remove_outliers(self, input_df, measurement, T_cut):
        '''
        The functionality here is in the find_outliers() function
        in the helper_functions file.
        '''
        to_remove = find_outliers(input_df, measurement, T_cut)
        getattr(self, f"{measurement}_data")["All Data"].drop(to_remove, inplace=True)
        getattr(self, f"{measurement}_data")["All Data"].reset_index(inplace=True, drop=True)
    
    def get_Kappa_interpolated(self, T_cut, s=1e-11):
        # Interpolate the Kappa data
        ref_data = self.Kappa_data["All Data"]
        # Evaluate this spline at the temperatures where Tau was measured
        eval_on = self.Tau_data["All Data"]
        ref_measurement, name = "Kappa", "Kappa"
        self.do_spline_fit(ref_data, eval_on, ref_measurement, name, T_cut=T_cut, s=s)
        
# Physics dictates that kappa is linear in most of the range below 100 K,
# where the T^3 radiation contribution is negligible (but becomes significant
# in the 100-150 K range). Tunneling physics causes this to break down in the
# range of a few K, however, due to the tunneling two-level systems in the
# silicon nitride membrane. This crossover makes it rather painful to model
# the Kappa data using a standard function model, so I use a spline model
# instead, as above.
#     def get_Kappa_fit(self, T_cut, plot_type="linear"):
#         # Gets a linear fit to the Kappa data below T_cut
#         df = self.Kappa_data["All Data"]
#         df = df.loc[df["T"] < T_cut]
#         X = np.array(df["T"])
#         y = np.array(df["Kappa"])
#         X = X.reshape(-1, 1)
#         reg = LinearRegression()
#         reg.fit(X, y)
#         kappa_pred = reg.predict(X)
#         result_df = pd.DataFrame(np.array([df["T"], kappa_pred]).T, columns=["T", "Kappa"])
#         plot_df(df, "Kappa", fit_df=result_df, plot_type="linear")
    
    def get_total_Cp(self, T_cut):
        T_list, Cp_list = [], []
        kappa_df, tau_df = self.Kappa_interpolated, self.Tau_data["All Data"]
        kappa_df = kappa_df.loc[kappa_df["T"] < T_cut]
        tau_df = tau_df.loc[tau_df["T"] < T_cut]
        for i in range(len(tau_df)):
            T_list.append(tau_df["T"][i])
            Cp_list.append(kappa_df["Kappa"][i]*tau_df["Tau"][i])
        self.total_Cp = pd.DataFrame(np.array([T_list, Cp_list]).T, columns=["T", "Cp"])
        
    def get_Cu_Cp(self):
        T_list, Cp_list = [], []
        filename = f"{self.sample} data/Copper Cp info.txt"
        with open(filename, "r") as f:
            num_lines = sum([1 for line in f])
        for i in range(3, num_lines + 1):
            line = getline(filename, i).split()
            T_list.append(float(line[0]))
            Cp_list.append(self.Cu_thickness/(300e-9)*float(line[2]))
        df = pd.DataFrame(np.array([T_list, Cp_list]).T, columns=["T", "Cp"])
        self.Cu_data = df
        
    def get_Cu_Cp_interpolated(self, T_cut, s=1e-18):
        # Interpolate the Cu Cp data
        ref_data = self.Cu_data
        # Evaluate this spline at the temperatures where Tau was measured
        eval_on = self.Tau_data["All Data"]
        ref_measurement, name = "Cp", "Cu_Cp"
        self.do_spline_fit(ref_data, eval_on, ref_measurement, name, T_cut=T_cut, s=s)
        
    def get_background_Cp(self):
        T_list, Cp_list = [], []
        filename = f"{self.sample} data/V19-007a background microcal only.txt"
        with open(filename, "r") as f:
            num_lines = sum([1 for line in f])
        for i in range(3, num_lines + 1):
            line = getline(filename, i).split()
            T_list.append(float(line[0]))
            Cp_list.append(float(line[1]))
        df = pd.DataFrame(np.array([T_list, Cp_list]).T, columns=["T", "Cp"])
        self.background_data = df
    
    def get_background_Cp_interpolated(self, T_cut, s=1e-18):
        # Interpolate the Cu Cp data
        ref_data = self.background_data
        # Evaluate this spline at the temperatures where Tau was measured
        eval_on = self.Tau_data["All Data"]
        ref_measurement, name = "Cp", "background_Cp"
        self.do_spline_fit(ref_data, eval_on, ref_measurement, name, T_cut=T_cut, s=s)

    def get_sample_Cp(self, T_cut):
        total, Cu, background = self.total_Cp, self.Cu_Cp_interpolated, self.background_Cp_interpolated
        for df in [total, Cu, background]:
            df = df.loc[df["T"] < T_cut]
        T_list, Cp_list, CpT3_list = [], [], []
        for i in range(len(total)):
            T = total["T"][i]
            Cp = total["Cp"][i] - Cu["Cp"][i] - background["Cp"][i]
            T_list.append(T)
            Cp_list.append(Cp)
            CpT3_list.append(Cp/T**3)
        df = pd.DataFrame(np.array([T_list, Cp_list, CpT3_list]).T, columns=["T", "Cp", "Cp/T^3"])
        self.sample_Cp = df
    
    def do_spline_fit(self, ref_data, eval_on, ref_measurement, name, T_cut=False, s=5e-14):
        if T_cut:
            #ref_data = ref_data[ref_data["T"] < T_cut]
            eval_on = eval_on.loc[eval_on["T"] < T_cut]
        data_interpolated = spline_fit(ref_data, eval_on, ref_measurement, s=s)
        result_df = pd.DataFrame(np.array([eval_on["T"], list(data_interpolated)]).T, columns=
                                ["T", ref_measurement])
        name = f"{name}_interpolated"
        setattr(self, name, result_df)
        if T_cut:
            ref_data = ref_data.loc[ref_data["T"] < T_cut]
        plot_df(ref_data, ref_measurement, fit_df=result_df)
    
    def get_n0(self, T_cut, plot_type="linear"):
        def STM_fit(T, c1, c3): # STM = Standard Tunneling Model
            return c1*T + c3*T**3
        df = self.sample_Cp
        df = df.loc[df["T"] < T_cut]
        popt, pcov = curve_fit(STM_fit, df["T"], df["Cp"])
        perr = np.sqrt(np.diag(pcov))
        c1, c3 = popt[0], popt[1]
        Cp_fit_series = STM_fit(df["T"], *popt)
        CpT3_list = [Cp_fit_series[i]/df["T"][i]**3 for i in range(len(df))]
        Cp_fit = pd.DataFrame(np.array([df["T"], Cp_fit_series, CpT3_list]).T,
                              columns=["T", "Cp", "Cp/T^3"])
        kB = 1.38e-23 # Boltzmann's constant in J/K
        A = (2.5e-3)**2 # The sample area in m^2
        n0 = c1/((np.pi**2/6)*kB**2*A*self.sample_thickness)
        stdev_n0 = (n0/c1)*perr[0]
        df_for_plot = self.sample_Cp
        df_for_plot = df_for_plot.loc[df_for_plot["T"] < 10]
        plot_df(df_for_plot, "Cp/T^3", plot_type=plot_type, fit_df=Cp_fit)
        self.n0, self.c1, self.c3, self.Cp_fit = n0, c1, c3, Cp_fit
        self.stdev_n0 = stdev_n0

    def plot_dict(self, measurement, plot_type="linear", combine_thermometers=False, fits=False, T_cuts=False):
        '''
        Inputs: `measurement`, a value in ["Rth", "Kappa", "Kappa points", "Tau"]
        '''
        if measurement == "Kappa points":
            plot_kappa_points(self.Kappa_data, plot_type=plot_type)
        else:
            fit_dict, T_cuts_dict = False, False
            if fits:
                fit_dict = getattr(self, f"{measurement}_fits")
            if T_cuts:
                T_cuts_dict = getattr(self, f"{measurement}_fits_cuts")
            plot_data(getattr(self, f"{measurement}_data"), plot_type, combine_thermometers, fit_dict, T_cuts_dict) 

    def cut_Taus(self):
        self._undo_Tau_cut = deepcopy(self.Tau_data)
        cut_data(self.Tau_data, self.thermometers)

    def cut_Kappas(self):
        self._undo_Kappa_cut = deepcopy(self.Kappa_data)
        cut_data(self.Kappa_data, self.thermometers)

    def undo_last_cut(self, measurement):
        before_cut = getattr(self, f"_undo_{measurement}_cut")
        setattr(self, f"{measurement}_data", before_cut)

    def reset_data(self, measurement):
        '''
        Inputs: `measurement`, a string chosen from ["Rth", "Kappa", "Tau"]
        '''
        orig_data = deepcopy(getattr(self, f"orig_{measurement}_data"))
        setattr(self, f"{measurement}_data", orig_data)

    def drop_runs(self, measurement_type, drop_list=False):
        measurement_dict = getattr(self, f"{measurement_type}_data")
        drop_runs_helper(measurement_dict, drop_list=drop_list)