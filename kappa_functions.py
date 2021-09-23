from linecache import getline
import numpy as np
from helper_functions import thermometers_from_dict, plot_with_opt
import matplotlib.pyplot as plt

def plot_kappa_points(kappa_dict, plot_type="linear"):
    thermometers = thermometers_from_dict(kappa_dict)
    for thermometer in thermometers:
        plt.figure(figsize=(10,8))
        df = kappa_dict[f"All {thermometer}"]
        points = df["point"].unique()
        # I know using max() here isn't as efficient as it could be, but it takes hardly any time
        # since there are generally only a few hundred lines in these DataFrames,
        # and the code is cleaner than the alternative
        for point in points:
            kappa_table = df[df["point"] == point]
            plot_with_opt(kappa_table["T"], kappa_table["Kappa"], plot_type=plot_type, marker="o", label = f"Point {point}")
        plt.legend()
        plt.title(f"Kappa points for {thermometer}")
        plt.xlabel("T (K)")
        plt.ylabel("Kappa (W/K)")
        plt.show()

def select_points(kappa_dict, num_kappa_points):
    thermometers = thermometers_from_dict(kappa_dict)
    for thermometer in thermometers:
        print(f"Which points do you want to * drop * for the {thermometer}?")
        print("(The default is to keep the highest two.)")
        points_str = input("Separate them by spaces and then press Enter.")
        if points_str == "":
            num = num_kappa_points[thermometer]
            points_to_drop = list(range(1, num - 1))
        else:
            points_to_drop = [int(i) for i in points_str.split()]
        for run in kappa_dict[thermometer].keys():
            df = kappa_dict[thermometer][run]
            kappa_dict[thermometer][run].drop(df[df["point"].isin(points_to_drop)].index, inplace = True)
            kappa_dict[thermometer][run].reset_index(inplace=True, drop=True)
        df = kappa_dict[f"All {thermometer}"]
        kappa_dict[f"All {thermometer}"].drop(df[df["point"].isin(points_to_drop)].index, inplace = True)
        kappa_dict[f"All {thermometer}"].reset_index(inplace=True, drop=True)