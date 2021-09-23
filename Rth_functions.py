import numpy as np
from helper_functions import thermometers_from_dict

def get_Rth_polys(Rth_dict, T_cuts_dict=False, degree_dict=False):
    '''
    Inputs:     `Rth_dict`,
                `T_cut_dict`, 
    Outputs:    `thermometer_polys`,
    '''
    thermometers = thermometers_from_dict(Rth_dict)
    thermometer_polys = {} # polynomials (applied differently for the Pt vs LT/HT thermometers)
    
    for thermometer in thermometers:
        if T_cuts_dict:
            T_cuts = T_cuts_dict[thermometer]
        else:
            T_cuts = [0, 1000] # 0 to 1000 K (all measurements are contained in this range)
        Rth_table = Rth_dict[f"All {thermometer}"]
        # Apply the high and low temperature cutoffs to each thermometer for the fits
        cut_table = Rth_table[Rth_table["T"] > T_cuts[0]]
        cut_table = cut_table[cut_table["T"] < T_cuts[1]]
        Rth_list = cut_table["Rth"]
        T_list =  cut_table["T"]
        if "Pt" not in thermometer: # LT or HT NbSi thermometer
            Rth_list = np.log(Rth_list)
            T_list =  np.log(T_list)
        if degree_dict:
            coeffs = np.polyfit(Rth_list, T_list, deg=degree_dict[thermometer])
        else:
            coeffs = np.polyfit(Rth_list, T_list, deg=2)
        poly = np.poly1d(coeffs)
        thermometer_polys[thermometer] = poly
    
    return thermometer_polys