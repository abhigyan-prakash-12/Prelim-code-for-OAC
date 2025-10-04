#This file generate nc files for scaled emissions
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from utility import scaled_emissions
from plot_maps import apply_region_weights
from utility import read_nc
import os

def scaled_emissions_to_nc(input_nc, output_nc, aggco2, year):
    """
    args:
    -input_nc - base/reference nc file path
    -output_nc - name of the output nc 
    -aggco2- desired aggregated co2 emissions in the new nc file
    """
    os.makedirs("inputs", exist_ok=True)
    xrds = xr.open_dataset(input_nc)
    scaled_co2 = scaled_emissions(aggco2, input_nc)

    new_ds = xrds.copy()
   
    new_ds['CO2'] = (xrds['CO2'].dims, scaled_co2)
    new_ds.attrs['Inventory_Year'] = year
    new_ds['CO2'].attrs['long_name'] = 'CO2'
    new_ds['CO2'].attrs['units'] = 'kg'

    output_path = os.path.join("inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved scaled emissions to {output_path}")

def projected_emissions_5years_nc(base_nc, start_year, percent_change_per_5year):
    """
    argss:
    -base_nc: path to the base/referecne nc file
    -start_year: the starting year of the base file
    -percent_change_per_5year: float, % growth every 5 years
    """
    xrds = xr.open_dataset(base_nc)
    base_co2 = xrds['CO2'].values
    base_total = np.nansum(base_co2)

    current_total = base_total
    current_year = start_year

    for step in range(1, 7):  
        current_year += 5
    
        growth_factor = (1 + percent_change_per_5year / 100) 
        current_total *= growth_factor
        scaled_co2 = base_co2 * (current_total / base_total)

        new_ds = xrds.copy()
        new_ds.attrs['Inventory_Year'] = current_year
        new_ds['CO2'] = (xrds['CO2'].dims, scaled_co2)
        new_ds['CO2'].attrs['long_name'] = 'CO2'
        new_ds['CO2'].attrs['units'] = 'kg'
        new_ds['CO2'].attrs['description'] = f"Projected CO2 emissions for {current_year}"
        new_ds['CO2'].attrs['scaling_applied'] = f"{percent_change_per_5year:.2f}% per 5 years over {5 * step} years"

        out_name = f"gen_emi_inv_{current_year}.nc"
        new_ds.to_netcdf(out_name)
        print(f"Created: {out_name} with total CO2 = {np.sum(scaled_co2):.2e} kg")

