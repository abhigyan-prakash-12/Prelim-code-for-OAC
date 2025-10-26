#This file generate nc files for scaled emissions
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from utils.utility import scaled_emissions, scaled_emissions_co2
from map_plotting.plot_maps import apply_region_weights
from utils.utility import read_nc
import os


# no longer functional 
def scaled_emissions_to_nc(input_nc, output_nc, aggco2, year):
    """
    args:
    -input_nc - base/reference nc file path
    -output_nc - name of the output nc 
    -aggco2- desired aggregated co2 emissions in the new nc file
    """
    os.makedirs("inputs", exist_ok=True)
    xrds = xr.open_dataset(input_nc)
    scaled_co2 = scaled_emissions_co2(aggco2, input_nc)
    
    new_ds = xrds.copy()
    
    new_ds['CO2'] = (xrds['CO2'].dims, scaled_co2)
    new_ds.attrs['Inventory_Year'] = year
    new_ds['CO2'].attrs['long_name'] = 'CO2'
    new_ds['CO2'].attrs['units'] = 'kg'

    output_path = os.path.join("inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved scaled emissions to {output_path}")


def scaled_emissions_to_nc_complete(input_nc, output_nc, spc, year):
    """
    args:
    -input_nc - base/reference nc file path
    -output_nc - name of the output nc 
    -spc- desired aggregated eissions in the order [co2, nox, h2o, dist]
    """
    # Need to add new species
    os.makedirs("inputs", exist_ok=True)
    xrds = xr.open_dataset(input_nc)
    scaled_emi = scaled_emissions(spc, input_nc)

    new_ds = xrds.copy()
    new_ds.attrs['Inventory_Year'] = year

    new_ds['CO2'] = (xrds['CO2'].dims, scaled_emi[0])
    new_ds['CO2'].attrs['long_name'] = 'CO2'
    new_ds['CO2'].attrs['units'] = 'kg'
    
    new_ds['H2O'] = (xrds['H2O'].dims, scaled_emi[3])
    new_ds['H2O'].attrs['long_name'] = 'H2O'
    new_ds['H2O'].attrs['units'] = 'kg'

    new_ds['NOx'] = (xrds['NOx'].dims, scaled_emi[2])
    new_ds['NOx'].attrs['long_name'] = 'NOx'
    new_ds['NOx'].attrs['units'] = 'kg'

    new_ds['distance'] = (xrds['distance'].dims, scaled_emi[1])
    new_ds['distance'].attrs['long_name'] = 'distance flown'
    new_ds['distance'].attrs['units'] = 'km'

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

def apply_region_weights_efficient(lat_vals, lon_vals, co2_vals, region_weights):
    region_bounds = {
        "North America": [-180, -45, 12, 70],
        "Atlantic region": [-45, -25, 12, 70],
        "South America": [-90, -25, -60, 12],
        "Pacific": [-180, -90, -60, 12],
        "Far North": [-180, 60, 70, 90],
        "Europe": [-25, 60, 35, 70],
        "Africa and Middle East": [-25, 60, -35, 35],
        "Asia": [60, 180, -10, 90],
        "Oceania": [110, 180, -50, -10]
    }

    lon_converted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
    weighted_co2 = co2_vals.copy().astype(np.float32)
    
    for region, bounds in region_bounds.items():
        if region not in region_weights:
            continue
        lon_min, lon_max, lat_min, lat_max = bounds
        weight = region_weights[region]
        region_mask = (
            (lon_converted >= lon_min) & (lon_converted <= lon_max) &
            (lat_vals >= lat_min) & (lat_vals <= lat_max)
        )
        weighted_co2[region_mask] *= weight

    return weighted_co2

def scaled_emissions_to_nc_with_weights(input_nc, output_nc, aggco2, year, region_weights, chunk_size=50000):
    os.makedirs("inputs", exist_ok=True)
    
    ds = xr.open_dataset(input_nc)
    total_points = len(ds['index'])
    
    original_total = float(ds['CO2'].sum())
    scale_factor = aggco2 / original_total
    weighted_co2 = np.zeros(total_points, dtype=np.float32)
    
    for start_idx in range(0, total_points, chunk_size):
        end_idx = min(start_idx + chunk_size, total_points)
        lat_chunk = ds['lat'].isel(index=slice(start_idx, end_idx)).values
        lon_chunk = ds['lon'].isel(index=slice(start_idx, end_idx)).values
        co2_chunk = ds['CO2'].isel(index=slice(start_idx, end_idx)).values * scale_factor
        weighted_chunk = apply_region_weights_efficient(lat_chunk, lon_chunk, co2_chunk, region_weights)
        weighted_co2[start_idx:end_idx] = weighted_chunk
    
    current_total = np.sum(weighted_co2)
    final_co2 = weighted_co2 * (aggco2 / current_total)
    
    new_ds = ds.copy()
    new_ds['CO2'] = (('index',), final_co2)
    new_ds.attrs['Inventory_Year'] = year
    new_ds['CO2'].attrs['long_name'] = 'CO2'
    new_ds['CO2'].attrs['units'] = 'kg'
    new_ds.attrs['region_weights'] = str(region_weights)

    output_path = os.path.join("inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved weighted scaled emissions to {output_path}")
    ds.close()
    #return output_path


def apply_region_weights_cont(lat_vals, lon_vals, dist_vals, region_weights):
    region_bounds = {
        "North America": [-180, -45, 12, 70],
        "Atlantic region": [-45, -25, 12, 70],
        "South America": [-90, -25, -60, 12],
        "Pacific": [-180, -90, -60, 12],
        "Far North": [-180, 60, 70, 90],
        "Europe": [-25, 60, 35, 70],
        "Africa and Middle East": [-25, 60, -35, 35],
        "Asia": [60, 180, -10, 90],
        "Oceania": [110, 180, -50, -10]
    }

    lon_converted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
    weighted_dist = dist_vals.copy().astype(np.float32)
    
    for region, bounds in region_bounds.items():
        if region not in region_weights:
            continue
        lon_min, lon_max, lat_min, lat_max = bounds
        weight = region_weights[region]
        region_mask = (
            (lon_converted >= lon_min) & (lon_converted <= lon_max) &
            (lat_vals >= lat_min) & (lat_vals <= lat_max)
        )
        weighted_dist[region_mask] *= weight

    return weighted_dist

def scaled_emissions_to_nc_with_weights_cont(input_nc, output_nc, aggdist, year, region_weights, chunk_size=50000):
    os.makedirs("inputs", exist_ok=True)
    
    ds = xr.open_dataset(input_nc)
    total_points = len(ds['index'])
    
    original_total = float(ds['distance'].sum())
    scale_factor = aggdist / original_total
    weighted_cont = np.zeros(total_points, dtype=np.float32)
    
    for start_idx in range(0, total_points, chunk_size):
        end_idx = min(start_idx + chunk_size, total_points)
        lat_chunk = ds['lat'].isel(index=slice(start_idx, end_idx)).values
        lon_chunk = ds['lon'].isel(index=slice(start_idx, end_idx)).values
        dist_chunk = ds['distance'].isel(index=slice(start_idx, end_idx)).values * scale_factor
        weighted_chunk = apply_region_weights_efficient(lat_chunk, lon_chunk, dist_chunk, region_weights)
        weighted_cont[start_idx:end_idx] = weighted_chunk
    
    current_total = np.sum(weighted_cont)
    final_cont = weighted_cont * (aggdist / current_total)
    
    new_ds = ds.copy()
    new_ds['distance'] = (('index',), final_cont)
    new_ds.attrs['Inventory_Year'] = year
    new_ds['distance'].attrs['long_name'] = 'distance flown'
    new_ds['distance'].attrs['units'] = 'km'
    new_ds.attrs['region_weights'] = str(region_weights)

    output_path = os.path.join("inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved weighted scaled emissions to {output_path}")
    ds.close()
    #return output_path

