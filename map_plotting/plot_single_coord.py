import xarray as xr
import numpy as np
from utils.utility import read_nc, pres_to_alt, unique_val
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.utility import scaled_emissions


def plot_pres(file, sp1= None, sp2=None, sp3=None, scaled_sp=None):
    """sp1-co2, sp2-npx, sp3-h20"""
    pres = read_nc(file)['pres']
    co2 = read_nc(file)['co2'] if sp1 is not None else None
    nox = read_nc(file)['nox'] if sp2 is not None else None
    h2o = read_nc(file)['h2o'] if sp3 is not None else None
    pres_arr = np.unique(pres)
    co2_pres = unique_val(co2, pres) if sp1 is not None else None
    nox_pres = unique_val(nox, pres) if sp2 is not None else None
    h2o_pres = unique_val(h2o, pres) if sp3 is not None else None
    scaled_sp = unique_val(scaled_sp, pres) if scaled_sp is not None else None


    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(np.log10(co2_pres), pres_arr, color='green', label='CO2')
        #co2_pres/1000
    else:
        None
    #Todo:NOT PLOT NORMALISED SCALED VALUES SINCE ITS THE SAME AS THE ORIGINAL 
    if scaled_sp is not None:    
        ax.plot(np.log10(scaled_sp), pres_arr, color = 'black', label='Scaled_species') 
    else:
        None
    if sp2 is not None:   
        ax.plot(np.log10(nox_pres), pres_arr, color='red', label='NOx') 
        #
    else:
         None    
    if sp3 is not None:    
        ax.plot(np.log10(h2o_pres), pres_arr, color='blue', label='H2O')
    else:
         None
    ax.set_xlabel('Gas emissions')
    ax.set_ylabel('Pressure level (hPa)')
    ax.set_title('CO2, NOx and H2O emissions vs Pressure level')
    ax.invert_yaxis() 
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_alt(file, unit, sp1=None, sp2=None, sp3=None, scaled_sp=None):
    """
    function to plot species emission with respect to altitude
    args:
    -sp1: co2
    -sp2: nox
    -sp3: h2o
    -scaled_sp: scaled species
    -unit: km or ft
    """
    pres = read_nc(file)['pres']
    co2 = read_nc(file)['co2'] if sp1 is not None else None
    nox = read_nc(file)['nox'] if sp2 is not None else None
    h2o = read_nc(file)['h2o'] if sp3 is not None else None
    
    sp1 = unique_val(co2, pres) if sp1 is not None else None
    if sp2 is not None:
        sp2 = unique_val(nox, pres)
    if sp3 is not None:
        sp3 = unique_val(h2o, pres)
    if scaled_sp is not None:
        scaled_sp = unique_val(scaled_sp, pres)

    if unit == "km":
        alt = pres_to_alt(np.unique(pres))[0]/1000
    elif unit == "ft":
        alt = pres_to_alt(np.unique(pres))[1]

    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(np.log10(sp1), alt, color='green', label='CO2')
    if scaled_sp is not None:
        ax.plot(np.log10(scaled_sp), alt, color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(np.log10(sp2), alt, color='red', label='NOx')
    if sp3 is not None:
        ax.plot(np.log10(sp3), alt, color='blue', label='H2O')

    ax.set_xlabel('Gas emissions (log10)')
    if unit == "km":
        ax.set_ylabel('Altitude in km')
    if unit == "ft":
        ax.set_ylabel('Altitude in ft')
    ax.set_title('Gas emissions vs Altitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_alt_thresh(file, thresh, unit, mode=None, sp1=None, sp2=None, sp3=None, scaled_sp=None):
    """
    function to plot species against altitude above or below a threshold
    args:
    -sp1: co2
    -sp2: nox
    -sp3: h2o
    -scaled_sp: scaled species
    -thresh: threshold altitude
    -unit: km or ft
    """
    pres = read_nc(file)['pres']
    co2 = read_nc(file)['co2'] if sp1 is not None else None
    nox = read_nc(file)['nox'] if sp2 is not None else None
    h2o = read_nc(file)['h2o'] if sp3 is not None else None

    sp1 = unique_val(co2, pres) if sp1 is not None else None
    if sp2 is not None:
        sp2 = unique_val(nox, pres)
    if sp3 is not None:
        sp3 = unique_val(h2o, pres)
    if scaled_sp is not None:
        scaled_sp = unique_val(scaled_sp, pres)

    if unit == "km":
        alt = pres_to_alt(np.unique(pres))[0]/1000
    elif unit == "ft":
        alt = pres_to_alt(np.unique(pres))[1]

    if mode == "above":
        mask = alt >= thresh
    elif mode == "below":
        mask = alt <= thresh

    if mode is not None:
        alt = alt[mask]
        sp1 = sp1[mask] if sp1 is not None else None
        sp2 = sp2[mask] if sp2 is not None else None
        sp3 = sp3[mask] if sp3 is not None else None
        scaled_sp = scaled_sp[mask] if scaled_sp is not None else None

    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(np.log10(sp1), alt, color='green', label='CO2')
    if scaled_sp is not None:
        ax.plot(np.log10(scaled_sp), alt, color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(np.log10(sp2), alt, color='red', label='NOx')
    if sp3 is not None:
        ax.plot(np.log10(sp3), alt, color='blue', label='H2O')

    ax.set_xlabel('Gas emissions (log10)')
    if unit == "km":
        ax.set_ylabel('Altitude in km')
    if unit == "ft":
        ax.set_ylabel('Altitude in ft')
    ax.set_title('Gas emissions vs Altitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lon(file, sp1=None, sp2=None, sp3=None, scaled_sp=None):
    """
    args:
    -sp1: co2
    -sp2: nox
    -sp3-: h2o
    -scaled_sp: scaled species
    """
    lon = read_nc(file)['lon']
    co2 = read_nc(file)['co2'] if sp1 is not None else None
    nox = read_nc(file)['nox'] if sp2 is not None else None
    h2o = read_nc(file)['h2o'] if sp3 is not None else None

    co2_lon = unique_val(co2, lon) if sp1 is not None else None
    nox_lon = unique_val(nox, lon) if sp2 is not None else None
    h2o_lon = unique_val(h2o, lon) if sp3 is not None else None
    scaled_sp_lon = unique_val(scaled_sp, lon) if scaled_sp is not None else None

    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(co2_lon / 1e9, np.unique(lon), color='green', label='CO2')
    if scaled_sp is not None:
        ax.plot(np.log10(scaled_sp_lon), np.unique(lon), color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(nox_lon / 1e9, np.unique(lon), color='red', label='NOx')
    if sp3 is not None:
        ax.plot(h2o_lon / 1e9, np.unique(lon), color='blue', label='H2O')

    ax.set_xlabel('Emissions (log base 10)')
    ax.set_ylabel('Longitude')
    ax.set_title('Emissions vs Longitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lat(file, sp1=None, sp2=None, sp3=None, scaled_sp=None):
    """
    -sp1: co2
    -sp2: nox
    -sp3: h2o
    -scaled_sp: scaled species
    """
    lat = read_nc(file)['lat']
    co2 = read_nc(file)['co2'] if sp1 is not None else None
    nox = read_nc(file)['nox'] if sp2 is not None else None
    h2o = read_nc(file)['h2o'] if sp3 is not None else None

    co2_lat = unique_val(co2, lat) if sp1 is not None else None
    nox_lat = unique_val(nox, lat) if sp2 is not None else None
    h2o_lat = unique_val(h2o, lat) if sp3 is not None else None
    scaled_sp_lat = unique_val(scaled_sp, lat) if scaled_sp is not None else None
    #print(co2_lat / nox_lat)
    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(co2_lat / 1e9, np.unique(lat), color='green', label='CO2')
    if scaled_sp is not None:
        ax.plot(scaled_sp_lat / 1e9, np.unique(lat), color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(nox_lat/ 1e9, np.unique(lat), color='red', label='NOx')
    if sp3 is not None:
        ax.plot(h2o_lat/ 1e9, np.unique(lat), color='blue', label='H2O')

    ax.set_xlabel('Emissions (millions of tonnes)')
    ax.set_ylabel('Latitude')
    ax.set_title('Emissions vs Latitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_co2_nox_ratio(nc_file):
    file = read_nc(nc_file)
    co2 = file['co2']
    nox = file['nox']
    pres = file['pres']
    pres_un = np.unique(pres)
    alt = pres_to_alt(pres_un)[1]
    co2_un = unique_val(co2, pres)
    nox_un = unique_val(nox, pres)
    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot( co2_un/nox_un, alt, color = 'black', label='ratio of co2 and nox')
    ax.set_xlabel('ratio')
    ax.set_ylabel('altitude (ft)')
    ax.set_title('Ratio of CO2 and nNOx plotted against altitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lat_alt(nc_file, unit,threshold=50000):

    data = read_nc(nc_file)
    co2 = data['co2']
    pres = data['pres']
    lat = data['lat']

    alt = pres_to_alt(pres)[0] if unit == "m" else pres_to_alt(pres)[1]
    lat_unique = np.unique(lat)
    alt_unique = np.unique(alt)

    co2_lat_alt = np.zeros((len(lat_unique), len(alt_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, alt_val in enumerate(alt_unique):
            mask = (lat == lat_val) & (alt == alt_val)
            #co2_lat_alt[i, j] = np.sum(co2[mask])
            co2_sum = np.sum(co2[mask])
            co2_lat_alt[i, j] = co2_sum if co2_sum >= threshold else np.nan
    lat_grid, alt_grid = np.meshgrid(lat_unique, alt_unique, indexing='ij')
    #np.log10(co2_lat_alt)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot = ax.pcolormesh(lat_grid, alt_grid, np.log10(co2_lat_alt), cmap='plasma', shading='auto')
    ax.set_xlabel("Latitude")
    ax.set_ylabel(f"Altitude ({unit})")
    ax.set_title("CO₂ Emissions by Latitude and Altitude")
    #ax.set_ylim(0, 45000)  
    cbar = plt.colorbar(plot, ax=ax, label='log₁₀(CO₂ emissions in kg)', pad=0.02)
    
    plt.tight_layout()
    plt.show()

plot_lon("Inventories/emi_inv_2020.nc",sp1=1, sp2=None,sp3=None)