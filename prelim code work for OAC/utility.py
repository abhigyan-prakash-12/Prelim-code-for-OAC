import xarray as xr
import numpy as np

def read_nc(nc_file):
    """
    args:
    -nc_file: file path for nc file to be read
    """
    xrds = xr.open_dataset(nc_file)
    lat = xrds['lat'].values
    lon = xrds['lon'].values
    pres = xrds['plev'].values
    co2 = xrds['CO2'].values
    h2o = xrds['H2O'].values
    nox = xrds['NOx'].values

    return {'lat':lat, 'lon': lon, 'pres':pres, 'co2': co2, 'nox': nox, 'h2o': h2o}

def unique_val(spc, arr):
    """
    function to get unique values of species with respect to a given coordinate
    args:
    -spc: array, species to get the unique values of 
    -arr: array of coordinate with respect to which we want the unique value"""
    arr_q = np.unique(arr)
    spc_arr = np.zeros_like(arr_q, dtype=np.float32)
    for i, v in enumerate(arr_q):
        spc_arr[i] = np.sum(spc[arr == v])
    return spc_arr

#def pres_to_alt(pressure_hpa):
    """
    approximate altitude from pressure values using barometric formula and standard values. works till 11km 
    however some values are above so may not be fully accurate
    args:
    -pressure_hpa: pressure value in hpa (standard for these nc files) that is to be converted in altitude

    output: tuple of altitude values (metres, feet)
    """
    alt_m = 44330 * (1 - (pressure_hpa / 1013.25) ** 0.1903)
    alt_ft = alt_m * 3.280
    return alt_m, alt_ft
def pres_to_alt(pressure_hpa):
    """
    Vectorized altitude estimation from pressure (hPa) using the barometric formula
    for both troposphere (<11 km) and stratosphere (11–20 km) under ISA conditions.

    Args:
        pressure_hpa: float or np.ndarray of pressure in hPa

    Returns:
        Tuple (alt_m, alt_ft): altitude in meters and feet
    """
    pressure_hpa = np.asarray(pressure_hpa)

    # Troposphere formula (P > 226.32 hPa → altitude < 11 km)
    alt_tropo = 44330 * (1 - (pressure_hpa / 1013.25) ** 0.1903)

    # Stratosphere formula (P ≤ 226.32 hPa → altitude ≥ 11 km)
    alt_strato = 11000 + 6341.62 * np.log(226.32 / pressure_hpa)

    # Combine based on pressure value
    alt_m = np.where(pressure_hpa > 226.32, alt_tropo, alt_strato)
    alt_ft = alt_m * 3.28084

    return alt_m, alt_ft
def scaled_emissions(aggco2, nc_file):
    """
    function to return scaled nc data, needs to be adpated for any species
    args-
    -aggco2: desired aggregated co2
    -nc_file: file path of nc file to be scaled
    """
    co2 = read_nc(nc_file)['co2']
    #xrds = xr.open_dataset(ref_nc_path)
    #co2 = xrds['CO2'].values
    tot_co2 = np.sum(co2)
    scf = aggco2/tot_co2
    scl_co2 = co2* scf
    return scl_co2