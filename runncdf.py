
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
"""netcdf_file = "Inventories/emi_inv_2025.nc"
xrds = xr.open_dataset(netcdf_file)
print(np.sum(xrds['distance'].values))
print(np.sum(xrds['H2O'].values))
print(np.sum(xrds['NOx'].values))
"""
"""netcdf_file = "example/emi_inv_2025.nc"
xrds = xr.open_dataset(netcdf_file)
#print(xrds)
# note that the emissions are measured in kg. total no of values-606168
lat = xrds['lat'].values
[-62. -61. -60. -59. -58. -57. -56. -55. -54. -53. -52. -51. -50. -49.
 -48. -47. -46. -45. -44. -43. -42. -41. -40. -39. -38. -37. -36. -35.
 -34. -33. -32. -31. -30. -29. -28. -27. -26. -25. -24. -23. -22. -21.
 -20. -19. -18. -17. -16. -15. -14. -13. -12. -11. -10.  -9.  -8.  -7.
  -6.  -5.  -4.  -3.  -2.  -1.   0.   1.   2.   3.   4.   5.   6.   7.
   8.   9.  10.  11.  12.  13.  14.  15.  16.  17.  18.  19.  20.  21.
  22.  23.  24.  25.  26.  27.  28.  29.  30.  31.  32.  33.  34.  35.
  36.  37.  38.  39.  40.  41.  42.  43.  44.  45.  46.  47.  48.  49.
  50.  51.  52.  53.  54.  55.  56.  57.  58.  59.  60.  61.  62.  63.
  64.  65.  66.  67.  68.  69.  70.  71.  72.  73.  74.  75.  76.  77.
  78.  79.  80.  81.  82.  83.  84.  85.  86.  87.  88.] 
  unique lat values. Len-151 

co2 = xrds['CO2'].values #co2 shape is 664350
lon = xrds['lon'].values
[0. 1. 2. 3. 4. 5. ... 358. 359.] 
lon values- 0 to 359. Len-360
h2o = xrds['H2O'].values
nox = xrds['NOx'].values
fuel = xrds['fuel'] # fuel used
dist = xrds['distance'] #distance flown
pres = xrds['plev'].values # pressure over each point. Measured in hPa
[ 148.2   155.4   163.    171.    179.4   188.2   197.5   207.1   217.3
  239.1   250.6   262.6   275.1   278.    288.1   301.9   315.4   329.9
  344.9   360.4   376.5   393.2   410.4   428.3   446.8   466.    485.8
  506.3   527.5   549.4   572.1   595.5   619.6   644.6   670.3   696.9
  724.4   752.7   781.9   812.    843.1   908.1   942.1   942.14  977.17
 1013.25]
 pressure levels. Len-46

#co2_log = np.log10(co2)
#h2o_log = np.log10(h2o)
#nox_log = np.log10(nox)"""
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

#co2 = read_nc('Inventories/emi_inv_2020.nc')['co2']
#h2o = read_nc('Inventories/emi_inv_2020.nc')['h2o']
#print(co2)
#print(co2[0]/h2o[0])
def pres_to_altog(pressure_hpa):
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
def plot_alt_threshog(file, thresh, unit, mode=None, sp1=None, sp2=None, sp3=None, scaled_sp=None):
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
        alt = pres_to_altog(np.unique(pres))[0]/1000
    elif unit == "ft":
        alt = pres_to_altog(np.unique(pres))[1]

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

def scaled_emissions_to_nc(input_nc, output_nc, aggco2):
    """
    args:
    -input_nc - base/reference nc file path
    -output_nc - name of the output nc 
    -aggco2- desired aggregated co2 emissions in the new nc file
    """
    xrds = xr.open_dataset(input_nc)
    scaled_co2 = scaled_emissions(aggco2, input_nc)

    new_ds = xrds.copy()
    new_ds['CO2'] = (xrds['CO2'].dims, scaled_co2)

    new_ds.to_netcdf(output_nc)
    print(f"Saved scaled emissions to {output_nc}")

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
        new_ds['CO2'] = (xrds['CO2'].dims, scaled_co2)
        new_ds['CO2'].attrs['description'] = f"Projected CO2 emissions for {current_year}"
        new_ds['CO2'].attrs['scaling_applied'] = f"{percent_change_per_5year:.2f}% per 5 years over {5 * step} years"

        out_name = f"emi_inv_{current_year}.nc"
        new_ds.to_netcdf(out_name)
        print(f"Created: {out_name} with total CO2 = {np.sum(scaled_co2):.2e} kg")

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


#PLOTTING FUNCTIONS
#can confirm h2o and co2 have the exact same emissions, Nox differs by a factor
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
        ax.plot(scaled_sp_lon / 1e9, np.unique(lon), color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(nox_lon / 1e9, np.unique(lon), color='red', label='NOx')
    if sp3 is not None:
        ax.plot(h2o_lon / 1e9, np.unique(lon), color='blue', label='H2O')

    ax.set_xlabel('Emissions (million tonnes)')
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

    fig2, ax = plt.subplots(figsize=(6, 5))
    if sp1 is not None:
        ax.plot(co2_lat / 1e9, np.unique(lat), color='green', label='CO2')
    if scaled_sp is not None:
        ax.plot(scaled_sp_lat / 1e9, np.unique(lat), color='black', label='Scaled Species')
    if sp2 is not None:
        ax.plot(nox_lat / 1e9, np.unique(lat), color='red', label='NOx')
    if sp3 is not None:
        ax.plot(h2o_lat / 1e9, np.unique(lat), color='blue', label='H2O')

    ax.set_xlabel('Emissions (million tonnes)')
    ax.set_ylabel('Latitude')
    ax.set_title('Emissions vs Latitude')
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_co2_emissions_key(file, threshold = 10000):
    """
    generates interactive map that shows emissions at diiferent altitudes
    args
    -file: nc file path
    -threshold- default value to make remove outlier data
    """
    lat = read_nc(file)['lat']
    co2 = read_nc(file)['co2'] 
    lon = read_nc(file)['lon'] 
    pres = read_nc(file)['pres']
    co2_masked = np.where(co2 >= threshold, co2, np.nan)
    co2_log = np.log10(co2_masked)
    #co2_log = np.log10(co2)

    pres_unique = np.unique(pres)
    n_levels = len(pres_unique)
    
    level_indices = [np.where(pres == p)[0] for p in pres_unique]
    print(len(level_indices))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    current_level = [0]
    idx0 = level_indices[current_level[0]]
    sc = ax.scatter(lon[idx0], lat[idx0], c=co2_log[idx0], cmap='plasma',s=2, transform=ccrs.PlateCarree(),vmin=10)
    cb = plt.colorbar(sc, ax=ax, label='log10(CO2 emissions in kg)')
    ax.axes.coastlines()
    ax.axes.stock_img()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_global() 
    #ax.set_xlim([0, 359])  
    #ax.set_ylim([np.min(lat), np.max(lat)])  
    title = ax.set_title("")

    def update_plot(level):
        idx = level_indices[level]
        sc.set_offsets(np.column_stack((lon[idx], lat[idx])))
        sc.set_array(co2_log[idx])
        pres_val = pres_unique[level]
        alt_m = pres_to_alt(pres_val)[0]
        alt_ft = pres_to_alt(pres_val)[1] 
        title.set_text(f"CO2 Emissions at Pressure: {pres_val:.2f} hPa ( aprox {alt_m:.1f} m or (approx {alt_ft:.1f}))")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'up':
            current_level[0] = (current_level[0] + 1) % n_levels
            update_plot(current_level[0])
        elif event.key == 'down':
            current_level[0] = (current_level[0] - 1) % n_levels
            update_plot(current_level[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()

def plot_co2_emissions(file):
    lat = read_nc(file)['lat']
    co2 = read_nc(file)['co2'] 
    lon = read_nc(file)['lon'] 
    pres = read_nc(file)['pres']

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(lon_grid, lat_grid, np.log10(co2_latlon), cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    plot.axes.set_global()
    plot.axes.coastlines()
    plot.axes.stock_img()
    ax.gridlines()
    plt.colorbar(plot, ax=ax, orientation='vertical', label='Log10 CO2 Emissions', pad=0.05)
    plt.title("Global CO2 Emissions by Latitude and Longitude")
    plt.tight_layout()
    plt.show()



def co2_reshape(file):
    lat = read_nc(file)['lat']
    co2 = read_nc(file)['co2'] 
    lon = read_nc(file)['lon'] 
    pres = read_nc(file)['pres']

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    pres_unique = np.unique(pres)
    co2_3d = np.zeros((len(lat_unique), len(lon_unique), len(pres_unique)))

    lat_idx = {val: idx for idx, val in enumerate(lat_unique)}
    lon_idx = {val: idx for idx, val in enumerate(lon_unique)}
    pres_idx = {val: idx for idx, val in enumerate(pres_unique)}

    for i in range(len(co2)):
        lat_val = lat[i]
        lon_val = lon[i]
        pres_val = pres[i]

        i_lat = lat_idx.get(lat_val)
        i_lon = lon_idx.get(lon_val)
        i_pres = pres_idx.get(pres_val)

        if i_lat is not None and i_lon is not None and i_pres is not None:
            co2_3d[i_lat, i_lon, i_pres] += co2[i]
    print(co2_3d)
    return co2_3d
#problem with reshape- total length is 151*360*45 = 2446200
# However the length of data available - 664350 which is only 27% of the total length- printing or plotting is useless


#def plot_co2_threshold(file,threshold_percentile, threshold_alt, mode="below", unit="m"):
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    pres = data['pres']
    co2 = data['co2']

    alt_km, alt_ft = pres_to_alt(pres)
    alt = alt_ft if unit == "ft" else alt_km * 1000  

    if mode == "below":
        alt_mask = alt <= threshold_alt
    elif mode == "above":
        alt_mask = alt > threshold_alt

    lat = lat[alt_mask]
    lon = lon[alt_mask]
    co2 = co2[alt_mask]

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    # Mask zeros so they don't appear black
    #co2_latlon_masked = np.ma.masked_where(co2_latlon == 0, co2_latlon)
    log_co2 = np.log10(co2_latlon)

    # Plot
    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(
        lon_grid, lat_grid, log_co2,
        cmap='plasma', shading='auto',
        transform=ccrs.PlateCarree()
    )
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()
    cbar = plt.colorbar(plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Log10 CO2 Emissions (arbitrary units)")
    unit_str = "m" if unit == "m" else "ft"

    threshold = np.percentile(co2_latlon, threshold_percentile)
    airport_mask = co2_latlon >= threshold
    #print(airport_mask)
    for i in range(co2_latlon.shape[0]):
        for j in range(co2_latlon.shape[1]):
            if airport_mask[i, j]:
                ax.plot(lon_unique[j], lat_unique[i], 'o', color='cyan', markersize=3, transform=ccrs.PlateCarree())

    plt.title(f"CO₂ Emissions {mode} {threshold_alt} {unit_str}")
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


def co2_emissions_airport(file, threshold_percentile=99):
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    co2 = data['co2']

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(lon_grid, lat_grid, np.log10(co2_latlon), cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()

    threshold = np.percentile(co2_latlon, threshold_percentile)
    airport_mask = co2_latlon >= threshold
    #print(airport_mask)
    for i in range(co2_latlon.shape[0]):
        for j in range(co2_latlon.shape[1]):
            if airport_mask[i, j]:
                ax.plot(lon_unique[j], lat_unique[i], 'o', color='cyan', markersize=3, transform=ccrs.PlateCarree())
    
    top_polluting_airports = [
        {"code": "DXB", "lat": 25.267, "lon": 55.3643},
        {"code": "LHR", "lat": 51.4680, "lon": -0.4551},
        {"code": "LAX", "lat": 33.9422, "lon": -118.4036},
        {"code": "JFK", "lat": 40.6446, "lon": -73.7797},
        {"code": "CDG", "lat": 49.0079, "lon": 2.5508},
        {"code": "PEK", "lat": 40.0799, "lon": 116.6031},
        {"code": "HKG", "lat": 22.3193, "lon": 114.1694},
        {"code": "SIN", "lat": 1.3586, "lon": 103.9899},
        {"code": "FRA", "lat": 50.0354, "lon": 8.5518},
        {"code": "ICN", "lat": 37.4587, "lon": 126.4420},
    ]

    for airport in top_polluting_airports:
        ax.plot(airport["lon"], airport["lat"], marker='x', color='red', markersize=10, transform=ccrs.PlateCarree())
        ax.text(airport["lon"] + 1, airport["lat"], airport["code"], fontsize=6, color='white',
                transform=ccrs.PlateCarree(), ha='left', va='center')
    plt.colorbar(plot, ax=ax, orientation='vertical', label='Log10 CO2 Emissions', pad=0.05)
    plt.title("Global CO₂ Emissions (Highlighting Potential Airports)")
    plt.tight_layout()
    plt.show()

def co2_emissions_airport_top_x(file, top_n=10):
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    co2 = data['co2']

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

    flat_indices = np.argsort(co2_latlon.ravel())[::-1][:top_n]
    top_coords = np.unravel_index(flat_indices, co2_latlon.shape)
    print(f"Top {top_n} CO₂ emission points:")
    print(f"{'Rank':<5} {'Latitude':<10} {'Longitude':<10} {'Emissions (kg)':>15}")
    print("-" * 45)
    for rank, (i, j) in enumerate(zip(*top_coords), 1):
        lat_val = lat_unique[i]
        lon_val = lon_unique[j]
        emission_val = co2_latlon[i, j]
        print(f"{rank:<5} {lat_val:<10.4f} {lon_val:<10.4f} {emission_val:>15.2f}")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(lon_grid, lat_grid, np.log10(co2_latlon), cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()

    for i, j in zip(*top_coords):
        ax.plot(lon_unique[j], lat_unique[i], 'o', color='cyan', markersize=6, transform=ccrs.PlateCarree(), zorder=10)

    """top_polluting_airports = [
        {"code": "DXB", "lat": 25.267, "lon": 55.3643},
        {"code": "LHR", "lat": 51.4680, "lon": -0.4551},
        {"code": "LAX", "lat": 33.9422, "lon": -118.4036},
        {"code": "JFK", "lat": 40.6446, "lon": -73.7797},
        {"code": "CDG", "lat": 49.0079, "lon": 2.5508},
        {"code": "PEK", "lat": 40.0799, "lon": 116.6031},
        {"code": "HKG", "lat": 22.3193, "lon": 114.1694},
        {"code": "SIN", "lat": 1.3586, "lon": 103.9899},
        {"code": "FRA", "lat": 50.0354, "lon": 8.5518},
        {"code": "ICN", "lat": 37.4587, "lon": 126.4420},
    ]

    for airport in top_polluting_airports:
        ax.plot(airport["lon"], airport["lat"], marker='x', color='red', markersize=15,
                transform=ccrs.PlateCarree(), zorder=11)
        ax.text(airport["lon"] + 1, airport["lat"], airport["code"], fontsize=7, color='white',
                transform=ccrs.PlateCarree(), ha='left', va='center', zorder=12)"""

    plt.colorbar(plot, ax=ax, orientation='vertical', label='Log10 CO2 Emissions', pad=0.05)
    plt.title(f"Global CO₂ Emissions – Top {top_n} Hotspots ")
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
    ax.set_title("CO2 Emissions by Latitude and Altitude")
    #ax.set_ylim(0, 45000)  
    cbar = plt.colorbar(plot, ax=ax, label='log10(CO2 emissions in kg)', pad=0.02)
    
    plt.tight_layout()
    plt.show()

def apply_region_weights(emission_map, lat_vals, lon_vals, region_weights):
    """region_bounds = {
        "North America": [-170, -50, 10, 70],
        "South America": [-90, -30, -60, 10],
        "Europe": [-25, 60, 35, 70],
        "Africa": [-20, 50, -35, 35],
        "Asia": [60, 180, -10, 80],
        "Oceania": [110, 180, -50, -10]
    }"""
    region_bounds = {
    "North America": [-170, -50, 10, 70],
    "South America": [-90, -30, -60, 10],
    "Europe": [-25, 40, 35, 70],  # Narrowed to avoid Middle East
    "Africa": [-20, 50, -35, 10],  # Upper bound reduced
    "Asia": [60, 180, 10, 80],  # Lower bound increased
    "Oceania": [110, 180, -50, -10],
    "Middle East": [40, 60, 10, 40],  # Between Europe, Africa, and Asia
    "Atlantic Ocean": [-50, -20, -60, 70],  # Between Americas and Europe/Africa
    "Far North": [-180, 180, 70, 90],  # Arctic region
    "Far South": [-180, 180, -90, -60]  # Antarctic region
    }

    adjusted_map = emission_map.copy()
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')

    for region, bounds in region_bounds.items():
        if region not in region_weights:
            continue
        lon_min, lon_max, lat_min, lat_max = bounds
        weight = region_weights[region]

        region_mask = (
            (lon_grid >= lon_min) & (lon_grid <= lon_max) &
            (lat_grid >= lat_min) & (lat_grid <= lat_max)
        )

        adjusted_map[region_mask] *= weight

    return adjusted_map


def project_emissions_from_nc(file, total_emission, region_weights):
    data = read_nc(file)
    co2 = data['co2']
    lat = data['lat']
    lon = data['lon']
    lon = np.where(lon > 180, lon - 360, lon)
    aggco2 = np.sum(co2)
    print(f"original co2 emission: {aggco2:.2e} kg")
    
    lat_bins = np.linspace(-90, 90, 181)   
    lon_bins = np.linspace(-180, 180, 361)

    
    co2_map, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins], weights=co2)
    co2_map /= np.sum(co2_map)   # normalise    
    co2_map *= total_emission              

    
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

    
    weighted_map = apply_region_weights(co2_map, lat_centers, lon_centers, region_weights)

    
    weighted_map *= total_emission / np.sum(weighted_map)

    print(f"Total emissions after weighting: {np.sum(weighted_map):.2e} kg")

    region_bounds = {
    "North America": [-170, -50, 10, 70],
    "South America": [-90, -30, -60, 10],
    "Europe": [-20, 40, 35, 70],  # Narrowed to avoid Middle East
    "Africa": [-20, 50, -35, 10],  # Upper bound reduced
    "Asia": [60, 180, 10, 70],  # Lower bound increased
    "Oceania": [110, 180, -50, -10],
    "Middle East": [40, 60, 10, 40],  # Between Europe, Africa, and Asia
    "Atlantic Ocean": [-50, -20, -60, 70],  # Between Americas and Europe/Africa
    "Far North": [-180, 180, 70, 90],  # Arctic region
    "Far South": [-180, 180, -90, -60]  # Antarctic region
    }
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')

    for region, (lon_min, lon_max, lat_min, lat_max) in region_bounds.items():
        region_mask = (
            (lon_grid >= lon_min) & (lon_grid <= lon_max) &
            (lat_grid >= lat_min) & (lat_grid <= lat_max)
        )
        region_sum = np.sum(weighted_map[region_mask])
        print(f"{region}: {region_sum:.2e} kg")
    return lon_bins, lat_bins, weighted_map


def plot_weighted_emissions(file, total_emission, region_weights):
    lon_edges, lat_edges, emissions = project_emissions_from_nc(file, total_emission, region_weights)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)
    masked_emissions = np.ma.masked_where(emissions <= 0, emissions)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(lon_edges, lat_edges, np.log10(masked_emissions), cmap='plasma', shading='auto', transform=ccrs.PlateCarree())

    ax.set_global()
    ax.coastlines()

    region_bounds = {
    "North America": [-170, -50, 10, 70],
    "South America": [-90, -30, -60, 10],
    "Europe": [-25, 40, 35, 70],  # Narrowed to avoid Middle East
    "Africa": [-20, 50, -35, 10],  # Upper bound reduced
    "Asia": [60, 180, 10, 80],  # Lower bound increased
    "Oceania": [110, 180, -50, -10],
    "Middle East": [40, 60, 10, 40],  # Between Europe, Africa, and Asia
    "Atlantic Ocean": [-50, -20, -60, 70],  # Between Americas and Europe/Africa
    "Far North": [-180, 180, 70, 90],  # Arctic region
    "Far South": [-180, 180, -90, -60]  # Antarctic region
    }

    for region, (lon_min, lon_max, lat_min, lat_max) in region_bounds.items():
        ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                               linewidth=1.2, edgecolor='cyan', facecolor='none', transform=ccrs.PlateCarree()))
        ax.text((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, region,
                transform=ccrs.PlateCarree(), color='cyan', fontsize=8, ha='center')

    plt.colorbar(plot, ax=ax, label='Log10 CO2 Emissions (kg)', orientation='vertical', pad=0.05)
    plt.title("Weighted CO2 Emission Projection Based on Real Data Template")
    plt.tight_layout()
    plt.show()

def simulate_emission_growth(file, base_emission, region_weights, growth_factors):
    """
    Simulate emissions growth over time based on multiplicative growth factors.
    Returns the emissions map at each time step.
    """
    lon_edges, lat_edges, _ = project_emissions_from_nc(file, base_emission, region_weights)

    emissions_over_time = []
    for multiplier in growth_factors:
        _, _, weighted_map = project_emissions_from_nc(file, base_emission * multiplier, region_weights)
        emissions_over_time.append(weighted_map)
    return lon_edges, lat_edges, emissions_over_time


def animate_emissions_growth(lon_edges, lat_edges, emissions_over_time, interval=1000):
    """
    Animate emissions maps over time.
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Compute shared color scale limits (log scale)
    all_data = np.stack(emissions_over_time)
    masked_all = np.ma.masked_less_equal(all_data, 0)
    vmin = np.log10(masked_all.min())
    vmax = np.log10(masked_all.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def update(frame):
        ax.clear()
        ax.set_global()
        ax.coastlines()
        emission_map = np.ma.masked_less_equal(emissions_over_time[frame], 0)
        plot = ax.pcolormesh(
            lon_edges, lat_edges, np.log10(emission_map),
            cmap='plasma', shading='auto', norm=norm,
            transform=ccrs.PlateCarree()
        )
        ax.set_title(f"Projected CO₂ Emissions (Year {2025 + 5 * frame})")

    anim = FuncAnimation(fig, update, frames=len(emissions_over_time), interval=interval)
    plt.show()






def plot_scaled(aggco2, file):
    scl_co2 = scaled_emissions(aggco2, file)
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    co2 = data['co2']

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))
    scl_co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])
            scl_co2_latlon[i, j] = np.sum(scl_co2[mask])

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

    total_original = np.sum(co2)
    total_scaled = np.sum(scl_co2)

    # Take log10 safely
    co2_log = np.log10(np.where(co2_latlon > 0, co2_latlon, np.nan))
    scl_co2_log = np.log10(np.where(scl_co2_latlon > 0, scl_co2_latlon, np.nan))

    # Define common vmin and vmax for shared color scale
    vmin = np.nanmin([co2_log, scl_co2_log])
    vmax = np.nanmax([co2_log, scl_co2_log])

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # Add space for shared colorbar

    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    plot_ax = ax.pcolormesh(lon_grid, lat_grid, co2_log, cmap='plasma',
                            shading='auto', transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    ax.set_title("Original CO2 Emissions (log10)", fontsize=10)
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()

    bx = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    plot_bx = bx.pcolormesh(lon_grid, lat_grid, scl_co2_log, cmap='plasma',
                            shading='auto', transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    bx.set_title("Scaled CO2 Emissions (log10)", fontsize=10)
    bx.set_global()
    bx.coastlines()
    bx.stock_img()
    bx.gridlines()

    # Shared colorbar on the right
    cax = fig.add_subplot(gs[2])
    cbar = plt.colorbar(plot_bx, cax=cax, orientation='vertical')
    cbar.set_label('Log10 CO2 Emissions')

    fig.suptitle(f"Original Map vs. Scaled Map\n"
                 f"Total Original: {total_original:.2e} kg | "
                 f"Total Scaled: {total_scaled:.2e} kg",
                 fontsize=12, y=1.05)

    plt.tight_layout()
    plt.show()
def plot_co2_threshold_topx(file, threshold_alt, mode="below", unit="m", top_x=5):
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    pres = data['pres']
    co2 = data['co2']

    alt_km, alt_ft = pres_to_alt(pres)
    alt = alt_ft if unit == "ft" else alt_km * 1000  

    if mode == "below":
        alt_mask = alt <= threshold_alt
    elif mode == "above":
        alt_mask = alt > threshold_alt

    lat = lat[alt_mask]
    lon = lon[alt_mask]
    co2 = co2[alt_mask]

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    log_co2 = np.log10(np.where(co2_latlon > 0, co2_latlon, np.nan))

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(
        lon_grid, lat_grid, log_co2,
        cmap='plasma', shading='auto',
        transform=ccrs.PlateCarree()
    )
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()
    cbar = plt.colorbar(plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Log10 CO2 Emissions (arbitrary units)")
    unit_str = "m" if unit == "m" else "ft"

    
    co2_flat = co2_latlon.flatten()
    top_indices = np.argsort(co2_flat)[-top_x:]  
    for idx in top_indices:
        i, j = np.unravel_index(idx, co2_latlon.shape)
        ax.plot(lon_unique[j], lat_unique[i], 'o', color='cyan', markersize=5, transform=ccrs.PlateCarree())
    

    plt.title(f"CO₂ Emissions {mode} {threshold_alt} {unit_str} + Top {top_x} Emitters")
    plt.tight_layout()
    plt.show()
def plot_co2_threshold(file, threshold_alt, mode="below", unit="m", top_x=5):
    data = read_nc(file)
    lat = data['lat']
    lon = data['lon']
    pres = data['pres']
    co2 = data['co2']

    alt_km, alt_ft = pres_to_alt(pres)
    alt = alt_ft if unit == "ft" else alt_km * 1000  

    if mode == "below":
        alt_mask = alt <= threshold_alt
    elif mode == "above":
        alt_mask = alt > threshold_alt

    lat = lat[alt_mask]
    lon = lon[alt_mask]
    co2 = co2[alt_mask]

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    co2_latlon = np.zeros((len(lat_unique), len(lon_unique)))

    for i, lat_val in enumerate(lat_unique):
        for j, lon_val in enumerate(lon_unique):
            mask = (lat == lat_val) & (lon == lon_val)
            co2_latlon[i, j] = np.sum(co2[mask])

    log_co2 = np.log10(np.where(co2_latlon > 0, co2_latlon, np.nan))

    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(
        lon_grid, lat_grid, log_co2,
        cmap='plasma', shading='auto',
        transform=ccrs.PlateCarree()
    )
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()
    cbar = plt.colorbar(plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Log10 CO2 Emissions (arbitrary units)")
    unit_str = "m" if unit == "m" else "ft"

    # Highlight top X emissive regions
    co2_flat = co2_latlon.flatten()
    top_indices = np.argsort(co2_flat)[-top_x:]  # Get indices of top X values
    top_coords = []  # To store coordinates and emission values

    for idx in reversed(top_indices):  # Reversed to print in descending order
        i, j = np.unravel_index(idx, co2_latlon.shape)
        lat_val = lat_unique[i]
        lon_val = lon_unique[j]
        emission_val = co2_latlon[i, j]
        top_coords.append((lat_val, lon_val, emission_val))
        ax.plot(lon_val, lat_val, 'o', color='cyan', markersize=5, transform=ccrs.PlateCarree())
        ax.text(lon_val, lat_val, f"{emission_val:.1e}", 
                color='white', fontsize=6, transform=ccrs.PlateCarree(), ha='center', va='center')

    # Print coordinates
    print(f"\nTop {top_x} CO₂ Emitting Regions ({mode} {threshold_alt}{unit_str}):")
    for idx, (lat_val, lon_val, emission_val) in enumerate(top_coords, 1):
        print(f"{idx}. Lat: {lat_val:.2f}, Lon: {lon_val:.2f}, Emissions: {emission_val:.2e}")

    plt.title(f"CO₂ Emissions {mode} {threshold_alt} {unit_str} + Top {top_x} Emitters")
    plt.tight_layout()
    plt.show()

#def main():
  

  #plot_lat_alt("Inventories/emi_inv_2025.nc", unit = 'm') 

  #plot_projected_emissions("example/emi_inv_2025.nc", total_emission=1e8, region_weights=region_weights)
  #plot_co2_emissions("example/emi_inv_2025.nc")
  #plot_co2_emissions("example/emi_inv_2050.nc")
  #co2_reshape("example/emi_inv_2025.nc")

  #plot_co2_emissions_key("Inventories/emi_inv_2025.nc")

  #plot_scaled(1e13,"Inventories/emi_inv_2025.nc")

  #plot_pres("example/emi_inv_2025.nc", sp1=1, scaled_sp=scaled_emissions(4e10, "example/emi_inv_2025.nc"))

  #plot_alt_thresh("example/emi_inv_2025.nc", thresh = 3000, unit="ft", mode="above", sp1=1)  

  #plot_lon("example/emi_inv_2025.nc", sp1=1)

  #plotting co2 and nox by along pres
  #plot_pres("example/emi_inv_2025.nc", sp1=1, sp3=1)
  #plot_co2_threshold("Inventories/emi_inv_2025.nc", threshold_alt=3000, mode="below", unit="ft", top_x=10)
  #plot_co2_nox_ratio("example/emi_inv_2025.nc")


  #co2_emissions_airport("example/emi_inv_2025.nc", threshold_percentile=99.97)
  #co2_emissions_airport_top_x("Inventories/emi_inv_2025.nc", top_n=10)
  

  #SAVING NEW NC FILEs
  #scaled_emissions_to_nc("example/emi_inv_2025.nc", "example/emi_inv_2025_scaled.nc", 5.2e9  )
  #projected_emissions_5years_nc("example/emi_inv_2020.nc", 2020, 01)

  #plot_alt("Inventories/emi_inv_2020.nc",'km',sp1=1, sp2=1,sp3=1)
  
"""region_weights = {
    "North America": 10,
    "South America": 0.8,
    "Europe": 1.2,
    "Africa": 0.9,
    "Asia": 15,
    "Oceania": 0.7,
    "Middle East": 1.3,
    "Atlantic Ocean": 0.5,
    "Far North": 0.6,
    "Far South": 0.4
    }
   growth_factors = [1.0, 5, 10, 20, 200]  # Simulate emissions increasing every 5 years

   lon_edges, lat_edges, emissions_list = simulate_emission_growth(
        file="Inventories/emi_inv_2025.nc",  # Replace with actual filename
        base_emission=1e12,    # Or whatever your base total emission is
        region_weights=region_weights,
        growth_factors=growth_factors)
                                        """

   #animate_emissions_growth(lon_edges, lat_edges, emissions_list, interval=1000)

#plot_weighted_emissions("Inventories/emi_inv_2025.nc", total_emission=1e12, region_weights=region_weights)
