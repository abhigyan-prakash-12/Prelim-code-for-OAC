import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from utility import read_nc, pres_to_alt, unique_val


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

def plot_co2_threshold(file,threshold_percentile, threshold_alt, mode="below", unit="m"):
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

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot = ax.pcolormesh(lon_grid, lat_grid, np.log10(co2_latlon), cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.stock_img()
    ax.gridlines()

    for i, j in zip(*top_coords):
        ax.plot(lon_unique[j], lat_unique[i], 'o', color='cyan', markersize=6, transform=ccrs.PlateCarree(), zorder=10)

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
        ax.plot(airport["lon"], airport["lat"], marker='x', color='red', markersize=15,
                transform=ccrs.PlateCarree(), zorder=11)
        ax.text(airport["lon"] + 1, airport["lat"], airport["code"], fontsize=7, color='white',
                transform=ccrs.PlateCarree(), ha='left', va='center', zorder=12)

    plt.colorbar(plot, ax=ax, orientation='vertical', label='Log10 CO2 Emissions', pad=0.05)
    plt.title(f"Global CO₂ Emissions – Top {top_n} Hotspots + Major Airports")
    plt.tight_layout()
    plt.show()

def apply_region_weights(emission_map, lat_vals, lon_vals, region_weights):
    region_bounds = {
        "North America": [-170, -50, 10, 70],
        "South America": [-90, -30, -60, 15],
        "Europe": [-25, 60, 35, 70],
        "Africa": [-20, 50, -35, 35],
        "Asia": [60, 180, -10, 80],
        "Oceania": [110, 180, -50, -10]
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
        "South America": [-90, -30, -60, 15],
        "Europe": [-25, 60, 35, 70],
        "Africa": [-20, 50, -35, 35],
        "Asia": [60, 180, -10, 80],
        "Oceania": [110, 180, -50, -10]
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
        "South America": [-90, -30, -60, 15],
        "Europe": [-25, 60, 35, 70],
        "Africa": [-20, 50, -35, 35],
        "Asia": [60, 180, -10, 80],
        "Oceania": [110, 180, -50, -10]
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
