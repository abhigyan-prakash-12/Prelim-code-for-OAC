import csv 
import numpy as np
from utils.Generating_nc_files import scaled_emissions_to_nc, scaled_emissions_to_nc_with_weights, scaled_emissions_to_nc_complete, scaled_emissions_to_nc_with_weights_cont
import xarray as xr

# Next targets - 
# DONE - make matrix_to_csv so that you can define start and end year as well as increments
# DONE - toml generation for each case. So we don't have to write a new toml each time for different years.
# - figure out how to convert several years of data into nc w/o storage overload
# DONE - weighted nc file generation and compare
# - generalise for all species

def csv_to_matrix_co2(csv_file):
    matrix = []
    with open(csv_file, mode = 'r') as file:
        file_op = csv.reader(file, delimiter=';')
        for i in file_op:
            matrix.append(i)       
    matrix = np.array(matrix)
    for row in matrix[1:]:
         row[1] = float(row[1]) * 1e9
         row[2] = float(row[2]) * 1e9
    return matrix

def matrix_to_co2(csv_file, start_year, end_year, step = 1):
    matrix = csv_to_matrix_co2(csv_file)      
    years = []
    co2 = [] 
    for row in matrix[1:]:
        years.append(row[0])
        co2.append(float(row[1]))

    start_year = str(start_year)
    end_year = str(end_year)    
    start_idx = years.index(start_year)
    end_idx = years.index(end_year)
      
    for i in range(start_idx, end_idx, step):
            nc_name = f"mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc('Inventories/emi_inv_2025.nc', nc_name, co2[i], float(years[i]))

def csv_to_matrix(csv_file):
    matrix = []
    with open(csv_file, mode = 'r') as file:
        file_op = csv.reader(file, delimiter=';')
        for i in file_op:
            matrix.append(i)       
    matrix = np.array(matrix)
    headers = list(matrix[0])
    new_headers = [
        headers[0],  # Years
        headers[1],  # CO2 emissions
        headers[6],  # Distance
        headers[2],  # NOx emissions
        headers[3],  # H2O emissions
        headers[4],  # Soot emissions
        headers[5]   # Sulfur emissions
    ]
    matrix[0] = new_headers
    for row in matrix[1:]:
        co2 = float(row[1]) * 1e9         # CO2 emissions (originally in giga tons --> kg)
        distance = float(row[6])          # Distance in km
        nox = float(row[2]) * 1e9         # NOx emissions (originally in giga tons --> kg)
        h2o = float(row[3]) * 1e9         # H2O emissions (originally in giga tons --> kg)
        soot = float(row[4]) * 1e9        # Soot emissions (originally in giga tons --> kg)
        sulfur = float(row[5]) * 1e9      # Sulfur emissions (originally in giga tons --> kg)

        row[1], row[2], row[3], row[4], row[5], row[6] = co2, distance, nox, h2o, soot, sulfur

    return matrix

def csv_to_ncdf_comp(csv_file, start_year, end_year, step = 1):
    matrix = csv_to_matrix(csv_file)      
    years = []
    co2 = [] 
    dist = []
    nox = []
    h2o = []
    
    for row in matrix[1:]:
        years.append(row[0])
        co2.append(float(row[1]))
        dist.append(float(row[2]))
        nox.append(float(row[3]))
        h2o.append(float(row[4]))
        
    start_year = str(start_year)
    end_year = str(end_year)    
    start_idx = years.index(start_year)
    end_idx = years.index(end_year)
      
    for i in range(start_idx, end_idx, step):
            nc_name = f"mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc_complete('Inventories/emi_inv_2025.nc', nc_name, [co2[i], dist[i], nox[i], h2o[i]], float(years[i]))
              
def csv_to_matrix_weighted_co2(csv_file, start_year, end_year, region_weights,  step = 1):
     matrix = csv_to_matrix(csv_file)
     years = []
     co2 = []
     for row in matrix[1:]:
        years.append(row[0])
        co2.append(float(row[1]))
     start_year = str(start_year)
     end_year = str(end_year)    
     start_idx = years.index(start_year)
     end_idx = years.index(end_year)

     if isinstance(region_weights, dict):
          final_region_weights = region_weights
          for i in range(start_idx, end_idx, step):
            nc_name = f"weighted_mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc_with_weights('Inventories/emi_inv_2025.nc', nc_name, co2[i], float(years[i]), final_region_weights)
     else:
          for idx, weights in enumerate(region_weights):
              for i in range(start_idx, end_idx, step):
                nc_name = f"weighted_mat_generated_nc_{years[i]}.nc"
                scaled_emissions_to_nc_with_weights('Inventories/emi_inv_2025.nc', nc_name, co2[i], float(years[i]), weights)

def csv_to_matrix_weighted_cont(csv_file, start_year, end_year, region_weights,  step = 1):
     matrix = csv_to_matrix(csv_file)
     years = []
     dist = []
     for row in matrix[1:]:
        years.append(row[0])
        dist.append(float(row[2]))
     start_year = str(start_year)
     end_year = str(end_year)    
     start_idx = years.index(start_year)
     end_idx = years.index(end_year)

     if isinstance(region_weights, dict):
          final_region_weights = region_weights
          for i in range(start_idx, end_idx, step):
            nc_name = f"dist_weighted_mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc_with_weights_cont('Inventories/emi_inv_2025.nc', nc_name, dist[i], float(years[i]), final_region_weights)
     else:
          for idx, weights in enumerate(region_weights):
              for i in range(start_idx, end_idx, step):
                nc_name = f"dist_weighted_mat_generated_nc_{years[i]}.nc"
                scaled_emissions_to_nc_with_weights_cont('Inventories/emi_inv_2025.nc', nc_name, dist[i], float(years[i]), weights)

def time_norm_ncdf(file, nc_name):
    matrix = csv_to_matrix("aviation_emissions_data.csv")[1:]  
    years = np.array([row[0] for row in matrix], dtype=float)
    co2 = np.array([float(row[1]) for row in matrix])
    dist = np.array([float(row[2]) for row in matrix])
    nox = np.array([float(row[3]) for row in matrix])
    h2o = np.array([float(row[4]) for row in matrix])



    fuel = co2 / 3.15 * 1e-9
    dis_per_fuel = dist / (fuel *1e9)
    EI_NOx = (nox * 27.79543563) / (fuel *1e9) * 0.03597713
    EI_H2O = h2o / (fuel *1e9)
    EI_CO2 = np.array([3.15] * len(years))
   

    ds = xr.Dataset(
        data_vars={
            "fuel":        (("time",), np.asarray(fuel, dtype=np.float32)),
            "EI_CO2":      (("time",), np.asarray(EI_CO2, dtype=np.float32)),
            "EI_NOx":      (("time",), np.asarray(EI_NOx, dtype=np.float32)),
            "EI_H2O":      (("time",), np.asarray(EI_H2O, dtype=np.float32)),
            "dis_per_fuel":(("time",), np.asarray(dis_per_fuel, dtype=np.float32))
            
        },
        coords={
            "time": ("time", np.asarray(years, dtype=np.int32))
        },
        attrs={
            "Title":       "Time normalization ",
            "Convention":  "CF-XXX",
            "Type":        "norm",
            "Author":      "Abhigyan Prakash based on OAC example",
        }
    )
    
    
    ds["fuel"].attrs.update(units="Tg yr-1", long_name="fuel mass")
    ds["EI_CO2"].attrs.update(units="", long_name="CO2 emission index")
    ds["EI_NOx"].attrs.update(units="", long_name="NOx emission index")
    ds["EI_H2O"].attrs.update(units="", long_name="H2O emission index")
    ds["dis_per_fuel"].attrs.update(units="km kg-1", long_name="distance per fuel")
    ds["time"].attrs.update(long_name="year")

    encoding = {var: {"dtype": "float32", "zlib": True, "complevel": 4} for var in ds.data_vars}
    encoding["time"] = {"dtype": "int32"}

    outpath = f"inputs/{nc_name}"
    ds.to_netcdf(outpath, encoding=encoding)

    print("Saved new NetCDF to:", outpath)
    print("Dataset summary:")
    print(ds)



"""test_weights = {
        "North America": 1.2,
        "Atlantic region": 1.0,
        "South America": 0.8,
        "Pacific": 0.9,
        "Far North": 1.1,
        "Europe": 13,
        "Africa and Middle East": 0.7,
        "Asia": 14,
        "Oceania": 0.6
    }"""


    
     
     