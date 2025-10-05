import csv 
import numpy as np
from Generating_nc_files import scaled_emissions_to_nc, scaled_emissions_to_nc_with_weights
import xarray as xr

# Next targets - 
# DONE - make matrix_to_csv so that you can define start and end year as well as increments
# DONE - toml generation for each case. So we don't have to write a new toml each time for different years.
# - figure out how to convert several years of data into nc w/o storage overload
# - weighted nc file generation

def csv_to_matrix(csv_file):
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


#rename it csv_to_matrix
def matrix_to_co2(csv_file, start_year, end_year, step = 1):
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
      
    for i in range(start_idx, end_idx, step):
            nc_name = f"mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc('Inventories/emi_inv_2025.nc', nc_name, co2[i], float(years[i]))
              
#matrix_to_co2('aviation_emissions_data.csv', 2000,2005)

def csv_to_matrix_weighted(csv_file, start_year, end_year, region_weights,  step = 1):
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

test_weights = {
        "North America": 1.2,
        "Atlantic region": 1.0,
        "South America": 0.8,
        "Pacific": 0.9,
        "Far North": 1.1,
        "Europe": 13,
        "Africa and Middle East": 0.7,
        "Asia": 14,
        "Oceania": 0.6
    }

#csv_to_matrix_weighted('aviation_emissions_data.csv', 2020, 2025, test_weights,  step = 1)         
          
    
     
     