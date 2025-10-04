import csv 
import numpy as np
from Generating_nc_files import scaled_emissions_to_nc
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


         


#testt = csv_to_matrix("aviation_emissions_data.csv")
#print(type(testt))
#print(testt.shape)
#csv_to_matrix('aviation_emissions_data.csv')      

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