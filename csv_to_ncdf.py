import csv 
import numpy as np
from Generating_nc_files import scaled_emissions_to_nc
import xarray as xr
import openairclim as oac

#QUESTION TO ANSWER: given the several years of data simulated, if we create an nc file of each to map the aggregated co2,
#                    then storage becomes an issue, do we approach it differently?

def csv_to_matrix(csv_file):
    matrix = []
    with open(csv_file, mode = 'r') as file:
        file_op = csv.reader(file, delimiter=';')
        for i in file_op:
            matrix.append(i)       
    return np.array(matrix)

#csv_to_matrix('aviation_emissions_data.csv')    

def matrix_to_co2(csv_file):
    matrix = csv_to_matrix(csv_file)      
    years = []
    co2 = [] 
    for row in matrix[1:]:
        years.append(row[0])
        co2.append(float(row[1]))
    start_year = years[0]
    end_year = years[-1]
    #base_nc = xr.open_dataset('Inventories/emi_inv_2025.nc')
    for i in range(len(years)):
        if i < 2:
            nc_name = f"mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc('Inventories/emi_inv_2025.nc', nc_name, co2[i], float(years[i]))






matrix_to_co2('aviation_emissions_data.csv')
