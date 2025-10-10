import xarray as xr

scaling = xr.open_dataset('oac/repository/time_scaling_example.nc')
#print(scaling['scaling'].values)

norm = xr.open_dataset('oac/repository/time_norm_example.nc')
print(norm) 