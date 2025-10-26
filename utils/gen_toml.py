import toml
import os
# use an example toml file, keep most stuff fixed but in the function allow certain parameters to change
# copy and edit template with the function to generate new toml file to run
# currently the import_csv codes and toml_gen codes will be different but can combine them later

def generate_toml(start_year, end_year, step, output_file, 
                       inv_species=None, out_species=None, weighted=None):

    os.makedirs("tomls", exist_ok=True)

    h = end_year - start_year + 1
    if inv_species is None:
        inv_species = ["CO2","H2O", "NOx", "distance"]
    else:
        inv_species = inv_species    
    if out_species is None:
        out_species = ["CO2","H2O", "cont"]
    else:
        out_species = out_species
    if weighted is None:
        inventory_files = [f"mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
    if weighted == "distance_weighted":
        inventory_files = [f"dist_weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
    if weighted  == "weighted":
        inventory_files = [f"weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]    

    
    toml_content = f'''

    # Species considered
    [species]
    # Species defined in emission inventories
    # possible values: "CO2", "H2O", "NOx", "distance"
    inv = {inv_species}
    # Assumed NOx species in emission inventory
    # possible values: "NO", "NO2"
    nox = "NO"
    # Output / response species
    # possible values: "CO2", "H2O", "O3", "CH4", "PMO", "cont"
    out = {out_species}

    # Emission inventories                                                    
    [inventories]
    dir = "inputs/"
    files = [ {", ".join([f'"{file}"' for file in inventory_files])}
    ]
    # base emission inventories, only considered if rel_to_base = true
    rel_to_base = false
    base.dir = "input/"
    base.files = [
        "rnd_inv_2020.nc",
        "rnd_inv_2030.nc",
        "rnd_inv_2040.nc",
        "rnd_inv_2050.nc",
    ]

    # Output options
    [output]
    # Full simulation run = true, climate metrics only = false
    full_run = true
    dir = "results/"
    name = "gen_{start_year}s"
    overwrite = true
    # Computation of 2D concentration responses is not yet supported.
    # possible values: false 
    concentrations = false

    # Time settings
    [time]
    dir = "input/"
    # Time range in years: t_start, t_end, step, (t_end not included)
    range = [{start_year}, {end_year}, {step}]
    # Time evolution of emissions
    # either type "scaling" or type "norm"
    #file = "time_scaling_example.nc"
    #file = "time_norm_example.nc"

    # Global background concentrations
    [background]
    dir = "oac/repository/"
    CO2.file = "co2_bg.nc"
    CO2.scenario = "SSP2-4.5"
    #CO2.scenario = "SSP1-1.9"
    #CO2.scenario = "SSP4-6.0"
    #CO2.scenario = "SSP3-7.0"
    CH4.file = "ch4_bg.nc"
    CH4.scenario = "SSP2-4.5"
    N2O.file = "n2o_bg.nc"
    N2O.scenario = "SSP2-4.5"

    # Response options
    [responses]
    dir = "oac/repository/"
    CO2.response_grid = "0D"
    CO2.conc.method = "Sausen&Schumann"
    # RF method based on Etminan et al. 2016 is used by default.
    #CO2.rf.method = "Etminan_2016"

    H2O.response_grid = "2D"
    H2O.rf.file = "resp_RF.nc"    # AirClim response surface

    O3.response_grid = "2D"
    O3.rf.file = "resp_RF_O3.nc"  # tagging
    #O3.rf.file = "resp_RF.nc"    # AirClim response surface, requires adjustment of CORR_RF_O3 !

    CH4.response_grid = "2D"
    CH4.tau.file = "resp_ch4.nc"  # tagging
    CH4.rf.method = "Etminan_2016"

    cont.response_grid = "cont"
    cont.resp.file = "resp_cont_lf.nc"

    # Temperature options
    [temperature]
    # valid methods: "Boucher&Reddy"
    method = "Boucher&Reddy"
    # Climate sensitivity parameter, Ponater et al. 2006, Table 1
    # https://doi.org/10.1016/j.atmosenv.2006.06.036
    CO2.lambda = 0.73
    # Efficacies, Ponater et al. 2006, Table 1
    H2O.efficacy = 1.14
    O3.efficacy = 1.37
    PMO.efficacy = 1.37
    CH4.efficacy = 1.14
    cont.efficacy = 0.59

    # Climate metrics options
    [metrics]
    # iterate over elements in lists types t_0 and H
    types = ["AGWP", "ATR", "AGTP"] # valid climate metrics: AGTP, AGWP, ATR
    H = [{h}]                        # Time horizon, t_final = t_0 + H - 1
    t_0 = [{start_year}]                    # Start time for metrics calculation

    # aircraft defined in inventory
    # following identifiers are NOT allowed: "TOTAL"
    # "DEFAULT" is used if "ac" coordinate not defined in emission inventories
    # G_250, eff_fac and PMrel must be defined for each aircraft if contrails are
    # to be calculated.
    [aircraft]
    types = ["DEFAULT"]
    DEFAULT.G_250 = 1.70   # Schmidt-Appleman mixing line slope at 250 hPa
    DEFAULT.eff_fac = 1.0  # efficiency factor compared to 0.333
    DEFAULT.PMrel = 1.0    # relative PM emissions compared to 1e15
    '''


    
    
    file_path = os.path.join("tomls", output_file)    
    with open(file_path, 'w') as f:
        f.write(toml_content)
    print(f"TOML file written to {output_file}")


    




