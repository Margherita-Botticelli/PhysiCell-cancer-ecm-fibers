from pyMCDS_ECM import *
import os
import re
import sys
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from spheroid_area_function import *
from delaunay_function import *

def reduce_mem_usage(df, verbose=True):
    """
    Reduce the memory usage of a DataFrame by downcasting numerical columns to more efficient types.

    Parameters:
    - df: pandas DataFrame to optimize.
    - verbose: Boolean flag to print memory reduction details (default is True).

    Returns:
    - df: DataFrame with optimized memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2  # Memory usage in MB
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            #### Optimize integer columns
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            # else:  # For float columns
            #     if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            #         df[col] = df[col].astype(np.float16)
            #     elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            #         df[col] = df[col].astype(np.float32)
            #     else:
            #         df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)', flush=True)
    
    return df

def snapshot_data(orientation,simulation, seed, snapshot, data_folder):
    """
    Process a single snapshot to extract cell data and simulation parameters.

    Parameters:
    - orientation: Orientation identifier for the simulation.
    - simulation: Simulation identifier.
    - seed: Random seed used in the simulation.
    - snapshot: Filename of the snapshot to process.
    - data_folder: Directory containing the snapshot data.

    Returns:
    - cell_df: DataFrame with processed cell data and simulation parameters.
    """
    #### Load cell and microenvironment data
    mcds = pyMCDS(snapshot + '.xml', data_folder)
    t = mcds.get_time()  # Get current time

    #### Load cell data
    cell_df = mcds.get_cell_df()
    cell_var = mcds.get_cell_variables()

    #### Define columns to remove from cell dataframe
    cell_var_remove = ['total_volume', 'cell_type', 'cycle_model', 'current_phase', 'elapsed_time_in_phase', 
                       'nuclear_volume', 'cytoplasmic_volume', 'fluid_fraction', 'calcified_fraction', 'orientation_x', 
                       'orientation_y', 'orientation_z', 'polarity', 'velocity_x', 'velocity_y', 'velocity_z', 
                       'number_of_nuclei', 'damage', 'total_attack_time', 'contact_with_basement_membrane', 'dead', 
                       'current_death_model', 'death_rates_x', 'death_rates_y', 'cytoplasmic_biomass_change_rate', 
                       'nuclear_biomass_change_rate', 'fluid_change_rate', 'calcification_rate', 'target_solid_cytoplasmic', 
                       'target_solid_nuclear', 'target_fluid_fraction', 'nuclear_radius', 'surface_area', 
                       'cell_BM_adhesion_strength', 'cell_BM_repulsion_strength', 'cell_adhesion_affinities', 
                       'attachment_elastic_constant', 'attachment_rate', 'detachment_rate', 'is_motile', 
                       'migration_bias', 'chemotaxis_index', 'chemotaxis_direction', 'chemotactic_sensitivities', 
                       'secretion_rates', 'uptake_rates', 'saturation_densities', 'net_export_rates', 
                       'internalized_total_substrates', 'fraction_released_at_death', 'fraction_transferred_when_ingested', 
                       'dead_phagocytosis_rate', 'live_phagocytosis_rates', 'attack_rates', 'damage_rate', 
                       'fusion_rates', 'transformation_rates']

    cell_df = cell_df.drop(cell_var_remove, axis=1)  # Drop unnecessary columns

    #### Rearrange columns and add simulation parameters
    dictionary = ['simulation','orientation', 'seed', 't'] + list(cell_df.columns)
    cell_df['t'] = t
    cell_df['simulation'] = simulation
    cell_df['seed'] = seed
    cell_df['orientation'] = orientation
    cell_df = cell_df[dictionary]

    #### Add simulation parameters from XML settings file
    tree = ET.parse(data_folder + 'PhysiCell_settings.xml')
    root = tree.getroot()

    #### Extract relevant parameters
    user_parameters = root.find('user_parameters') # type: ignore
    cell_definitions = root.find('cell_definitions') # type: ignore
    cell_definition = cell_definitions.find('cell_definition') # type: ignore
    phenotype = cell_definition.find('phenotype') # type: ignore
    cycle = phenotype.find('cycle') # type: ignore
    motility = phenotype.find('motility') # type: ignore
    mechanics = phenotype.find('mechanics') # type: ignore
    phase_transition_rates = cycle.find('phase_transition_rates') # type: ignore

    # sigma = float(user_parameters.find('sigma').text)  # type: ignore
    # delta = float(user_parameters.find('delta').text)  # type: ignore
    # cell_df['sigma'] = sigma
    # cell_df['delta'] = delta

    initial_ecm_density = user_parameters.find('initial_ecm_density')  # type: ignore
    initial_ecm_density = float(initial_ecm_density.text)  # type: ignore
    cell_df['initial_ecm_density'] = initial_ecm_density  # type: ignore

    initial_anisotropy = user_parameters.find('initial_anisotropy')  # type: ignore
    initial_anisotropy = float(initial_anisotropy.text)  # type: ignore
    cell_df['initial_anisotropy'] = initial_anisotropy  # type: ignore

    cell_cell_adhesion_strength = mechanics.find('cell_cell_adhesion_strength') # type: ignore
    cell_adh = float(cell_cell_adhesion_strength.text)  # type: ignore
    cell_df['cell_adh'] = cell_adh

    cell_cell_repulsion_strength = mechanics.find('cell_cell_repulsion_strength') # type: ignore
    cell_rep = float(cell_cell_repulsion_strength.text)  # type: ignore
    cell_df['cell_rep'] = cell_rep

    prolif = float(phase_transition_rates.find('rate').text)  # type: ignore
    max_mot_speed = float(motility.find('speed').text)  # type: ignore
    cell_df['prolif'] = prolif
    cell_df['max_mot_speed'] = max_mot_speed

    #### Compute spheroid area and Delaunay distance
    spheroid_area = spheroid_area_function(cell_df)
    cell_df['spheroid_area'] = spheroid_area

    delaunay_distance = delaunay_distance_function(cell_df, figure=True)
    cell_df['delaunay_distance'] = delaunay_distance

    #### Set index based on snapshot name
    index = int(snapshot.replace("output", ""))
    cell_df = cell_df.set_index(pd.Series([index] * len(cell_df.index)))

    return cell_df


def simulation_data(data_folder_dir, simulation, orientation, seed):
    """
    Process all snapshots for a given simulation and save the resulting DataFrame.

    Parameters:
    - data_folder_dir: Directory containing simulation output data.
    - simulation: Simulation identifier.
    - orientation: orientation identifier for the simulation.
    - seed: Random seed used in the simulation.

    Returns:
    - None
    """
    print(f"\n#### {simulation=}, {orientation=}, {seed=} ####\n", flush=True)

    #### Check if DataFrame already exists 
    df_path = data_folder_dir + f'output_{orientation}_{simulation}_{seed}/dataframe_{orientation}_{simulation}_{seed}.pkl'

    #### Prepare data folder and process files
    data_folder = data_folder_dir + f'output_{orientation}_{simulation}_{seed}/'
    files = os.listdir(data_folder)
    files.sort()

    df_list = []
    for file in files:
        if not re.search('_ECM\.mat', file):
            continue
        snapshot = file.split('_')[0]
        # print(f'Processing snapshot {snapshot}', flush=True)
        df_new = snapshot_data(orientation,simulation, seed, snapshot, data_folder)

        #### Reduce memory usage and append to list
        df_new = reduce_mem_usage(df_new, verbose=False)
        df_list.append(df_new)

    #### Combine all DataFrames and save
    df = pd.concat(df_list, copy=False, axis=0)
    df.to_pickle(df_path)

    return

