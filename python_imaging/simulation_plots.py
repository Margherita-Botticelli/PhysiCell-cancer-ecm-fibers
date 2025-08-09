from pyMCDS_ECM import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed # type: ignore
import pandas as pd
from simulation_data import *
from box_plots import *
from heatmaps import *
from spheroid_area_function import spheroid_area_function
from delaunay_function import *
from cell_plus_environment_movie_maker import create_plot
from skimage import io
from PIL import Image
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from joblib import Parallel, delayed


#### Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    
    #### Set figure resolution for high-quality plots
    mpl.rcParams['figure.dpi'] = 300
    
    #### Set plotting style for consistency
    plt.style.use('seaborn-v0_8-colorblind')

    #### Specify the project to work on
    proj = 'ecm_fibers'  # Options: 'tests', 'ecm_fibers', 'ecm_density'
    
    #### Define folders for saving results and loading data
    save_folder = f'../results/{proj}/'
    data_folder = f'../data/{proj}/'

    #### Define simulations to run
    # simulations = list(range(0,36)) #### split anisotropy 0.5
    # simulations = list(range(36,72)) #### split anisotropy 0.2
    # simulations = list(range(72,108)) #### split anisotropy 0.8
    simulations = list(range(0,108)) #### split anisotropy 0.5, 0.2, 0.8
    # simulations = list(range(108,114)) #### spheroid (tangential, random and radial) with adh 4.0
    # simulations = list(range(114,120)) #### spheroid (tangential, random and radial) with adh 0.4
    # simulations = list(range(120,156)) #### random anisotropy 0.0

    #### Flag to determine if existing data should be replaced
    replace = False # True # False # Options: True if you want to replace existing data, False if you want to use existing data

    #### List of fibre orientations to test
    orientations = ['split'] # ['radial', 'random','tangential'] # ['random'] # ['random'] # Options:'random', 'radial', 'tangential'

    #### Number of random seeds for simulations
    n_seeds = 10
    seeds = list(range(0, n_seeds))

    #### Flags for different types of plots
    title = False # Title on plots
    box_plots = True # Box plots
    heatmaps_speed_vs_degr = False  
    heatmaps_speed_vs_initial_ecm_density = False
    heatmaps_chemo_vs_ecm_sensitivity = False
    heatmaps_reorientation_vs_orientation = False
    heatmap_fiber_orientation = False
    heatmap_time_points = [48*60]#[24*60,48*60]  # Example: [48*60], [96*60]
    time_point_images = False # 'all' # False # True # True # False # Options: 'all' if you want all time points
    times = [48*60]#[24*60,48*60] # [0,48*60] # [96*60] # Time points to consider for images, Options: [24*60, 48*60, 72*60, 96*60]
    video = False 

    #### Define a name for the simulation set based on its indices
    simulation_name = '_'.join(str(s) for s in simulations)
    
    ########## PROCESSING SIMULATION DATA ###########
    orientations_list = np.repeat([orientations] * len(simulations), len(seeds))
    simulations_list = np.repeat([simulations], len(seeds)* len(orientations))
    seeds_list = seeds * len(simulations) * len(orientations)
    data_folder_list = [data_folder] * len(seeds) * len(simulations) * len(orientations)

    if replace == True:
        Parallel(n_jobs=-1)(delayed(simulation_data)(data_folder, simulation, orientation, seed) for data_folder, simulation, orientation, seed in zip(data_folder_list, simulations_list, orientations_list, seeds_list))

        #### Notify when parallel processing is complete
        print('Parallel end\n', flush=True)

    #### Initialize an empty DataFrame and list to store simulation data
    df = pd.DataFrame()
    df_list = []

    #### Combine data from different simulations into one DataFrame
    for sim in simulations:
        for orientation in orientations:
            for seed in seeds:
                df_new = pd.read_pickle(data_folder + f'output_{orientation}_{sim}_{seed}/dataframe_{orientation}_{sim}_{seed}.pkl')
                df_list.append(df_new)

    #### Concatenate all dataframes into a single dataframe
    df = pd.concat(df_list, copy=False, axis=0)

    print('Dataframe ready!\n', flush=True)

    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    ########## BOX PLOTS ###########
    if box_plots:

        for cell_speed in np.unique(df['max_mot_speed']).astype(float):
            for initial_anisotropy in np.unique(df['initial_anisotropy']).astype(float):
            
                for time_point in heatmap_time_points:
                    data = df[(df['t'] == time_point) & (df['max_mot_speed'] == cell_speed) & (df['initial_anisotropy'] == initial_anisotropy)]
                    # plot_cell_distances(data, save_folder, title=title)

                    initial_anisotropy_heatmap = np.unique(df['initial_anisotropy']).astype(float)
                    for initial_anisotropy in initial_anisotropy_heatmap:
                        data_initial_anisotropy = df[(df['t'] == time_point) & (df['initial_anisotropy'] == initial_anisotropy)]
                        simulation_numbers = np.unique(data_initial_anisotropy['simulation']).astype(int)
                        simulation_name = simulation_numbers[0]
                        plot_cell_distances_split(data_initial_anisotropy,simulation_name, save_folder, title=title)
                    
                    data_anisotropy = df[
                        (df['t'] == time_point) &
                        (df['chemotaxis_bias'] == 0.2) &
                        (df['ecm_sensitivity'].isin([0.6, 0.8, 1.0]))
                    ]
                    simulation_numbers = np.unique(data_anisotropy['simulation']).astype(int)
                    simulation_name = simulation_numbers[0]
                    plot_cell_distances_split_anisotropy(data_anisotropy,simulation_name, save_folder, title=title)
                plt.close('all')

        # for sim in simulations:
        #     for time_point in times:
        #         print(f'Box plots for simulation {sim} at time point {time_point}', flush=True)
        #         seed = 0
        #         data = df[(df['simulation'] == sim) & (df['t'] == time_point) & (df['seed'] == seed)]
        #         plot_fibre_orientation(data, data_folder, save_folder, title=title)
        #         plt.close('all')

        print('Box plots done!', flush=True)


    ######## HEATMAP PLOTS #########
    for time_point in heatmap_time_points:

        #### Check if heatmaps for chemo vs ecm sensitivity should be generated
        if heatmaps_chemo_vs_ecm_sensitivity:

            for orientation in orientations:
        
                #### Further filter data by the current initial_ecm_density
                data = df[(df['ID'] == 0) & ((df['t'] == 0) | (df['t'] == time_point)) & (df['orientation'] == orientation)]
                
                #### Get first simulation ID to name the plot
                simulations_heatmap = np.unique(data['simulation']).astype(int)
                simulation_name_plot = simulations_heatmap[0]
                
                #### Generate the heatmap for spheroid area growth
                plots_chemo_vs_ecm_sensitivity_spheroid_area_growth(data, simulation_name_plot, save_folder, title=title)
                print('plots_chemo_vs_ecm_sensitivity_spheroid_area_growth done!', flush=True)

                #### Further filter data
                data = df[(df['ID'] == 0) & ((df['t'] == time_point)) & (df['orientation'] == orientation)]
                
                #### Get first simulation ID to name the plot
                simulations_heatmap = np.unique(data['simulation']).astype(int)
                simulation_name_plot = simulations_heatmap[0]

                #### Generate the heatmap for Delaunay mean distance
                plots_chemo_vs_ecm_sensitivity_delaunay(data, simulation_name_plot, save_folder, title=title)
                print('plots_chemo_vs_ecm_sensitivity_delaunay done!', flush=True)

                #### Further filter data
                data = df[((df['t'] == time_point)) & (df['orientation'] == orientation)]
                
                #### Get first simulation ID to name the plot
                simulations_heatmap = np.unique(data['simulation']).astype(int)
                simulation_name_plot = simulations_heatmap[0]

                #### Generate the heatmap for invasion
                plots_chemo_vs_ecm_sensitivity_invasion(data, simulation_name_plot, save_folder, title=title)
                print('plots_chemo_vs_ecm_sensitivity_invasion done!', flush=True)
       
        #### Check if heatmaps for reorientation vs orientation should be generated
        if heatmaps_reorientation_vs_orientation:

            chemotaxis_heatmap = np.unique(df['chemotaxis_bias']).astype(float)
            initial_anisotropy_heatmap = np.unique(df['initial_anisotropy']).astype(float)
            ecm_sensitivity_heatmap = np.unique(df['ecm_sensitivity']).astype(float)
            max_mot_speed_heatmap = np.unique(df['max_mot_speed']).astype(float)
            displacement_rate_heatmap = np.unique(df['ecm_displacement_rate']).astype(float)
            
            for ecm_displacement_rate in displacement_rate_heatmap:
                for max_mot_speed in max_mot_speed_heatmap:
                    for chemotaxis in chemotaxis_heatmap:
                        for initial_anisotropy in initial_anisotropy_heatmap:
                            for ecm_sensitivity in ecm_sensitivity_heatmap:

                                #### Further filter data by the current initial_ecm_density
                                data = df[(df['ID'] == 0) & ((df['t'] == 0) | (df['t'] == time_point)) & (df['chemotaxis_bias'] == chemotaxis) & (df['initial_anisotropy'] == initial_anisotropy) & (df['ecm_sensitivity'] == ecm_sensitivity) & (df['ecm_displacement_rate'] == ecm_displacement_rate) & (df['max_mot_speed'] == max_mot_speed)]
                                
                                #### Get first simulation ID to name the plot
                                simulations_heatmap = np.unique(data['simulation']).astype(int)
                                simulation_name_plot = simulations_heatmap[0]
                                
                                #### Generate the heatmap for spheroid area growth
                                plots_reorientation_vs_orientation_spheroid_area_growth(data, simulation_name_plot, save_folder, title=title)
                                print('plots_reorientation_vs_orientation_spheroid_area_growth done!', flush=True)

                                #### Further filter data
                                data = df[(df['ID'] == 0) & ((df['t'] == time_point)) & (df['chemotaxis_bias'] == chemotaxis) & (df['initial_anisotropy'] == initial_anisotropy) & (df['ecm_sensitivity'] == ecm_sensitivity)& (df['ecm_displacement_rate'] == ecm_displacement_rate) & (df['max_mot_speed'] == max_mot_speed)]
                                
                                #### Get first simulation ID to name the plot
                                simulations_heatmap = np.unique(data['simulation']).astype(int)
                                simulation_name_plot = simulations_heatmap[0]

                                #### Generate the heatmap for Delaunay mean distance
                                plots_reorientation_vs_orientation_delaunay(data, simulation_name_plot, save_folder, title=title)
                                print('plots_reorientation_vs_orientation_delaunay done!', flush=True)

                                #### Further filter data
                                data = df[((df['t'] == time_point)) & (df['chemotaxis_bias'] == chemotaxis) & (df['initial_anisotropy'] == initial_anisotropy) & (df['ecm_sensitivity'] == ecm_sensitivity)& (df['ecm_displacement_rate'] == ecm_displacement_rate) & (df['max_mot_speed'] == max_mot_speed)]
                                
                                #### Get first simulation ID to name the plot
                                simulations_heatmap = np.unique(data['simulation']).astype(int)
                                simulation_name_plot = simulations_heatmap[0]

                                #### Generate the heatmap for invasion
                                plots_reorientation_vs_orientation_invasion(data, simulation_name_plot, save_folder, title=title)
                                print('plots_reorientation_vs_orientation_invasion done!', flush=True)

        if heatmap_fiber_orientation:
            data = df[(df['t'] == time_point)]

            #### Get first simulation ID to name the plot
            simulations_heatmap = np.unique(data['simulation']).astype(int)
            simulation_name_plot = simulations_heatmap[0]
            
            #### Generate the heatmap for fiber orientation
            fiber_orientation_heatmap(data, data_folder, simulation_name_plot, save_folder, title=title)
            print('plot_fiber_orientation_heatmap done!', flush=True)


    #### Close all plot figures to free up memory
    plt.close('all')


    ######### TIME POINT IMAGE ##############
    def generate_image(sim, seed, orientation, t, df, data_folder, save_folder, title):
        #### Filter data
        # seed = 0  # You had this reset to 0 inside the loop — but it's passed in now
        data = df[(df['simulation'] == sim) & (df['orientation'] == orientation) & (df['seed'] == seed) & (df['t'] == t)]

        # print(data, flush=True)
        if data.empty:
            return  # Skip if no data found

        #### Get time point to find snapshot
        time_step = data[data['ID'] == 0].index.values.astype(int)[0]
        snapshot = 'output' + '{:08d}'.format(int(time_step))
        data_folder_sim = data_folder + f'output_{orientation}_{sim}_{seed}/'
        save_name = save_folder + f'images/full_image_{orientation}_{sim}_{seed}_t{int(t):04}.png'

        print(f'{orientation=}, {sim=}, {t=}', flush=True)
        #### Generate images
        create_plot(data, snapshot, data_folder_sim, save_name, output_plot=True, title=title) 
        plt.close('all')


    #### Check if time point images should be generated
    if time_point_images:
        tasks = []
        for sim in simulations:
            # for seed in seeds:
                seed = 0
                for orientation in orientations:
                    if time_point_images:
                        if time_point_images == 'all':
                            times = np.unique(df[(df['seed'] == seed) & (df['ID'] == 0)]['t']).astype(int)
                        for t in times:
                            tasks.append((sim, seed, orientation, t))

        #### Run image generation in parallel
        Parallel(n_jobs=-1)(delayed(generate_image)(sim, seed, orientation, t, df, data_folder,save_folder, title)for sim, seed, orientation, t in tasks)


    #### Check if video should be generated
    if video:
        for sim in simulations:
            seed = 0
            for orientation in orientations:
                video_name = save_folder + f'animations/video_{orientation}_{sim}_{seed}.mp4'

                #### Find generated images
                images = save_folder + f'images/full_image_{orientation}_{sim}_{seed}_t*.png'

                #### Generate video
                os.system(f'ffmpeg -y -framerate 10 -pattern_type glob -i \'{images}\' -c:v libx264 -pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" {video_name}')

                print('Video ready!', flush=True)

