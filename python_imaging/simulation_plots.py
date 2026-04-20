from pyMCDS_ECM import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import warnings
from joblib import Parallel, delayed  # type: ignore
import os

# Import project-specific modules
from simulation_data import *
from box_plots import *
from heatmaps import *
from spheroid_area_function import spheroid_area_function
from delaunay_function import *
from cell_plus_environment_movie_maker import create_plot
from grid_images import grid_images

# ---------------------------
# Suppress future warnings
# ---------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == '__main__':

    # ---------------------------
    # Set figure resolution for high-quality plots
    # ---------------------------
    mpl.rcParams['figure.dpi'] = 300
    plt.style.use('seaborn-v0_8-colorblind')

    # ---------------------------
    # Specify the project to work on
    # ---------------------------
    proj = 'ecm_fibers'  # Options: 'tests', 'ecm_fibers', 'ecm_density'
    save_folder = f'../results/{proj}/'
    data_folder = f'../data/{proj}/'

    # ---------------------------
    # Simulations and orientations
    # ---------------------------
    
    # simulations = list(range(156,181)) # displacement vs degradation
    # orientations = ['random']
    # row_param = 'ecm_displacement_rate'
    # col_param = 'ecm_density_rate' 

    simulations = list(range(181,186)) # orientation vs reorientation rate
    orientations = ['random', 'radial', 'tangential']
    row_param = 'orientation' 
    col_param = 'fiber_reorientation_rate' 
    
    # simulations = list(range(186,222)) # [189] # # ecm sens vs chemo bias random 
    # orientations = ['random']
    # row_param = 'ecm_sensitivity'
    # col_param = 'chemotaxis_bias'

    # simulations = list(range(222,258)) # [247] # ecm sens vs chemo bias split
    # orientations = ['split']
    # row_param = 'ecm_sensitivity' 
    # col_param = 'chemotaxis_bias'

    # simulations = list(range(258,267)) # list(range(267,285)) # ecm sens vs init anisotropy split
    # orientations = ['split']
    # row_param = 'ecm_sensitivity' 
    # col_param = 'initial_anisotropy'

    #############################

    # sweep_params = [
    # 'orientation', 'initial_ecm_density', 'initial_anisotropy',
    # 'ecm_displacement_rate', 'ecm_density_rate', 'fiber_reorientation_rate', 'fiber_realignment_rate',
    # 'chemotaxis_bias', 'ecm_sensitivity', 'max_mot_speed', 
    # ]

    sweep_params = ['orientation', 'ecm_displacement_rate', 'ecm_density_rate', 'fiber_reorientation_rate', 'fiber_realignment_rate','chemotaxis_bias', 'ecm_sensitivity', 'max_mot_speed', 'initial_anisotropy', 'cell_adh']

    # sweep_params = ['chemotaxis_bias', 'ecm_sensitivity', 'initial_anisotropy']

    # Determine which parameters are "other parameters" (everything except row and col)
    other_params = [p for p in sweep_params if p not in [row_param, col_param]]

    n_seeds = 10
    seeds = list(range(n_seeds))

    # ---------------------------
    # Plot flags
    # ---------------------------
    title = False
    box_plots = False
    heatmap_plots = True
    heatmap_time_points = [96*60]
    time_point_images = False
    images_grid = False
    times = range(0,96*60+1, 60) # [96*60] # 
    video = False

    replace = False  # True: replace existing data, False: use existing data

    # ---------------------------
    # Simulation name
    # ---------------------------
    simulation_name = '_'.join(str(s) for s in simulations)

    # ---------------------------
    # Prepare simulation data for parallel processing
    # ---------------------------
    orientations_list = np.repeat([orientations] * len(simulations), len(seeds))
    simulations_list = np.repeat([simulations], len(seeds) * len(orientations))
    seeds_list = seeds * len(simulations) * len(orientations)
    data_folder_list = [data_folder] * len(seeds) * len(simulations) * len(orientations)

    if replace:
        Parallel(n_jobs=-1)(
            delayed(simulation_data)(data_folder, simulation, orientation, seed)
            for data_folder, simulation, orientation, seed
            in zip(data_folder_list, simulations_list, orientations_list, seeds_list)
        )
        print('Parallel end\n', flush=True)

    # ---------------------------
    # Load all simulation data into a single DataFrame
    # ---------------------------
    df_list = []
    for sim in simulations:
        for orientation in orientations:
            for seed in seeds:
                path = data_folder + f'output_{orientation}_{sim}_{seed}/dataframe_{orientation}_{sim}_{seed}.pkl'
                df_list.append(pd.read_pickle(path))
    df = pd.concat(df_list, copy=False, axis=0)
    print('Dataframe ready!\n', flush=True)

    # Group by all "other parameters" to make a heatmap for each combination
    grouped = df.groupby(other_params) if other_params else [((), df)]

    for group_vals, df_group in grouped if other_params else grouped:
                # Prepare a title with the current combination of other parameters
        if other_params:
            if isinstance(group_vals, tuple):
                title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, group_vals))
            else:
                title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, [group_vals]))
        else:
            title_str = None

    print(title_str)

    # ---------------------------
    # Set seaborn plotting style
    # ---------------------------
    sns.set_context("paper")
    sns.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

   # ---------------------------
    # BOX PLOTS
    # ---------------------------
    if box_plots:

        # Group exactly like heatmaps/images
        grouped = df.groupby(other_params) if other_params else [((), df)]

        for group_vals, df_group in grouped if other_params else grouped:

            if df_group.empty:
                continue

            # Build title for this parameter combination
            if other_params and title:
                if isinstance(group_vals, tuple):
                    title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, group_vals))
                else:
                    title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, [group_vals]))
            else:
                title_str = None

            print(title_str)

            for time_point in heatmap_time_points:

                data_time = df_group[df_group['t'] == time_point]
                if data_time.empty:
                    continue

                sim_name = int(np.unique(data_time['simulation'])[0])

                # ---------------------------
                # Split by initial anisotropy
                # ---------------------------
                for anisotropy_val in np.unique(data_time['initial_anisotropy']).astype(float):

                    data_initial = data_time[data_time['initial_anisotropy'] == anisotropy_val]

                    if data_initial.empty:
                        continue

                    plot_cell_distances_split(data_initial,sim_name,save_folder,title=title_str)

                # ---------------------------
                # Split for chemotaxis & ECM sensitivity
                # ---------------------------
                # data_anisotropy = data_time[
                #     (data_time['chemotaxis_bias'] == 0.2) &
                #     (data_time['ecm_sensitivity'].isin([0.6, 0.8, 1.0]))
                # ]

                plot_cell_distances_split_anisotropy(data_time,sim_name,save_folder,title=title_str)

            plt.close('all')

        print('Box plots done!', flush=True)


    # ---------------------------
    # HELPER FUNCTION: Filter data and get first simulation
    # ---------------------------
    def filter_data(df, time_point=None, **kwargs):
        data = df.copy()
        if time_point is not None:
            data = data[(data['t'] == time_point)]
        for key, val in kwargs.items():
            if isinstance(val, list):
                data = data[data[key].isin(val)]
            else:
                data = data[data[key] == val]
        if data.empty:
            return None, None
        sim_name = int(np.unique(data['simulation'])[0])
        return data, sim_name

    # ---------------------------
    # HEATMAP PLOTS
    # ---------------------------
    if heatmap_plots == True:
        for time_point in heatmap_time_points:
    
            # Group by all "other parameters" to make a heatmap for each combination
            grouped = df.groupby(other_params) if other_params else [((), df)]

            for group_vals, df_group in grouped if other_params else grouped:
                # Prepare a title with the current combination of other parameters
                if other_params and title:
                    if isinstance(group_vals, tuple):
                        title_str = ", ".join(f"{param}={val}" for param, val in zip(other_params, group_vals))
                    else:
                        title_str = ", ".join(f"{param}={val}" for param, val in zip(other_params, [group_vals]))
                else:
                    title_str = None

                # Only plot if dataframe is not empty
                if df_group.empty:
                    continue

                simulation_numbers = np.unique(df_group['simulation']).astype(int)
                simulation_number = simulation_numbers[0]
                # orientation = str(df_group['orientation'].iloc[0])
                # simulation_name = f"{orientation}_{simulation_number}"
                simulation_name = f"{simulation_number}"

                # Filter by current time point
                data = df_group[(df_group['ID'] == 0) & ((df_group['t'] == 0) | (df_group['t'] == time_point))]

                heatmap_spheroid_area(data, simulation_name, save_folder, row_param=row_param, col_param=col_param, title=title_str)

                data = df_group[(df_group['t'] == time_point)]

                heatmap_delaunay(data, simulation_name, save_folder, row_param=row_param, col_param=col_param, title=title_str)
                heatmap_invasion(data, simulation_name, save_folder, row_param=row_param, col_param=col_param, title=title_str)
                heatmap_fiber_orientation(data, simulation_name, data_folder, save_folder, row_param=row_param, col_param=col_param, title=title_str)

            plt.close('all')

    # ---------------------------
    # TIME POINT IMAGE GENERATION
    # ---------------------------
    def generate_image(sim, seed, orientation, t, df, data_folder, save_folder, title):
        """Generates images for a single simulation/seed/timepoint."""
        data = df[(df['simulation'] == sim) &
                  (df['orientation'] == orientation) &
                  (df['seed'] == seed) &
                  (df['t'] == t)]
        if data.empty: return

        time_step = data[data['ID'] == 0].index.values.astype(int)[0]
        snapshot = 'output' + f'{int(time_step):08d}'
        data_folder_sim = data_folder + f'output_{orientation}_{sim}_{seed}/'
        save_name = save_folder + f'images/full_image_{orientation}_{sim}_{seed}_t{int(t):04}.png'

        print(f'{orientation=}, {sim=}, {t=}', flush=True)
        create_plot(data, snapshot, data_folder_sim, save_name, output_plot=True, title=title)
        plt.close('all')

    if time_point_images:
        row_vals =  np.unique(df[row_param])
        col_vals =  np.unique(df[col_param])

        tasks = [(sim, 0, orientation, t) for sim in simulations for orientation in orientations for t in times]
        df_images = df[(df['seed'] == 0) & (df[row_param].isin(row_vals)) & (df[col_param].isin(col_vals))].copy()
        Parallel(n_jobs=-1)(delayed(generate_image)(sim, seed, orientation, t, df_images, data_folder, save_folder, title) for sim, seed, orientation, t in tasks)

    # ---------------------------
    # GRID OF IMAGES
    # ---------------------------
    if images_grid or video:

        # Group ONCE (not inside time loop)
        grouped_all = df.groupby(other_params) if other_params else [((), df)]

        # One "group" = one video
        for group_vals, df_group_all in grouped_all if other_params else grouped_all:

            if df_group_all.empty:
                continue

            # Build title for this parameter combination
            if other_params and title:
                if isinstance(group_vals, tuple):
                    title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, group_vals))
                else:
                    title_str = ", ".join(f"{p}={v}" for p, v in zip(other_params, [group_vals]))
            else:
                title_str = None

            print(title_str)

            # Simulation name belongs to the GROUP, not to time
            sim_name = sorted(df_group_all["simulation"].unique())[0]

            orientation = str(df_group_all['orientation'].iloc[0])

            tasks = []

            if images_grid:
                # ---- LOOP TIMES INSIDE GROUP ----
                for t in times:
                    df_time = df_group_all[df_group_all['t'] == t].copy()
                    if df_time.empty:
                        print(f"No data for time={t} (images grid skipped)", flush=True)
                        continue

                    # Determine grid axes present at this time
                    row_vals = np.unique(df_time[row_param])
                    col_vals = np.unique(df_time[col_param])

                    # Only keep seed 0 + snapshot 0 for thumbnails
                    df_images = df_time[
                        (df_time['ID'] == 0) &
                        (df_time['seed'] == 0) &
                        (df_time[row_param].isin(row_vals)) &
                        (df_time[col_param].isin(col_vals))
                    ].copy()

                    if df_images.empty:
                        continue

                    tasks.append(
                        (df_images, row_param, col_param, t, sim_name, save_folder, title_str)
                    )

                # ---- GENERATE GRID IMAGES FOR THIS GROUP ----
                if tasks:
                    Parallel(n_jobs=-1)(
                        delayed(grid_images)(df_t, r, c, tval, simn, sf, ttl)
                        for df_t, r, c, tval, simn, sf, ttl in tasks
                    )

            # ---- BUILD VIDEO ONCE PER GROUP/SIM ----
            if video:
                video_name = save_folder + f'animations/video_{orientation}_{sim_name}.mp4'
                images = save_folder + f'images/grid_images_{orientation}_{sim_name}_t*.png'

                os.system(
                    f'ffmpeg -y -framerate 10 -pattern_type glob -i "{images}" '
                    f'-c:v libx264 -pix_fmt yuv420p '
                    f'-vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" {video_name}'
                )
                print(f'Video ready for simulation {sim_name}!', flush=True)
        