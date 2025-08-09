import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy.ndimage
from pyMCDS_ECM import *

def plots_chemo_vs_ecm_sensitivity_delaunay(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between chemotaxis bias and ECM sensitivity for the Delaunay distance

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    orientation = data['orientation'].iloc[0]
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    seeds = np.unique(data['seed'])
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    r_orientation = data['fiber_reorientation_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]

    # chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    # ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    delaunay_distance_mean = []
    delaunay_distance_std = []
    chemotaxis_bias = []
    ecm_sensitivity = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        df_sim = data[data['simulation'] == simulation]

        delaunay_distance = df_sim['delaunay_distance'].to_numpy()
        delaunay_distance_mean.append(np.mean(delaunay_distance))
        delaunay_distance_std.append(np.std(delaunay_distance))

        chemotaxis_bias.append(round(float(df_sim['chemotaxis_bias'].iloc[0]), 3))
        ecm_sensitivity.append(round(float(df_sim['ecm_sensitivity'].iloc[0]), 5))

    #### Create DataFrame for heatmap
    columns = np.unique(chemotaxis_bias)
    index = np.flip(np.unique(ecm_sensitivity))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, mean, std in zip(chemotaxis_bias, ecm_sensitivity, delaunay_distance_mean, delaunay_distance_std):
        df[x][y] = mean
        annot = f"{mean:.2f}\n±{std:.2f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()


    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df,cmap=cmap,vmin=15, vmax=30,ax=ax)
    hmap = seaborn.heatmap(df, cmap=cmap, vmin=10, vmax=30, annot=annot_arr, annot_kws={"fontsize":15}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15}
    )
    cbar = ax.collections[0].colorbar
    # Place colorbar at the bottom (default for horizontal)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'Delaunay mean distance [$\mu$m]',fontsize=15)


    #### Set axis labels and title
    plt.ylabel(f'ECM sensitivity', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Chemotaxis bias', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    if title:
        # plt.title(r'$\bf{Delaunay\,mean\,distance\,at\,{%i}\,h}$' %(max(t)/60) + '\n' r'Proliferation $r_{div}$=0.00072 min$^{-1}$',color='black', fontsize=15)
        
        # plt.title(r'Proliferation $r_{div}$=%f min$^{-1}$' %(round(prolif,3)),fontsize=15)

        plt.title(r'$\bf{Delaunay\,mean\,distance\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density} {r_displacement=}, r_orie={fiber_reorientation_rate}\n{orientation=}, {initial_ecm_density=}', fontsize=15)

    plt.savefig(save_folder + f'plots/chemo_vs_ecm_sensitivity_delaunay_{orientation}_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()


def plots_chemo_vs_ecm_sensitivity_spheroid_area_growth(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between max migration speed and degradation rate for spheroid area growth

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    orientation = data['orientation'].iloc[0]
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    r_orientation = data['fiber_reorientation_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]
    # chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    # ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    spheroid_area_ratio_mean = []
    spheroid_area_ratio_std = []
    chemotaxis_bias = []
    ecm_sensitivity = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        df_sim = data[data['simulation'] == simulation]

        spheroid_area_init = df_sim[df_sim['t'] == min(t)]['spheroid_area'].to_numpy()
        spheroid_area_fin = df_sim[df_sim['t'] == max(t)]['spheroid_area'].to_numpy()
        spheroid_area_ratio = spheroid_area_fin / spheroid_area_init

        spheroid_area_ratio_mean.append(np.mean(spheroid_area_ratio))
        spheroid_area_ratio_std.append(np.std(spheroid_area_ratio))

        chemotaxis_bias.append(round(float(df_sim['chemotaxis_bias'].iloc[0]), 3))
        ecm_sensitivity.append(round(float(df_sim['ecm_sensitivity'].iloc[0]), 5))

    #### Create DataFrame for heatmap
    columns = np.unique(chemotaxis_bias)
    index = np.flip(np.unique(ecm_sensitivity))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, area_mean, area_std in zip(chemotaxis_bias, ecm_sensitivity, spheroid_area_ratio_mean, spheroid_area_ratio_std):
        df[x][y] = area_mean
        annot = f"{area_mean:.2f}\n±{area_std:.2f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()


    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df, cmap=cmap, vmin=1, vmax=8,ax=ax)
    hmap = seaborn.heatmap(df,cmap=cmap, vmin=1, vmax=8, annot=annot_arr, annot_kws={"fontsize":15}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15}
    )
    cbar = ax.collections[0].colorbar
    # Place colorbar at the bottom (default for horizontal)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Growth relative to t$_0$',fontsize=15)


    #### Set axis labels and title
    plt.ylabel(f'ECM sensitivity', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Chemotaxis bias', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    
    if title:
        # plt.title(r'$\bf{Spheroid\,growth\,relative\,to\,t_0\,at\,{%i}\,h}$' %(max(t)/60) + '\n' r'Proliferation $r_{div}$=0.00072 min$^{-1}$',color='black', fontsize=15)
        
        # plt.title(r'Proliferation $r_{div}$=%f min$^{-1}$' %(round(prolif,3)),fontsize=15)

        plt.title(r'$\bf{Spheroid\,growth\,relative\,to\,t_0\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density}, {r_displacement=}, r_orie={fiber_reorientation_rate}\n{orientation=}, {initial_ecm_density=}', fontsize=15)

    plt.savefig(save_folder + f'plots/chemo_vs_ecm_sensitivity_spheroid_area_growth_{orientation}_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()


def plots_chemo_vs_ecm_sensitivity_invasion(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between max migration speed and degradation rate for invasion

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    seeds = data['seed'].unique()
    orientation = data['orientation'].iloc[0]
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    r_orientation = data['fiber_reorientation_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]

    # chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    # ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    invasion_mean = []
    invasion_std = []
    chemotaxis_bias = []
    ecm_sensitivity = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        df_sim = data[data['simulation'] == simulation]

        invasion = []
        for seed in seeds:
            df_seed = df_sim[(df_sim['seed'] == seed)]
            position_x = df_seed['position_x'].to_numpy()
            position_y = df_seed['position_y'].to_numpy()
            # distances = np.sqrt(position_x**2 + position_y**2)

            # invasion_distance = np.percentile(distances, 95)
            invasion_distance = np.percentile(position_y + 500, 50)
            invasion.append(invasion_distance)

        #### Calculate mean and standard deviation of cell count
        invasion_mean.append(np.mean(invasion))
        invasion_std.append(np.std(invasion))

        chemotaxis_bias.append(round(float(df_sim['chemotaxis_bias'].iloc[0]), 3))
        ecm_sensitivity.append(round(float(df_sim['ecm_sensitivity'].iloc[0]), 5))

    #### Create DataFrame for heatmap
    columns = np.unique(chemotaxis_bias)
    index = np.flip(np.unique(ecm_sensitivity))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, area_mean, area_std in zip(chemotaxis_bias, ecm_sensitivity, invasion_mean, invasion_std):
        df[x][y] = area_mean
        annot = f"{area_mean:.1f}\n±{area_std:.1f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()

    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df, cmap=cmap, vmin=1, vmax=8,ax=ax)
    # hmap = seaborn.heatmap(df,cmap=cmap, vmin=100, vmax=600, annot=annot_arr, annot_kws={"fontsize":15}, fmt="s")
    hmap = seaborn.heatmap(df,cmap=cmap, vmin=0, vmax=1000, annot=annot_arr, annot_kws={"fontsize":15}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15}
    )
    cbar = ax.collections[0].colorbar
    # Place colorbar at the bottom (default for horizontal)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'Invasion [$\mu$m]',fontsize=15)

    #### Set axis labels and title
    plt.ylabel(f'ECM sensitivity', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Chemotaxis bias', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    
    if title:
        plt.title(r'$\bf{Invasion\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density}, {r_displacement=}, r_orie={fiber_reorientation_rate}\n{orientation=}, {initial_ecm_density=}', fontsize=15)


    plt.savefig(save_folder + f'plots/chemo_vs_ecm_sensitivity_invasion_{orientation}_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()


def plots_reorientation_vs_orientation_delaunay(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between chemotaxis bias and ECM sensitivity for the Delaunay distance

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    seeds = np.unique(data['seed'])
    initial_anisotropy = data['initial_anisotropy'].iloc[0]
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    # r_orientation = data['fiber_reorientation_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]
    max_speed = data['max_mot_speed'].iloc[0]

    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    delaunay_distance_mean = []
    delaunay_distance_std = []
    r_orientation = []
    orientation = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))
    orientations = list(dict.fromkeys(data['orientation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        for o in orientations:
            df_sim = data[(data['simulation'] == simulation) & (data['orientation'] == o)]

            delaunay_distance = df_sim['delaunay_distance'].to_numpy()
            delaunay_distance_mean.append(np.mean(delaunay_distance))
            delaunay_distance_std.append(np.std(delaunay_distance))

            r_orientation.append(df_sim['fiber_reorientation_rate'].iloc[0])
            orientation.append(o)

    #### Create DataFrame for heatmap
    columns = np.unique(r_orientation)
    index = np.flip(np.unique(orientation))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, mean, std in zip(r_orientation, orientation, delaunay_distance_mean, delaunay_distance_std):
        df[x][y] = mean
        annot = f"{mean:.2f}\n±{std:.2f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()


    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df,cmap=cmap,vmin=15, vmax=30,ax=ax)
    hmap = seaborn.heatmap(
        df, cmap=cmap, vmin=15, vmax=40, annot=annot_arr,
        annot_kws={"fontsize":15, 'color': 'white'}, fmt="s",
        cbar_kws={"orientation": "horizontal", "pad": 0.15}
    )
    cbar = ax.collections[0].colorbar
    # Place colorbar at the bottom (default for horizontal)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'Delaunay mean distance [$\mu$m]', fontsize=15)


    #### Set axis labels and title
    plt.ylabel(f'Initial fibre orientation', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Reorientation rate', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    if title:
        # plt.title(r'$\bf{Delaunay\,mean\,distance\,at\,{%i}\,h}$' %(max(t)/60) + '\n' r'Proliferation $r_{div}$=0.00072 min$^{-1}$',color='black', fontsize=15)
        
        # plt.title(r'Proliferation $r_{div}$=%f min$^{-1}$' %(round(prolif,3)),fontsize=15)

        plt.title(r'$\bf{Delaunay\,mean\,distance\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density}, {r_displacement=}\nchemo_bias={chemotaxis_bias}, ECM_sens={ecm_sensitivity},S_cm={max_speed}\n{initial_ecm_density=}, {initial_anisotropy=}', fontsize=15)

    plt.savefig(save_folder + f'plots/reorientation_vs_orientation_delaunay_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()


def plots_reorientation_vs_orientation_spheroid_area_growth(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between max migration speed and degradation rate for spheroid area growth

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    initial_anisotropy = data['initial_anisotropy'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]
    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]
    max_speed = data['max_mot_speed'].iloc[0]

    spheroid_area_ratio_mean = []
    spheroid_area_ratio_std = []
    r_orientation = []
    orientation = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))
    orientations = list(dict.fromkeys(data['orientation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        for o in orientations:
            df_sim = data[(data['simulation'] == simulation) & (data['orientation'] == o)]

            spheroid_area_init = df_sim[df_sim['t'] == min(t)]['spheroid_area'].to_numpy()
            spheroid_area_fin = df_sim[df_sim['t'] == max(t)]['spheroid_area'].to_numpy()
            spheroid_area_ratio = spheroid_area_fin / spheroid_area_init

            spheroid_area_ratio_mean.append(np.mean(spheroid_area_ratio))
            spheroid_area_ratio_std.append(np.std(spheroid_area_ratio))

            r_orientation.append(df_sim['fiber_reorientation_rate'].iloc[0])
            orientation.append(o)

    #### Create DataFrame for heatmap
    columns = np.unique(r_orientation)
    index = np.flip(np.unique(orientation))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, area_mean, area_std in zip(r_orientation, orientation, spheroid_area_ratio_mean, spheroid_area_ratio_std):
        df[x][y] = area_mean
        annot = f"{area_mean:.2f}\n±{area_std:.2f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()

    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df, cmap=cmap, vmin=1, vmax=8,ax=ax)
    hmap = seaborn.heatmap(df, cmap=cmap, vmin=1, vmax=6, annot=annot_arr, annot_kws={"fontsize":15, 'color':'white'}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15})
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Growth relative to t$_0$', fontsize=15)

    #### Set axis labels and title
    plt.ylabel(f'Initial fibre orientation', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Reorientation rate', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    
    if title:
        # plt.title(r'$\bf{Spheroid\,growth\,relative\,to\,t_0\,at\,{%i}\,h}$' %(max(t)/60) + '\n' r'Proliferation $r_{div}$=0.00072 min$^{-1}$',color='black', fontsize=15)
        
        # plt.title(r'Proliferation $r_{div}$=%f min$^{-1}$' %(round(prolif,3)),fontsize=15)

        plt.title(r'$\bf{Spheroid\,growth\,relative\,to\,t_0\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density}, {r_displacement=}\nchemo_bias={chemotaxis_bias}, ECM_sens={ecm_sensitivity},S_cm={max_speed}\n{initial_ecm_density=}, {initial_anisotropy=}', fontsize=15)

    plt.savefig(save_folder + f'plots/reorientation_vs_orientation_spheroid_area_growth_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()


def plots_reorientation_vs_orientation_invasion(data, simulation_name, save_folder, title=True):
    """
    Plot the relationship between max migration speed and degradation rate for invasion

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - simulation_name: String identifier for the simulation.
    - save_folder: Directory path to save the plot.
    - title: Boolean to decide whether to include a title on the plot
    """
    # plt.figure()
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})

    #### Collect simulation parameters from the DataFrame
    seeds = data['seed'].unique()
    t = np.unique(data['t']).astype(int)
    prolif = round(float(data['prolif'].iloc[0]), 5)
    cell_adh = round(float(data['cell_adh'].iloc[0]), 1)
    cell_rep = data['cell_rep'].iloc[0]
    initial_ecm_density = data['initial_ecm_density'].iloc[0]
    initial_anisotropy = data['initial_anisotropy'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    # r_orientation = data['fiber_reorientation_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]
    max_speed = data['max_mot_speed'].iloc[0]

    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    invasion_mean = []
    invasion_std = []
    r_orientation = []
    orientation = []

    #### Unique simulations in the dataset
    simulations = list(dict.fromkeys(data['simulation'].values.tolist()))
    orientations = list(dict.fromkeys(data['orientation'].values.tolist()))

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        for o in orientations:
            df_sim = data[(data['simulation'] == simulation) & (data['orientation'] == o)]
        
            invasion = []
            for seed in seeds:
                df_seed = df_sim[(df_sim['seed'] == seed)]
                position_x = df_seed['position_x'].to_numpy()
                position_y = df_seed['position_y'].to_numpy()
                distances = np.sqrt(position_x**2 + position_y**2)

                invasion_distance = np.percentile(distances, 95)
                invasion.append(invasion_distance)

            #### Calculate mean and standard deviation of cell count
            invasion_mean.append(np.mean(invasion))
            invasion_std.append(np.std(invasion))

            r_orientation.append(df_sim['fiber_reorientation_rate'].iloc[0])
            orientation.append(o)

    #### Create DataFrame for heatmap
    columns = np.unique(r_orientation)
    index = np.flip(np.unique(orientation))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')

    #### Fill DataFrame with means and standard deviations
    for x, y, area_mean, area_std in zip(r_orientation, orientation, invasion_mean, invasion_std):
        df[x][y] = area_mean
        annot = f"{area_mean:.1f}\n±{area_std:.1f}"
        annot_df[x][y] = annot

    annot_arr = annot_df.to_numpy()

    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    #### Plot heatmap
    # hmap = seaborn.heatmap(df, cmap=cmap, vmin=1, vmax=8,ax=ax)
    hmap = seaborn.heatmap(df, cmap=cmap, vmin=100, vmax=700, annot=annot_arr, annot_kws={"fontsize":15, 'color':'white'}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15})
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'Invasion [$\mu$m]', fontsize=15)

    #### Set axis labels and title
    plt.ylabel(f'Initial fibre orientation', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15,rotation=45,va='top')
    plt.xlabel(f'Reorientation rate', color='black', fontsize=15)
    plt.xticks(color='black', fontsize=15)
    
    if title:
        plt.title(r'$\bf{Invasion\,at\,{%i}\,h}$' %(max(t)/60) + f'\nr_degr={r_density}, {r_displacement=}\nchemo_bias={chemotaxis_bias}, ECM_sens={ecm_sensitivity},S_cm={max_speed}\n{initial_ecm_density=}, {initial_anisotropy=}', fontsize=15)


    plt.savefig(save_folder + f'plots/reorientation_vs_orientation_invasion_{simulation_name}_t{int(max(t)/60)}.png', bbox_inches="tight")
    plt.close()

def fiber_orientation_heatmap(data, data_folder, simulation_name, save_folder, title=True):

    #### Extract relevant parameters from the data
    t = data['t'].iloc[0]
    time_step = data[data['ID'] == 0].index.values.astype(int)[0]

    #### Simulation parameters
    orientation = data['orientation'].iloc[0]

    #### Unique simulations in the dataset
    seeds = data['seed'].unique()
    simulations = data['simulation'].unique()

    mean_angle = []
    std_angle = []
    chemotaxis_bias_list = []
    ecm_sensitivity_list = []

    #### Iterate over each simulation to compute statistics
    for simulation in simulations:
        df_sim = data[(data['simulation'] == simulation)]

        chemotaxis_bias_list.append(df_sim['chemotaxis_bias'].iloc[0])
        ecm_sensitivity_list.append(df_sim['ecm_sensitivity'].iloc[0])
        
        mean_angle_seed = []
        std_angle_seed = []
        for seed in seeds:
            # print(f"Processing simulation: {simulation}, seed: {seed}, chemotaxis bias: {df_sim['chemotaxis_bias'].iloc[0]}, ECM sensitivity: {df_sim['ecm_sensitivity'].iloc[0]}", flush=True)

            #### Get time point to find snapshot
            snapshot = 'output' + '{:08d}'.format(int(time_step))
            data_folder_sim = data_folder + f"output_{orientation}_{simulation}_{seed}/"

            mcds = pyMCDS(snapshot + '.xml', data_folder_sim)
            mcds.load_ecm(snapshot + '_ECM.mat', data_folder_sim)

            #### Extract ECM fiber orientation components
            ECM_x = mcds.data['ecm']['ECM_fields']['x_fiber_orientation'][:, :, 0]
            ECM_y = mcds.data['ecm']['ECM_fields']['y_fiber_orientation'][:, :, 0]

            #### Calculate the angle of the fiber orientation and put into list
            angles = np.arctan2(ECM_y, ECM_x) * (180 / np.pi)
            angles[angles < 0] += 180  # Convert negative angles to positive
            angles = angles.flatten()

            angles = np.abs(angles - 90) /90  # Normalize angles to [0, 1] range

            mean_angle_seed.append(np.percentile(angles,50))
        
        mean_angle.append(np.mean(mean_angle_seed))
        std_angle.append(np.std(mean_angle_seed))

    #### Create a heatmap with column chemotaxis bias and index ECM sensitivity showing the mean and std of fiber orientation
    columns = np.unique(data['chemotaxis_bias'])
    index = np.flip(np.unique(data['ecm_sensitivity']))
    df = pd.DataFrame(columns=columns, index=index).fillna(0.0)
    annot_df = pd.DataFrame(columns=columns, index=index).fillna('NaN')
    for x, y, mean, std in zip(chemotaxis_bias_list, ecm_sensitivity_list, mean_angle, std_angle):
        df[x][y] = mean
        annot = f"{mean:.2f}\n±{std:.2f}"
        # annot = f"{std:.2f}"
        annot_df[x][y] = annot
    
    annot_arr = annot_df.to_numpy()

    #### Create heatmap
    fig, ax = plt.subplots(figsize=(7,7))
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold',
        'axes.labelweight': 'bold'})
    color_light = seaborn.color_palette('colorblind')[0]
    color_dark = seaborn.color_palette('dark')[0]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])
    hmap = seaborn.heatmap(df, cmap=cmap, vmin=0, vmax=1, annot=annot_arr, annot_kws={"fontsize":15}, fmt="s", cbar_kws={"orientation": "horizontal", "pad": 0.15}
    )
    cbar = ax.collections[0].colorbar
    # Place colorbar at the bottom (default for horizontal)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Normalised fibre orientation relative to 90°', fontsize=15)

    #### Save the heatmap
    plt.ylabel(f'ECM sensitivity', color='black', fontsize=15)
    plt.yticks(color='black', fontsize=15, rotation=45, va='top')
    plt.xlabel(f'Chemotaxis bias', color='black', fontsize=15)      
    plt.xticks(color='black', fontsize=15)
    if title:
        plt.title(r'$\bf{Fiber\,orientation\,angle\,at\,{%i}\,h}$' %(int(t/60)) + f'\nOrientation={orientation}', fontsize=15)
    plt.savefig(save_folder + f'plots/fiber_orientation_heatmap_{orientation}_{simulation_name}_t{int(t/60)}.png', bbox_inches="tight")
    plt.close()

