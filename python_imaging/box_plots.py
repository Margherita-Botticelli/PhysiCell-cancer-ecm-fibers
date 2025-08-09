from cgitb import small
from pyMCDS_ECM import *
import matplotlib.pyplot as plt
import seaborn 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
from collections import defaultdict


def plot_cell_distances(data, save_folder, title=True):
    """
    Plots of cell distances from the origin
    
    Parameters:
    data (DataFrame): The input data containing simulation results.
    save_folder (str): Directory path where the plot will be saved.
    title (bool): Whether to include a title in the plot
    """
    #### Extract relevant parameters from the data
    simulation = data['simulation'].iloc[0]
    t = data['t'].iloc[0]

    #### Simulation parameters
    max_mot_speed = data['max_mot_speed'].iloc[0]
    r_degr = data['ecm_density_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    # color_rib = seaborn.color_palette('colorblind')[0]

    #### Get unique fibre orientations'
    orientations = data['orientation'].unique()
    bins = np.arange(0, 721, 20) # Bins of 20 microns
    seeds = data['seed'].unique()

    bin_data = defaultdict(list)
    perc_cell_count_distances = []
    #### Iterate over fibre orientations and seeds
    for orientation in orientations:
        #### Filter data for the current orientation
        df_orientation = data[data['orientation'] == orientation]

        cell_count_distances_bin = []
        for seed in seeds:
            
            df_seed = df_orientation[df_orientation['seed'] == seed]

            #### Cell distances from the origin
            position_x = df_seed['position_x'].to_numpy()
            position_y = df_seed['position_y'].to_numpy()
            distances = np.sqrt(position_x**2 + position_y**2)

            perc_cell_count_distances_seed = np.percentile(distances, 90)

            cell_count_distances_bin_seed = []
            #### Calculate the number of cells in each bin
            for lo, hi in zip(bins, bins[1:]):
                count = np.sum((distances >= lo) & (distances < hi))
                cell_count_distances_bin_seed.append(count)

            #### Append the cell count distances for the current seed
            cell_count_distances_bin.append(cell_count_distances_bin_seed)


        mean_cell_count_distances_bin = np.mean(cell_count_distances_bin, axis=0)
        std_cell_count_distances_bin = np.std(cell_count_distances_bin, axis=0)
        perc_cell_count_distances.append(np.mean(perc_cell_count_distances_seed))


        for lo, hi, mean, std in zip(bins[:-1], bins[1:], mean_cell_count_distances_bin, std_cell_count_distances_bin):
            #### Append data to the bin_data dictionary
            bin_data['orientation'].append(orientation)
            bin_data['bin_lo'].append(lo)
            bin_data['bin_hi'].append(hi)
            bin_data['mean_cell_count'].append(mean)
            bin_data['std_cell_count'].append(std)
 
    bin_data = pd.DataFrame(bin_data)
    bin_data[bin_data['mean_cell_count'] == 0] = np.nan  # Set mean cell count to NaN if it is 0


    #### plot of cell distances
    plt.figure(figsize=(5, 5), num=simulation+3)
    plt.rcParams.update({'font.weight': 'bold', 
        'axes.labelweight': 'bold'})
    seaborn.set_context("paper")
    seaborn.set_style('ticks')
    seaborn.despine()

    #### DataFrame to store the mean cell count distances for each orientation

    for orientation, perc_cell_count_distances_orient in zip(orientations, perc_cell_count_distances):
        bin_data_orientation = bin_data[bin_data['orientation'] == orientation]
        
        if(orientation == 'random'):
            color_rib = seaborn.color_palette('colorblind')[0]
        elif(orientation == 'radial'):
            color_rib = seaborn.color_palette('colorblind')[1]
        elif(orientation == 'tangential'):
            color_rib = seaborn.color_palette('colorblind')[2]

        # plt.plot( bin_data_orientation['bin_lo'],bin_data_orientation['mean_cell_count'], color=color_rib, linewidth=2, label=orientation)
        # plt.fill_betweenx(bin_data_orientation['bin_lo'], bin_data_orientation['mean_cell_count'] - bin_data_orientation['std_cell_count'],
                    #  bin_data_orientation['mean_cell_count'] + bin_data_orientation['std_cell_count'], color=color_rib, alpha=0.2)

        plt.plot( bin_data_orientation['bin_lo'],bin_data_orientation['mean_cell_count'], color=color_rib, linewidth=2, label=orientation)
        plt.fill_between(bin_data_orientation['bin_lo'], bin_data_orientation['mean_cell_count'] - bin_data_orientation['std_cell_count'],
                         bin_data_orientation['mean_cell_count'] + bin_data_orientation['std_cell_count'], color=color_rib, alpha=0.2)
        #### Add vertical lines for the 90th percentile
        plt.axvline(x=perc_cell_count_distances_orient, color=color_rib, linestyle='--', linewidth=1.5)
    plt.xlabel('Cell distance from origin [$\mu$m]', fontsize=15)
    plt.ylabel('Cell count', fontsize=15)
    plt.xticks(np.arange(0, 721, 120),fontsize=15)
    plt.yticks(np.arange(0, 301, 20),fontsize=15)

    legend_elements = [
        mlines.Line2D([], [], color=seaborn.color_palette('colorblind')[1], markersize=15, label='Radial'),
        mlines.Line2D([], [], color=seaborn.color_palette('colorblind')[0], markersize=15, label='Random'),
        mlines.Line2D([], [], color=seaborn.color_palette('colorblind')[2], markersize=15, label='Tangential')]

    plt.legend(handles=legend_elements, frameon=False, loc='upper left',fontsize=15)
    plt.title(r'$\bf{Cell\,distances\,from\,origin\,at\,{%i}\,h}$' %(t/60) + f'\nchemo={chemotaxis_bias}, ecm_sens={ecm_sensitivity}, S_cm={max_mot_speed}\n {r_degr=}, {r_displacement=}, r_orie={fiber_reorientation_rate}', fontsize=15)

    plt.savefig(save_folder + f'plots/cell_distances_{simulation}.png', bbox_inches="tight", dpi=300)


def plot_cell_distances_split(data, simulation_name, save_folder, title=True):
    """
    Plots of cell distances from the origin
    
    Parameters:
    data (DataFrame): The input data containing simulation results.
    save_folder (str): Directory path where the plot will be saved.
    title (bool): Whether to include a title in the plot
    """
    #### Extract relevant parameters from the data
    t = data['t'].iloc[0]

    #### Get unique fibre orientations'
    orientations = ['Perpendicular','Parallel']
    bins = np.arange(0, 501, 20) # Bins of 20 microns
    seeds = data['seed'].unique()

    bin_data = defaultdict(list)
    perc_cell_count_distances_perpendicular = []
    perc_cell_count_distances_parallel = []
    
    for ecm_sensitivity in data['ecm_sensitivity'].unique():
        for chemotaxis_bias in data['chemotaxis_bias'].unique():
            data_filtered = data[(data['chemotaxis_bias'] == chemotaxis_bias) & (data['ecm_sensitivity'] == ecm_sensitivity)]

            #### Iterate over fibre  and seeds
            for orientation in orientations:
                
                #### Filter data for the current orientation
                if orientation == 'Perpendicular':
                    df_orientation = data_filtered[data_filtered['position_y'] > 20]
                elif orientation == 'Parallel':
                    df_orientation = data_filtered[data_filtered['position_y'] < -20]

                cell_count_distances_bin = []
                perc_cell_count_distances_seed = []

                for seed in seeds:
                    
                    df_seed = df_orientation[df_orientation['seed'] == seed]

                    #### Cell distances from the origin
                    distances = df_seed['position_y'].to_numpy() 
                    
                    
                    if distances[0] < 0:
                        distances = -distances

                    perc_cell_count_distances_seed.append(np.percentile(distances, 50))

                    cell_count_distances_bin_seed = []
                    
                    #### Calculate the number of cells in each bin
                    for lo, hi in zip(bins, bins[1:]):
                        count = np.sum((distances >= lo) & (distances < hi))
                        cell_count_distances_bin_seed.append(count)

                    #### Append the cell count distances for the current seed
                    cell_count_distances_bin.append(cell_count_distances_bin_seed)

                mean_cell_count_distances_bin = np.mean(cell_count_distances_bin, axis=0)
                std_cell_count_distances_bin = np.std(cell_count_distances_bin, axis=0)
                
                if orientation == 'Perpendicular':
                    perc_cell_count_distances_perpendicular.append(np.mean(perc_cell_count_distances_seed))
                elif orientation == 'Parallel': 
                    perc_cell_count_distances_parallel.append(np.mean(perc_cell_count_distances_seed))

                for lo, hi, mean, std in zip(bins[:-1], bins[1:], mean_cell_count_distances_bin, std_cell_count_distances_bin):
                    #### Append data to the bin_data dictionary
                    bin_data['chemotaxis_bias'].append(float(chemotaxis_bias))
                    bin_data['ecm_sensitivity'].append(float(ecm_sensitivity))
                    bin_data['orientation'].append(orientation)
                    bin_data['bin_lo'].append(lo)
                    bin_data['bin_hi'].append(hi)
                    bin_data['mean_cell_count'].append(mean)
                    bin_data['std_cell_count'].append(std)

    bin_data = pd.DataFrame(bin_data)
    bin_data[bin_data['mean_cell_count'] == 0] = np.nan  # Set mean cell count to NaN if it is 0
    # print(bin_data,flush=True)

    bin_data['fillbetween_1'] = bin_data['mean_cell_count'] - bin_data['std_cell_count']
    bin_data['fillbetween_2'] = bin_data['mean_cell_count'] + bin_data['std_cell_count']

    seaborn.set_context("paper")

    g = seaborn.FacetGrid(bin_data, col="chemotaxis_bias", row="ecm_sensitivity", hue='orientation', margin_titles=True,despine=False,legend_out=True, aspect=1.3, height=2)
    
    g.figure.subplots_adjust(wspace=0, hspace=0)
    
    g.map_dataframe(plt.plot, 'bin_lo', 'mean_cell_count')
    #### Change the color of the lines based on orientation
    for ax in g.axes.flat:  
        for line in ax.lines:
            if line.get_label() == 'Parallel':
                line.set_color(seaborn.color_palette('colorblind')[0])
            elif line.get_label() == 'Perpendicular':
                line.set_color(seaborn.color_palette('colorblind')[1])  

    g.add_legend(title='Orientation', label_order=['Parallel', 'Perpendicular'], title_fontsize=15, fontsize=15)
    
    # Force rendering to calculate size correctly
    g.figure.canvas.draw()

    # Now get the legend box size (in display coordinates)
    bbox = g.legend.get_window_extent()

    # Optionally convert to figure coordinates if needed
    bbox_fig_coords = bbox.transformed(g.figure.transFigure.inverted())

    # Move legend using figure coordinates
    seaborn.move_legend(g, "lower center", bbox_to_anchor=(0.5 - bbox_fig_coords.width / 2, -bbox_fig_coords.height), ncol=2, title_fontsize=15, fontsize=15)

    #### Add fill between lines for standard deviation with different colors for each orientation
    g.map_dataframe(plt.fill_between, 'bin_lo', 'fillbetween_1', 'fillbetween_2', alpha=0.2)

    #### Change the color of the fill between lines based on orientation
    for ax in g.axes.flat:
        for fill in ax.collections:
            if fill.get_label() == 'Parallel':
                fill.set_color(seaborn.color_palette('colorblind')[0])
            elif fill.get_label() == 'Perpendicular':
                fill.set_color(seaborn.color_palette('colorblind')[1])

    #### Add vertical lines for the percentile
    for ax, perc_perp, perc_para in zip(g.axes.flat, perc_cell_count_distances_perpendicular, perc_cell_count_distances_parallel):
        ax.axvline(x=perc_perp, linestyle='--', linewidth=1.5, color=seaborn.color_palette('colorblind')[1])
        ax.axvline(x=perc_para, linestyle='--', linewidth=1.5, color=seaborn.color_palette('colorblind')[0])

    #### Set axis labels and titles
    g.set_axis_labels('Distance from origin [$\mu$m]', 'Cell count')
    g.set_titles(col_template='Chemotaxis bias: {col_name:.1f}', row_template='ECM sensitivity: {row_name:.1f}', fontweight='bold') 
    g.set(xticks=np.arange(0, 501, 100), yticks=np.arange(0, 51, 10))
    g.set(xticklabels=np.arange(0, 501, 100), yticklabels=np.arange(0, 51, 10))
    g.tight_layout()
    g.savefig(save_folder + f'plots/cell_distances_split_table_{simulation_name}_t{int(t/60)}.png', bbox_inches="tight", dpi=300)


def plot_cell_distances_split_anisotropy(data, simulation_name, save_folder, title=True):
    """
    Plots of cell distances from the origin
    
    Parameters:
    data (DataFrame): The input data containing simulation results.
    save_folder (str): Directory path where the plot will be saved.
    title (bool): Whether to include a title in the plot
    """
    #### Extract relevant parameters from the data
    t = data['t'].iloc[0]

    #### Get unique fibre orientations'
    orientations = ['Perpendicular','Parallel']
    bins = np.arange(0, 501, 20) # Bins of 20 microns
    seeds = data['seed'].unique()

    bin_data = defaultdict(list)
    perc_cell_count_distances_perpendicular = []
    perc_cell_count_distances_parallel = []
    
    for ecm_sensitivity in data['ecm_sensitivity'].unique():
        for initial_anisotropy in sorted(data['initial_anisotropy'].unique()):
            data_filtered = data[(data['initial_anisotropy'] == initial_anisotropy) & (data['ecm_sensitivity'] == ecm_sensitivity)]

            #### Iterate over fibre  and seeds
            for orientation in orientations:
                
                #### Filter data for the current orientation
                if orientation == 'Perpendicular':
                    df_orientation = data_filtered[data_filtered['position_y'] > 20]
                elif orientation == 'Parallel':
                    df_orientation = data_filtered[data_filtered['position_y'] < -20]

                cell_count_distances_bin = []
                perc_cell_count_distances_seed = []

                for seed in seeds:
                    df_seed = df_orientation[df_orientation['seed'] == seed]

                    #### Cell distances from the origin
                    distances = df_seed['position_y'].to_numpy() 
                    
                    if distances[0] < 0:
                        distances = -distances

                    perc_cell_count_distances_seed.append(np.percentile(distances, 50))

                    cell_count_distances_bin_seed = []
                    #### Calculate the number of cells in each bin
                    for lo, hi in zip(bins, bins[1:]):
                        count = np.sum((distances >= lo) & (distances < hi))
                        cell_count_distances_bin_seed.append(count)

                    #### Append the cell count distances for the current seed
                    cell_count_distances_bin.append(cell_count_distances_bin_seed)

                mean_cell_count_distances_bin = np.mean(cell_count_distances_bin, axis=0)
                std_cell_count_distances_bin = np.std(cell_count_distances_bin, axis=0)
                
                if orientation == 'Perpendicular':
                    perc_cell_count_distances_perpendicular.append(np.mean(perc_cell_count_distances_seed))
                elif orientation == 'Parallel': 
                    perc_cell_count_distances_parallel.append(np.mean(perc_cell_count_distances_seed))

                for lo, hi, mean, std in zip(bins[:-1], bins[1:], mean_cell_count_distances_bin, std_cell_count_distances_bin):
                    #### Append data to the bin_data dictionary
                    bin_data['initial_anisotropy'].append(float(initial_anisotropy))
                    bin_data['ecm_sensitivity'].append(float(ecm_sensitivity))
                    bin_data['orientation'].append(orientation)
                    bin_data['bin_lo'].append(lo)
                    bin_data['bin_hi'].append(hi)
                    bin_data['mean_cell_count'].append(mean)
                    bin_data['std_cell_count'].append(std)

    bin_data = pd.DataFrame(bin_data)
    bin_data[bin_data['mean_cell_count'] == 0] = np.nan  # Set mean cell count to NaN if it is 0
    # print(bin_data,flush=True)

    bin_data['fillbetween_1'] = bin_data['mean_cell_count'] - bin_data['std_cell_count']
    bin_data['fillbetween_2'] = bin_data['mean_cell_count'] + bin_data['std_cell_count']

    g = seaborn.FacetGrid(bin_data, col="initial_anisotropy", row="ecm_sensitivity", hue='orientation', margin_titles=True,despine=False,legend_out=True, aspect=1.3, height=2)
    
    g.figure.subplots_adjust(wspace=0, hspace=0)

    g.map_dataframe(plt.plot, 'bin_lo', 'mean_cell_count')
    #### Change the color of the lines based on orientation
    for ax in g.axes.flat:  
        for line in ax.lines:
            if line.get_label() == 'Parallel':
                line.set_color(seaborn.color_palette('colorblind')[0])
            elif line.get_label() == 'Perpendicular':
                line.set_color(seaborn.color_palette('colorblind')[1])  

    g.add_legend(title='Orientation', label_order=['Parallel', 'Perpendicular'], title_fontsize=10, fontsize=10)
    
    # Force rendering to calculate size correctly
    g.figure.canvas.draw()

    # Now get the legend box size (in display coordinates)
    bbox = g.legend.get_window_extent()

    # Optionally convert to figure coordinates if needed
    bbox_fig_coords = bbox.transformed(g.figure.transFigure.inverted())

    # Move legend using figure coordinates
    seaborn.move_legend(g, "lower center", bbox_to_anchor=(0.5 - bbox_fig_coords.width / 2, -bbox_fig_coords.height), ncol=2, title_fontsize=10, fontsize=10)

    #### Add fill between lines for standard deviation with different colors for each orientation
    g.map_dataframe(plt.fill_between, 'bin_lo', 'fillbetween_1', 'fillbetween_2', alpha=0.2)

    #### Change the color of the fill between lines based on orientation
    for ax in g.axes.flat:
        for fill in ax.collections:
            if fill.get_label() == 'Parallel':
                fill.set_color(seaborn.color_palette('colorblind')[0])
            elif fill.get_label() == 'Perpendicular':
                fill.set_color(seaborn.color_palette('colorblind')[1])

    #### Add vertical lines for the percentile
    for ax, perc_perp, perc_para in zip(g.axes.flat, perc_cell_count_distances_perpendicular, perc_cell_count_distances_parallel):
        ax.axvline(x=perc_perp, linestyle='--', linewidth=1.5, color=seaborn.color_palette('colorblind')[1])
        ax.axvline(x=perc_para, linestyle='--', linewidth=1.5, color=seaborn.color_palette('colorblind')[0])

    #### Set axis labels and titles
    # seaborn.set_style('ticks')
    seaborn.set_context("paper")
    
    g.set_axis_labels('Distance from origin [$\mu$m]', 'Cell count')
    g.set_titles(col_template='Initial anisotropy: {col_name:.1f}', row_template='ECM sensitivity: {row_name:.1f}', fontweight='bold') 
    g.set(xticks=np.arange(0, 501, 100), yticks=np.arange(0, 51, 10))
    g.set(xticklabels=np.arange(0, 501, 100), yticklabels=np.arange(0, 51, 10))
    g.tight_layout()
    g.savefig(save_folder + f'plots/cell_distances_split_anisotropy_table_{simulation_name}_t{int(t/60)}.png', bbox_inches="tight", dpi=300)


def plot_fibre_orientation(data, data_folder, save_folder, title=True):
    #### Extract relevant parameters from the data
    simulation = data['simulation'].iloc[0]
    seed = data['seed'].iloc[0]
    t = data['t'].iloc[0]

    #### Simulation parameters
    max_mot_speed = data['max_mot_speed'].iloc[0]
    r_degr = data['ecm_density_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]
    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    orientation = data['orientation'].iloc[0]

    time_step = data[data['ID'] == 0].index.values.astype(int)[0]

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

    #### Create a histogram of the fiber orientation angles
    fig, ax = plt.subplots(figsize=(5, 5), num=simulation+2)
    plt.rcParams.update({'font.weight': 'bold', 
        'axes.labelweight': 'bold'})
    seaborn.set_context("paper")
    seaborn.set_style('ticks')  
    seaborn.despine()

    #### Show percentage of voxels in each bin
    ax.hist(angles, bins=np.arange(0, 181, 10), weights= 100 *np.ones(len(angles))/len(angles), color=seaborn.color_palette('colorblind')[0], linewidth=2, edgecolor='black', alpha=0.7)

    plt.xlabel('Fiber orientation angle [degrees]', fontsize=15)
    plt.xticks(np.arange(0, 181, 30), fontsize=15)
    plt.ylabel('Percentage of voxels', fontsize=15)
    plt.yticks(np.arange(0, 101, 10), fontsize=15)  # Adjust depending on actual % range

    plt.title(r'$\bf{ECM\,fiber\,orientation\,at\,{%i}\,h}$' %(time_step/60) + f'\nchemo={chemotaxis_bias}, ecm_sens={ecm_sensitivity}, S_cm={max_mot_speed}\n {r_degr=}, {r_displacement=}, r_orie={fiber_reorientation_rate}', fontsize=15)

    plt.savefig(save_folder + f'plots/fibre_orientation_{simulation}_{int(t)}.png', bbox_inches="tight", dpi=300)
