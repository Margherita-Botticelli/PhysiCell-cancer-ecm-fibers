import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import textwrap
from pyMCDS_ECM import *

def plot_generic_heatmap(
    data: pd.DataFrame,
    row_param: str,
    col_param: str,
    value_col: str,
    value_std_col: str = None,
    simulation_name: str = "",
    save_folder: str = "",
    title: str = None,
    suptitle: str = None,
    vmin: float = None,
    vmax: float = None,
    cmap_palette: str = 'colorblind',
    annot_fontsize: int = 15,
    fmt: str = "s"
):
    """
    Generic heatmap plotting function with dynamic row/column parameters.
    Title can include additional information about other parameters.
    """

    def format_label(param):
        if param == "ecm_sensitivity":
            return "ECM sensitivity"
        elif param == "fiber_reorientation_rate":
            return "Fibre reorientation rate"
        elif param == "ecm_density_rate":
            return "ECM degradation rate"
        elif param == "ecm_displacement_rate":
            return "ECM displacement rate"
        else:
            label = param.replace("_", " ")
            return label[0].upper() + label[1:] if label else label

    row_vals = np.flip(np.unique(data[row_param]))
    col_vals = np.unique(data[col_param])

    df = pd.DataFrame(columns=col_vals, index=row_vals).fillna(0.0)
    annot_df = pd.DataFrame(columns=col_vals, index=row_vals).fillna('NaN')

    for _, row in data.iterrows():
        r = row[row_param]
        c = row[col_param]
        df.at[r, c] = row[value_col]
        if value_std_col:
            annot_df.at[r, c] = f"{row[value_col]:.2f}\n±{row[value_std_col]:.2f}"
        else:
            annot_df.at[r, c] = f"{row[value_col]:.2f}"

    annot_arr = annot_df.to_numpy()

    fig, ax = plt.subplots(figsize=(7,7))
    sns.set_context("paper")
    sns.set_style("ticks")
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    color_light = sns.color_palette('colorblind')[0]
    color_dark = sns.color_palette('dark')[0]

    #### Define colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['white', color_light, color_dark])

    hmap = sns.heatmap(
        df, cmap=cmap, vmin=vmin, vmax=vmax,
        annot=annot_arr, annot_kws={"fontsize": annot_fontsize}, fmt=fmt,
        cbar_kws={"orientation": "horizontal", "pad": 0.15}, ax=ax
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=annot_fontsize)
    cbar.set_label(label=suptitle, fontsize=annot_fontsize)

    plt.xlabel(format_label(col_param), fontsize=annot_fontsize)
    plt.ylabel(format_label(row_param), fontsize=annot_fontsize)

    plt.xticks(color='black', fontsize=annot_fontsize)
    plt.yticks(color='black', fontsize=annot_fontsize, rotation=45, va='top')

    if title:
        wrapped_title = textwrap.fill(str(title), width=70)
        plt.title(f'\n{wrapped_title}', fontsize=annot_fontsize)
        # plt.title(r'$\bf{' + str(suptitle).replace(' ', r'\ ') + r'}$' + f'\n{wrapped_title}', fontsize=annot_fontsize)

    if save_folder:
        save_path = f"{save_folder}plots/heatmap_{simulation_name}_{row_param}_vs_{col_param}.png"
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def heatmap_delaunay(data, simulation_name, save_folder, row_param, col_param, title=None):
    """Prepare Delaunay distance data and plot heatmap."""
    rows = []
    grouped = data.groupby([row_param, col_param])

    for (r_val, c_val), df_sub in grouped:

        rows.append({
            row_param: r_val,
            col_param: c_val,
            'mean': df_sub['delaunay_distance'].mean(),
            'std': df_sub['delaunay_distance'].std()
        })

    df_plot = pd.DataFrame(rows)
    plot_generic_heatmap(
        df_plot,
        row_param=row_param,
        col_param=col_param,
        value_col='mean',
        value_std_col='std',
        simulation_name='delaunay_' + simulation_name,
        save_folder=save_folder,
        suptitle = r'Delaunay mean distance [$\mu$m]',
        title = title,
        vmin=15,
        vmax=30
    )


def heatmap_spheroid_area(data, simulation_name, save_folder, row_param, col_param, title=None):
    """Prepare spheroid area growth data and plot heatmap."""
    rows = []
    grouped = data.groupby([row_param, col_param])

    for (r_val, c_val), df_sub in grouped:

        t_min = df_sub['t'].min()
        t_max = df_sub['t'].max()

        ratios = []

        for seed in df_sub['seed'].unique():

            df_seed = df_sub[df_sub['seed'] == seed]

            spheroid_init = df_seed[df_seed['t'] == t_min]['spheroid_area'].to_numpy()
            spheroid_fin = df_seed[df_seed['t'] == t_max]['spheroid_area'].to_numpy()

            if len(spheroid_init) > 0 and len(spheroid_fin) > 0:
                ratios.append((spheroid_fin / spheroid_init).mean())

        rows.append({
            row_param: r_val,
            col_param: c_val,
            'mean': np.mean(ratios) if ratios else np.nan,
            'std': np.std(ratios) if ratios else np.nan
        })

    df_plot = pd.DataFrame(rows)

    plot_generic_heatmap(
        df_plot,
        row_param=row_param,
        col_param=col_param,
        value_col='mean',
        value_std_col='std',
        simulation_name='spheroid_area_' + simulation_name,
        save_folder=save_folder,
        suptitle=r'Spheroid area growth relative to t$_0$',
        title=title,
        vmin=1,
        vmax=6
    )


def heatmap_invasion(data, simulation_name, save_folder, row_param, col_param, title=None):
    """Prepare invasion distance data and plot heatmap."""
    rows = []

    grouped = data.groupby([row_param, col_param])

    for (r_val, c_val), df_sub in grouped:

        invasion_distances = []

        for seed in df_sub['seed'].unique():

            df_seed = df_sub[df_sub['seed'] == seed]

            distances = np.sqrt(
                df_seed['position_x']**2 +
                df_seed['position_y']**2
            )

            if len(distances) > 0:
                invasion_distances.append(np.percentile(distances, 95))
                # invasion_distances.append(np.percentile(df_seed['position_y'].to_numpy() + 500, 50))

        rows.append({
            row_param: r_val,
            col_param: c_val,
            'mean': np.mean(invasion_distances) if invasion_distances else np.nan,
            'std': np.std(invasion_distances) if invasion_distances else np.nan
        })

    df_plot = pd.DataFrame(rows)

    plot_generic_heatmap(
        df_plot,
        row_param=row_param,
        col_param=col_param,
        value_col='mean',
        value_std_col='std',
        simulation_name='invasion_' + simulation_name,
        save_folder=save_folder,
        suptitle=r'Invasion [$\mu$m]',
        title=title,
        vmin=0,
        vmax=700#1000
    )


def heatmap_fiber_orientation(data, simulation_name, data_folder, save_folder, row_param, col_param, title=None):
    """
    Prepare ECM fiber orientation data and plot heatmap.

    Computes the normalised fibre orientation relative to 90°
    from ECM fields and aggregates mean ± std across seeds.
    """

    rows = []

    # Extract time step information
    t = data['t'].iloc[0]
    time_step = data[data['ID'] == 0].index.values.astype(int)[0]
    orientation = data['orientation'].iloc[0]

    grouped = data.groupby([row_param, col_param])

    for (r_val, c_val), df_sub in grouped:

        mean_angle_seed = []

        for seed in df_sub['seed'].unique():

            df_seed = df_sub[df_sub['seed'] == seed]
            simulation = df_seed['simulation'].iloc[0]

            snapshot = 'output' + '{:08d}'.format(int(time_step))
            data_folder_sim = (
                data_folder +
                f"output_{orientation}_{simulation}_{seed}/"
            )

            mcds = pyMCDS(snapshot + '.xml', data_folder_sim)
            mcds.load_ecm(snapshot + '_ECM.mat', data_folder_sim)

            ECM_x = mcds.data['ecm']['ECM_fields']['x_fiber_orientation'][:, :, 0]
            ECM_y = mcds.data['ecm']['ECM_fields']['y_fiber_orientation'][:, :, 0]

            angles = np.arctan2(ECM_y, ECM_x) * (180 / np.pi)
            angles[angles < 0] += 180
            angles = angles.flatten()

            # Normalise relative to 90 degrees
            angles = np.abs(angles - 90) / 90

            mean_angle_seed.append(np.percentile(angles, 50))

        rows.append({
            row_param: r_val,
            col_param: c_val,
            'mean': np.mean(mean_angle_seed) if mean_angle_seed else np.nan,
            'std': np.std(mean_angle_seed) if mean_angle_seed else np.nan
        })

    df_plot = pd.DataFrame(rows)

    plot_generic_heatmap(
        df_plot,
        row_param=row_param,
        col_param=col_param,
        value_col='mean',
        value_std_col='std',
        simulation_name='fiber_orientation_' + simulation_name,
        save_folder=save_folder,
        suptitle='Normalised fibre orientation',
        title=title,
        vmin=0,
        vmax=1
    )