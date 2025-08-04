from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn

def delaunay_distance_function(data, save_folder='../results/ecm_fibers/', figure=False):
    """
    Calculate the mean Delaunay distance and optionally plot the Delaunay network.

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - save_folder: Directory path to save the plot (default is '../results/').
    - figure: Boolean to decide whether to generate and save a figure (default is False).

    Returns:
    - edges_lengths_mean: Mean length of the edges in the Delaunay network.
    """

    #### Extract simulation parameters from the DataFrame
    simulation = data['simulation'].iloc[0]
    seed = data['seed'].iloc[0]
    t = data['t'].iloc[0]
    radius = data['radius'].to_numpy()
    position_x = data['position_x'].to_numpy()
    position_y = data['position_y'].to_numpy()

    #### Simulation parameters
    prolif = data['prolif'].iloc[0]
    max_mot_speed = data['max_mot_speed'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_degr = data['ecm_density_rate'].iloc[0]
    r_displacement = data['ecm_displacement_rate'].iloc[0]
    r_orientation = data['fiber_reorientation_rate'].iloc[0]
    chemotaxis_bias = data['chemotaxis_bias'].iloc[0]
    ecm_sensitivity = data['ecm_sensitivity'].iloc[0]
    fiber_reorientation_rate = data['fiber_reorientation_rate'].iloc[0]

    orientation = data['orientation'].iloc[0]
    #### Number of cells
    n = len(position_x)

    #### Prepare coordinates for Delaunay triangulation
    coordinates = np.stack((position_x, position_y), axis=1)
    delaunay_network = Delaunay(coordinates)

    #### Extract edges from the Delaunay triangulation
    edges = []
    for tri in delaunay_network.simplices:
        edges.extend([
            sorted([tri[0], tri[1]]),
            sorted([tri[1], tri[2]]),
            sorted([tri[0], tri[2]])
        ])
    edges = np.unique(edges, axis=0)

    #### Calculate edge lengths
    edge_lengths = []
    for i, j in edges:
        pt1, pt2 = coordinates[i], coordinates[j]
        length = np.linalg.norm(pt1 - pt2)
        edge_lengths.append(length)

    edge_lengths = np.array(edge_lengths)
    edges_lengths_mean = np.mean(edge_lengths)
    edges_lengths_90 = np.percentile(edge_lengths, 90)

    #### Optionally plot the Delaunay network
    if figure and t == 96 * 60 and seed == 0:  # Only plot for the last time point and seed 0
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

        plt.grid(False)
        edge = 500
        plt.xticks([])
        plt.yticks([])
        plt.ylim(-edge, edge)
        plt.xlim(-edge, edge)

        #### Plot points
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'o',
                 color=seaborn.color_palette('colorblind')[2], zorder=0)

        #### Plot edges with color depending on length
        for (i, j), length in zip(edges, edge_lengths):
            color = 'red' if length >= edges_lengths_90 else 'black'
            lw = 1.0 if length >= edges_lengths_90 else 0.5
            plt.plot(
                [coordinates[i][0], coordinates[j][0]],
                [coordinates[i][1], coordinates[j][1]],
                color=color, lw=lw, zorder=1
            )

        #### Add a scale bar to the plot
        scalebar = AnchoredSizeBar(ax.transData, 100, r'100 [$\mu$m]', 'lower right',
                                   pad=0.1, color='black', frameon=False, size_vertical=1,
                                   fontproperties={'size': 15})
        ax.add_artist(scalebar)

        plt.title(f'Delaunay network at t={int(t)} h\n'
                  f'Edges mean length: {edges_lengths_mean:.2f} $\mu$m\n'
                  f'90th percentile: {edges_lengths_90:.2f} $\mu$m\n'
                  f'Simulation: {simulation}, Seed: {seed}\n'
                  f'chemo={chemotaxis_bias}, ecm_sens={ecm_sensitivity}, S_cm={max_mot_speed}\n {r_degr=}, {r_displacement=}, r_orie={fiber_reorientation_rate}', fontsize=15)
        #### Save the plot
        plt.savefig(save_folder + f'statistics/delaunay_{simulation}_{seed}_t{int(t)}.png', bbox_inches='tight', dpi=600)
        plt.close()

    return edges_lengths_mean
