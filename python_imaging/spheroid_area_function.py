from skimage import draw, util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def spheroid_area_function(data, save_folder='../results/', figure=False):
    """
    Compute the area covered by cells in a spheroid mask and optionally save a visual representation.

    Parameters:
    - data: pandas DataFrame containing the simulation data.
    - save_folder: Directory path to save the plot (default is '../results/').
    - figure: Boolean to decide whether to generate and save a figure (default is False).

    Returns:
    - spheroid_area: Area covered by cells in the spheroid mask.
    """
    
    #### Extract simulation parameters from the DataFrame
    # ribose = data['ribose'].iloc[0]
    simulation = data['simulation'].iloc[0]
    seed = data['seed'].iloc[0]
    t = data['t'].iloc[0]
    radius = data['radius'].to_numpy()
    position_x = data['position_x'].to_numpy()
    position_y = data['position_y'].to_numpy()
    prolif = data['prolif'].iloc[0]
    cell_adh = data['cell_adh'].iloc[0]
    cell_rep = data['cell_rep'].iloc[0]
    max_mot_speed = data['max_mot_speed'].iloc[0]
    r_anisotropy = data['fiber_realignment_rate'].iloc[0]
    r_density = data['ecm_density_rate'].iloc[0]
    r_orientation = data['fiber_reorientation_rate'].iloc[0]

    #### Initialize a mask for the spheroid area with zeros
    spheroid_area_mask = np.zeros((5000, 5000))
    
    #### Loop over each cell and draw its contribution to the spheroid mask
    for pos_x, pos_y, r in zip(position_x, position_y, radius):
        #### Convert cell position to mask coordinates
        pos_x = (pos_x + 500) * 5
        pos_y = (500 - pos_y) * 5
        
        #### Draw a circle for each cell on the mask
        circle_coordinates = draw.disk((pos_y, pos_x), r*5, shape=spheroid_area_mask.shape)
        spheroid_area_mask[circle_coordinates] = 1.0

    #### Calculate the area covered by the cells
    spheroid_area = np.count_nonzero(spheroid_area_mask) / 25

    #### Optionally save and plot the spheroid mask
    if figure:
        #### Create a figure for the plot
        fig = plt.figure()
        ax = plt.gca()

        #### Define a colormap from black to white
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", 'white'])
        
        #### Display the spheroid mask using the colormap
        plt.imshow(spheroid_area_mask, extent=(-500, 500, -500, 500), cmap=cmap)
        
        #### Set plot aesthetics
        ax.set_facecolor("black")
        plt.grid(False)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.ylim(-500, 500)
        plt.xlim(-500, 500)

        #### Add a scale bar to the plot
        scalebar = AnchoredSizeBar(ax.transData,
            100, r'100 [$\mu$m]', 'lower right', 
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=1,
            fontproperties={'size':15})
        ax.add_artist(scalebar)

        ax.axis('scaled')

        #### Set plot style
        plt.style.use('ggplot')
        plt.style.use('seaborn-v0_8-colorblind')

        #### Save the plot to the specified folder
        # plt.savefig(save_folder + f'/statistics/cell_image_rib{ribose}_{simulation}_{seed}_t{int(t)}.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/statistics/cell_image_{simulation}_{seed}_t{int(t)}.png', bbox_inches='tight')
        plt.close()

    return spheroid_area
