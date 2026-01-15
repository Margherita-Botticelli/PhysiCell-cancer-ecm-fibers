from calendar import c
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn

def grid_images(df_images, row_param, col_param, t, simulation_name, save_folder, title=False):
    # Assign image paths
    for sim in np.unique(df_images['simulation']):
        orientations = np.unique(df_images['orientation']).astype(str)
        for orientation in orientations:
            df_images.loc[
                (df_images['simulation'] == sim) & (df_images['orientation'] == orientation),
                "img_path"
            ] = (save_folder + f"images/full_image_{orientation}_{sim}_0_t{t:04}.png")

    # Unique values
    row_vals = sorted(df_images[row_param].unique())[::-1]
    col_vals = sorted(df_images[col_param].unique())
    nrows, ncols = len(row_vals), len(col_vals)

    seaborn.set_context("paper", font_scale=0.6)
    seaborn.set_style('ticks')
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    fig, axes = plt.subplots(
        len(row_vals), len(col_vals),
        figsize=(len(col_vals)*2, len(row_vals)*2),
        squeeze=False
    )
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot images and labels
    for i, p1 in enumerate(row_vals):
        for j, p2 in enumerate(col_vals):
            ax = axes[i, j]
            match = df_images[(df_images[row_param] == p1) &
                              (df_images[col_param] == p2)]
            if not match.empty:
                img = mpl.image.imread(match.iloc[0]["img_path"])
                ax.imshow(img)
            ax.axis("off")

            # Column titles (top row)
            if i == 0:
                if col_param == 'chemotaxis_bias':
                    label = f'Chemotaxis bias: {p2}'
                elif col_param == 'initial_anisotropy':
                    label = f'Initial anisotropy: {p2}'
                elif col_param == 'fiber_reorientation_rate':
                    label = f'Fiber reorientation rate: {p2}'
                else:
                    label = f'{col_param}: {p2}'
                ax.set_title(label, fontweight='bold', color='black')

            # Row labels (right side of last column)
            if j == ncols - 1:
                if row_param == 'ecm_sensitivity':
                    rlabel = f'ECM sensitivity: {p1}'
                elif row_param == 'orientation':
                    rlabel = f'Orientation: {p1}'
                else:
                    rlabel = f'{row_param}: {p1}'
                ax.text(1.05, 0.5, rlabel, va='center', ha='left', rotation=270,
                        transform=ax.transAxes, fontweight='bold', color='black')

    # One horizontal colorbar per column, in the dedicated bottom row
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", seaborn.color_palette('colorblind')[3]])
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for j in range(ncols):
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap),
            ax=axes[:, j],  # all axes in column j
            orientation='horizontal', 
            fraction=0.046,  # height of colorbar
            pad=0.02, # space between colorbar and axes
            shrink=0.8
        )
        cbar.outline.set_visible(False)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(length=2) 
        cbar.set_label('ECM density', color='black')

    # Figure title
    if title:
        fig.suptitle(title, fontweight='bold', color='black')

    # Save tightly; no tight_layout() needed with constrained_layout
    fig.savefig(
        save_folder + f'images/grid_images_{simulation_name}_t{int(t):04}.png',
        bbox_inches='tight', pad_inches=0.04, dpi=300
    )

    print(f"Saved grid images for simulation {simulation_name} at time {int(t/60)}.", flush=True)
    plt.close(fig)
