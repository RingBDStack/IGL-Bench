from xml.etree.ElementInclude import XINCLUDE
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import ipdb

#draw heatmap

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    #ax.set_xticks(np.arange(data.shape[1]))
    #ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    #ax.set_xticklabels(col_labels)
    #ax.set_yticklabels(row_labels)
    
    #show every 4 ticks
    ax.set_xticks(np.arange(0,data.shape[1],1))
    ax.set_yticks(np.arange(0,data.shape[0],1))
    ## ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_single_grid(data_grid, name, x_name=None, y_name=None, x_start=0, std_lists=None, val_min=-np.inf, val_max=np.inf, **kwargs):
    """
    plot a grid chart, with color indicates values

    param data_grid: 2D numpy.array, of shape (x, y)
    param name: name for grid

    """

    x_index = np.arange(data_grid.shape[0]).tolist()
    y_index = np.arange(data_grid.shape[1]).tolist()

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = plt.gca()

    im, cbar = heatmap(data_grid.T, y_index, x_index, ax=ax,
                   cmap="YlGn", vmin=0, vmax=1,
                cbarlabel="memory sel ratio")

    fig.tight_layout()

    ax.set_title("{} w.r.t {}".format(y_name, x_name))

    ax.set_aspect(5)

    return fig



