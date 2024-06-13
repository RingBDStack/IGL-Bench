#from matplotlib import colors
#import matplotlib.pyplot as plt
import numpy as np
import ipdb

def plot_dist1D(data_array, label, step=30, **kwargs):
    """
    plot a scatter plot to visualize distribution, for 1D case

    """

    lower = data_array.min()
    upper = data_array.max()

    x = np.linspace(lower, upper, step)
    interval = (upper-lower)/(step-1)

    y = np.zeros(x.size)

    for data in data_array:
        i = int((data-lower)/(interval+0.0000000001))
        y[i] += 1

    x = x+interval/2

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()

    plt.plot(x, y, color='blue', alpha=0.3,) # distribution fit
    y_max = max(y)

    y = [np.random.rand()*y_max for data in data_array] 
    cmap = kwargs.get('cmap') or 'viridis'
    sizes = kwargs.get('node_size') or 80
    plt.scatter(data_array, y, c=label,  s=sizes, cmap=cmap) #scatter graph

    plt.colorbar()  # show color scale

    return fig