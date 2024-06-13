from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def plot_chart(data_lists, name_list, x_name=None, y_name=None, x_start=0, std_lists=None, val_min=-np.inf, val_max=np.inf, **kwargs):
    """
    plot a line chart, each line correspond to one list inside data_lists

    param data_lists: [[y1,y2,y3,...], [y1,y2,y3,...], ...]
    param name_list: [name_for_line1, name_for_line2, ...]

    """
    x = np.arange(x_start, x_start+len(data_lists[0]))

    fig = plt.figure(dpi=150)
    fig.clf()
    plots = []
    for idx, (data_list,name) in enumerate(zip(data_lists,name_list)):
        plot, = plt.plot(x, data_list, 'b-+',label = name)
        if std_lists is not None:
            plt.fill_between(x, [np.clip(y1-y2, val_min, val_max) for y1,y2 in zip(data_list,std_lists[idx])], [y1+y2 for y1,y2 in zip(data_list,std_lists[idx])], alpha=0.2)

        plots.append(plot)

    if x_name is not None:    
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)

    plt.legend(handles=plots)

    return fig



