from matplotlib import colors
import numpy as np
import ipdb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA

def plot_tsne(x_array, label_array, **kwargs):
    
    if x_array.shape[-1] >10:
        pca = PCA(n_components=4)
        trans_x = pca.fit_transform(x_array)
        scale = 100/(trans_x.max()+0.000000000000001)
        x_array = trans_x*scale

    fig = plt.figure(dpi=150)
    fig.clf()

    X_embedded = TSNE(n_components=2,random_state=21).fit_transform(x_array[:,:])

    fig.clf()

    cmap = kwargs.get('cmap') or 'viridis'
    sizes = kwargs.get('node_size') or 80
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=label_array,  s=sizes, cmap=cmap) #scatter graph

    return fig