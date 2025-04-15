import os
import numpy as np
import torch
from IGL_Bench.algorithm.HyperIMBA.Poincare import PoincareModel
from torch_geometric.utils import degree, to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_ricci_and_poincare(dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(current_dir, '../../..')

    ricci_file = os.path.join(file_dir,f'hyperemb/{dataset.data_name}.edge_list')
    keys_file = os.path.join(file_dir,f'hyperemb/{dataset.data_name}_keys.npy')
    values_file = os.path.join(file_dir,f'hyperemb/{dataset.data_name}_values.npy')

    if os.path.exists(ricci_file) and os.path.exists(keys_file) and os.path.exists(values_file):
        print(f"Files for {dataset.data_name} already exist, skipping computation.")
        return
    
    os.makedirs(os.path.dirname(ricci_file), exist_ok=True)
    os.makedirs(os.path.dirname(keys_file), exist_ok=True)
    os.makedirs(os.path.dirname(values_file), exist_ok=True)

    G = to_networkx(dataset)
    orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()  # save an intermediate result

    curvature = "ricciCurvature"
    ricci_results = {}
    ricci = {}
    for i, (n1, n2) in enumerate(list(G_orc.edges()), 0):
        ricci[i] = [int(n1), int(n2), G_orc[n1][n2][curvature]]

    # Save ricci results
    weights = [ricci[i] for i in ricci.keys()]
    np.savetxt(ricci_file, weights, fmt="%d %d %.16f")

    # Poincare Model computation
    degrees = np.array(degree(dataset.edge_index[0], num_nodes=dataset.num_nodes) + degree(dataset.edge_index[1], num_nodes=dataset.num_nodes))
    edges_list = list(dataset.edge_index.t().numpy())
    labels = dict(enumerate(dataset.y.numpy() + 1, 0))
    device = torch.device('cpu')  
    dim = 2
    model = PoincareModel(edges_list, node_weights=degrees * 0.2, node_labels=labels, n_components=dim, 
                          eta=0.01, n_negative=10, name="hierarchy", device=device)
    model.init_embeddings()
    model.train(num_epochs=1)  

    # Save the Poincare model embeddings
    weights = model.embeddings
    keys = np.array([item for item in model.emb_dict.keys()])
    values = np.array([item for item in model.emb_dict.values()])
    np.save(keys_file, keys)
    np.save(values_file, values)

    print(f"Computation for {dataset.data_name} completed and files saved.")

