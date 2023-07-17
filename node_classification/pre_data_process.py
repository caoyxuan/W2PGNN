from data_util import create_node_classification_dataset, _create_dgl_graph
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import torch_geometric
from utils.learner import estimate_graphon
from sklearn.cluster import KMeans
import argparse
from tqdm import tqdm
def load_pre_data(name,load_dir = "data/single/"):
    # load_dir = "/data/srtpgroup/gcc_modified"
    load_path = os.path.join(load_dir, "{}.bin".format(name))
    print("start loading graphs of {}!".format(name))
    g = load_graphs(load_path)
    num_nodes = int(g[1]['graph_sizes'].item())
    print(num_nodes)
    edge_index = torch.stack(g[0][0].edges())
    print("data loading finished!")
    return edge_index, num_nodes

def get_subgraphs(names,pre_path,load_path,ego_hops,node_threshold=None,edge_threshold=None):
    for pre_data in names:
        if not os.path.exists( load_path  + pre_data + ".pt"):
            subgraphs=[]
            load_dir = load_path
            edge_index,num_nodes =load_pre_data(pre_data,load_dir)
            candidates = np.random.choice(num_nodes, size=int(num_nodes * 0.001), replace=False)
            for t in tqdm(range(len(candidates))):
                n_id = candidates[t]
                new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                               num_hops=ego_hops,
                                                                                               edge_index=edge_index,
                                                                                               relabel_nodes=True)
                if len(new_nodes) >node_threshold:
                    continue
                adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
                subgraphs.append(adj)
            torch.save(subgraphs, pre_path  + pre_data + ".pt")
