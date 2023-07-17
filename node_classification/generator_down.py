from data_util import create_node_classification_dataset, _create_dgl_graph
import torch
import torch_geometric
from utils.learner import estimate_graphon
import os
import numpy as np
def load_down_data(name,ego_hops,save_path):
    data =create_node_classification_dataset(name).data
    num_nodes = len(data.y)
    edge_index = data.edge_index
    candidates = [i for i in range(num_nodes)]
    subgraphs = []
    for t in range(len(candidates)):
        n_id = candidates[t]
        new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                       num_hops=ego_hops,
                                                                                       edge_index=edge_index,
                                                                                       relabel_nodes=True)
        adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
        subgraphs.append(adj)
    print(len(subgraphs))
    torch.save(subgraphs, save_path + "down/" + name + ".pt")
    return subgraphs
def estimate_generator_down(args,down_path=None):
    if down_path is None:
        subgraphs= load_down_data(args.down_data,ego_hops=args.ego_hops,save_path=args.file_path)
    else:
        subgraphs = torch.load(down_path)
    step_func, non_para_graphon = estimate_graphon(np.array(subgraphs), method=args.method, args=args)
    if not os.path.exists(args.save_path + "down/" + args.down_data ):
        os.makedirs(args.save_path + "down/" + args.down_data )
    np.save(args.save_path + "down/" + args.down_data + "/graphon.npy", non_para_graphon)
    np.save(args.save_path + "down/" + args.down_data + "/func.npy", step_func)
    print("down graphon estimated done!")
    return step_func, non_para_graphon
