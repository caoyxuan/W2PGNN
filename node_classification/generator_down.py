from data_util import create_node_classification_dataset, _create_dgl_graph
import torch_geometric
from utils.learner import estimate_graphon
import numpy as np
import argparse
import torch
import os
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--epoch', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--gpu', type=int,default=0,help='gpu')
parser.add_argument('--func', type=int, default="0")
parser.add_argument('--ego_hops', type=int, default=2)
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
parser.add_argument('--method', type=str, default="LG",
                    help='downstream data path')
parser.add_argument('--pre_data', type=str, default="imdb_facebook")
parser.add_argument('--down_data', type=str, default="usa_airport")
parser.add_argument('--save_path', type=str, default="data/graphons/",
                    help='downstream data path')
parser.add_argument('--load_path', type=str, default="data/graphons/",
                    help='downstream data path')
parser.add_argument('--file_path', type=str, default="data/dataset/",
                    help='downstream data path')
args = parser.parse_args()
assert args.gpu is not None and torch.cuda.is_available()
def load_down_data(name):
    data =create_node_classification_dataset(name).data
    print(data)
    num_nodes = len(data.y)
    edge_index = data.edge_index
    down_dgl = _create_dgl_graph(edge_index)
    return edge_index,num_nodes,down_dgl
def estimate_generator_down(args):
    edge_index, num_nodes,_ = load_down_data(args.down_data)
    candidates = [i for i in range(num_nodes)]
    subgraphs = []
    for t in range(len(candidates)):
        n_id = candidates[t]
        new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                       num_hops=args.ego_hops,
                                                                                       edge_index=edge_index,
                                                                                       relabel_nodes=True)
        adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
        subgraphs.append(adj)
    print(len(subgraphs))
    step_func, non_para_graphon = estimate_graphon(np.array(subgraphs), method=args.method, args=args)
    if not os.path.exists(args.save_path + "down/" + args.down_data ):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(args.save_path + "down/" + args.down_data )
    np.save(args.save_path + "down/" + args.down_data + "/graphon.npy", non_para_graphon)
    np.save(args.save_path + "down/" + args.down_data + "/func.npy", step_func)

estimate_generator_down(args)
