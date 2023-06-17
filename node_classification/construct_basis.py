# from utils.data_util import create_node_classification_dataset, _create_dgl_graph
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import torch_geometric
from utils.learner import estimate_graphon
from sklearn.cluster import KMeans
import argparse
from tqdm import tqdm
from utils.cal_topo_statistcs import cal_topo_graphs
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--epoch', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--gpu', type=int,default=0,help='gpu')
parser.add_argument('--func', type=int, default="0")
parser.add_argument('--method', type=str, default="LP",
                    help='method to estimate graphon')
parser.add_argument('--ego_hops', type=int, default=2)
parser.add_argument('--pre_data', type=str, default="imdb")
parser.add_argument('--down_data', type=str, default="usa_airport")
parser.add_argument('--save_path', type=str, default="data/graphons/",
                    help='downstream data path')
parser.add_argument('--load_path', type=str, default="data/graphons/",
                    help='downstream data path')
parser.add_argument('--file_path', type=str, default="data/dataset/",
                    help='downstream data path')
parser.add_argument('--node_threshold', type=int, default=1000,
                    help='subgraph size threshold')
parser.add_argument('--edge_threshold', type=int, default=10000,
                    help='subgraph size threshold')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
args = parser.parse_args()
assert args.gpu is not None and torch.cuda.is_available()
def load_single_pre_data(name,load_dir = "data/single/"):
    load_path = os.path.join(load_dir, "single_{}.bin".format(name))
    print("start loading graphs!")
    g = load_graphs(load_path)
    num_nodes = int(g[1]['graph_sizes'].item())
    print(num_nodes)
    edge_index = torch.stack(g[0][0].edges())
    print("data loading finished!")
    return edge_index, num_nodes
# def load_pre_data(name,load_dir):
#     dataset = name
#     load_dir = "data/caoyuxuan/gcc/two_data"
#     load_path = os.path.join("merge_{}.bin".format(dataset))
#     print(load_path)
#     g = dgl.data.utils.load_graphs(load_path)
#     graphs = g[0]
#     edge_indices = []
#     for i in range(len(graphs)):
#         edge_indices.append(torch.stack(g[0][i].edges()))
#     return graphs, edge_indices
def get_subgraphs(name,ego_hops):
    pre_datas = name.split("_")
    total_subgraphs = []
    for pre_data in pre_datas:
        subgraphs=[]
        load_dir = "/data/srtpgroup/gcc_modified/"
        edge_index,num_nodes =load_single_pre_data(pre_data,load_dir)
        candidates = np.random.choice(num_nodes, size=int(num_nodes * 0.001), replace=False)
        for t in tqdm(range(len(candidates))):
            n_id = candidates[t]
            new_nodes, new_edge_index, new_index, _ = torch_geometric.utils.k_hop_subgraph(int(n_id),
                                                                                           num_hops=ego_hops,
                                                                                           edge_index=edge_index,
                                                                                           relabel_nodes=True)
            if len(new_nodes) > args.node_threshold > args.node_threshold or len(edge_index) > args.edge_threshold:
                continue
            adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index).toarray()
            subgraphs.append(adj)
            total_subgraphs.append(adj)
        np.save("data/subgraphs/"+pre_data+".npy",np.array(subgraphs))
    return np.array(total_subgraphs)
def construct_basis(args):
    dataname =args.pre_data
    graphs = np.load("data/subgraphs/"+args.pre_data+".npy",allow_pickle=True)
    # graphs = get_subgraphs(dataname,args.ego_hops)
    print(graphs.shape)
    print(graphs[0].shape)
    method = args.method
    splits = ["integer", "topo","domain"]
    save_path = args.save_path
    for split in (splits):
        if split == "integer":
            if not os.path.exists(save_path + "integer/" + dataname):
                os.makedirs(save_path + "integer/" + dataname)
            print(np.array(graphs).shape)
            step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
            np.save(save_path + "integer/" + dataname+ "/graphon.npy", non_para_graphon)
            np.save(save_path + "integer/" + dataname+ "/func.npy", step_func)
        elif split == "domain":
            pre_datas = dataname.split("_")
            for pre in pre_datas:
                if not os.path.exists(save_path + "domain/" + dataname + pre):
                    os.makedirs(save_path + "domain/" + dataname + pre)
                graphs = np.load("data/subgraphs/" +pre + ".npy").tolist()
                step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
                np.save(save_path + "domain/" + pre + "/graphon.npy", non_para_graphon)
                np.save(save_path + "domain/"+ pre + "/func.npy", step_func)
        elif split == "topo":
            if not os.path.exists(save_path + "topo/" + dataname):
                os.makedirs(save_path + "topo/" + dataname)
            topo_feats = cal_topo_graphs(graphs)
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(topo_feats)
            y_kmeans = kmeans.predict(topo_feats)
            cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
            np.save(save_path + "topo/" + dataname+ "/cluster_log.npy", cluster_graphs)
            for k, deg_graphs in enumerate(cluster_graphs):
                step_func, non_para_graphon = estimate_graphon(np.array(graphs)[deg_graphs], method=method, args=args)
                np.save(save_path + "topo/" + dataname+ "/graphon" + str(k) + ".npy",
                        non_para_graphon)
                np.save(save_path + "topo/" + dataname+ "/func" + str(k) + ".npy", step_func)
            print("Kmeans topo Done!")
construct_basis(args)
