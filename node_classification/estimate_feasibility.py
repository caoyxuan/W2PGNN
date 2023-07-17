import math
import os
import numpy as np
import argparse
from utils import simulator
import torch
from tqdm import tqdm
print(os.getcwd())
# from utils.loader import MoleculeDataset
from generator_down import estimate_generator_down
import torch_geometric
from utils.cal_topo_statistcs import cal_topo_graphs
from utils.learner import estimate_graphon
from sklearn.cluster import KMeans
from pre_data_process import get_subgraphs
import cv2
from cal_distance import alpha_fit,mean_fit
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--pre_datasets', type=str, default=["imdb","msu","dblp_snap"],nargs='+',help='pre-training datasets to construct different domains')
parser.add_argument('--down_data', type=str, default="usa_airport",
                    help='downstream data')
parser.add_argument('--load_path', type=str, default="data/demo/dataset/", help="Path to save split data.")
parser.add_argument('--file_path', type=str, default="data/demo/dataset/", help="Path to save raw data.")
parser.add_argument('--save_path', type=str, default="data/demo/graphons/", help="Path to save raw data.")
parser.add_argument('--method', type=str, default="LP", help="Method to estimate graphon.")
parser.add_argument('--variant', type=str, default="domain", help="Method to estimate graphon.")
parser.add_argument('--device', type=int, default=1,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--domain_num', type=int, default=3,
                    help='how many domains')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--epochs', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--node_threshold', type=int, default=1000)

args = parser.parse_args()

def construct_basis(datas,args):
    graphs = []
    if isinstance(datas[0], torch_geometric.data.data.Data):
        for j in range(len(datas)):
            matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                        num_nodes=datas[j].num_nodes).toarray()
            graphs.append(matrix_graph)
    elif isinstance(datas[0], np.ndarray):
        graphs = datas
    step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=args.method, args=args)
    return step_func,non_para_graphon
def estimate_feasibility(pre_funcs, down_graphon,alpha=True):
    max_size = max([pre_funcs[i].shape[0] for i in range(len(pre_funcs))])
    max_size = max(max_size, down_graphon.shape[0])
    pre_funcs = [cv2.resize(pre_funcs[i], dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR) for i in
                 range(len(pre_funcs))]
    down_graphon = cv2.resize(down_graphon, dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
    if alpha == False:
        mean_graphon = mean_fit(pre_funcs)
        mean_dis_gw = simulator.gw_distance(mean_graphon, down_graphon)
        return mean_dis_gw
    else:
        dis, alpha = alpha_fit(pre_funcs, len(pre_funcs), down_graphon, args.learning_rate, args.beta1, args.beta2,
                               args.weight_decay, args.device, args.epochs)
        return dis
def estimate_feas(args):
    #load single downstream data and fit downstream graphon
    down_path = args.file_path + "down/" + args.down_data + ".pt"
    down_func, down_graphon = estimate_generator_down(args, down_path)
    #load pre-train data
    pre_path = args.file_path + "pre/"
    get_subgraphs(args.pre_datasets, pre_path,args.load_path,ego_hops=2,node_threshold=args.node_threshold)
    if args.variant == "topo":
        datas,topo_graphs = [],[]
        for i in range(args.domain_num):
            dataname = args.pre_datasets[i]
            datas += torch.load(pre_path+dataname + ".pt")
            if isinstance(datas[0], torch_geometric.data.data.Data):
                for j in tqdm(range(len(datas))):
                    topo_graphs.append(torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                                    num_nodes=datas[j].num_nodes).toarray())
            elif isinstance(datas[0],np.ndarray):
                topo_graphs = datas
        topo_feats = cal_topo_graphs(topo_graphs)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(topo_feats)
        y_kmeans = kmeans.predict(topo_feats)
        cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
        topo_graphons,topo_funcs = [],[]
        for k, deg_graphs in enumerate(cluster_graphs):
            step_func, non_para_graphon = estimate_graphon(np.array(topo_graphs)[deg_graphs], method=args.method,
                                                           args=args)
            topo_funcs.append(step_func)
            topo_graphons.append(non_para_graphon)
        mean_graphon = mean_fit(topo_graphons)
        mean_dis_gw = estimate_feasibility(mean_graphon, down_graphon, alpha=False)
        alpha_dis_gw = estimate_feasibility(topo_funcs, down_graphon, alpha=True)
        print("topological feasibility without alpha: {}".format(mean_dis_gw))
        print("topological feasibility with alpha: {}".format(alpha_dis_gw))
    elif args.variant == "integr":
        datas = []
        for i in range(args.domain_num):
            dataname = args.pre_datasets[i]
            datas += torch.load(pre_path +dataname + ".pt")
        integr_func , integr_graphon = construct_basis(datas, args)
        mean_dis_gw = estimate_feasibility(integr_graphon, down_graphon, alpha=False)
        print("integrate feasibility : {}".format(mean_dis_gw))
    elif args.variant == "domain":
        if args.domain_num <= 1:
            print("Domain Number should be larger than 1 !")
        else:
            domain_graphons, domain_funcs = [],[]
            for i in range(args.domain_num):
                dataname = args.pre_datasets[i]
                datas = torch.load(pre_path +dataname + ".pt")
                step_func, non_para_graphon = construct_basis(datas, args)
                domain_graphons.append(non_para_graphon)
                domain_funcs.append(step_func)
            mean_graphon = mean_fit(domain_graphons)
            mean_dis_gw = estimate_feasibility(mean_graphon, down_graphon, alpha=False)
            alpha_dis_gw = estimate_feasibility(domain_funcs, down_graphon, alpha=True)
            print("domain feasibility without alpha: {}".format(mean_dis_gw))
            print("domain feasibility with alpha: {}".format(alpha_dis_gw))
estimate_feas(args)
