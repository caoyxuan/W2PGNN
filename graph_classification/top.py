import math
import os
import numpy as np
import argparse
from utils import simulator
import torch
from tqdm import tqdm
from utils.loader import MoleculeDataset
from generator_down import estimate_generator_down
import torch_geometric
from utils.cal_topo_statistcs import cal_topo_graphs
from utils.learner import estimate_graphon
from sklearn.cluster import KMeans
import cv2
from estimate_feasibility import alpha_fit,mean_fit
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--pre_data', type=str, default="zinc_standard_agent",
                    help='pretrain data')
parser.add_argument('--down_data', type=str, default="bace",
                    help='downstream data')
parser.add_argument('--load_path', type=str, default="data/demo/dataset/", help="Path to save split data.")
parser.add_argument('--file_path', type=str, default="data/demo/dataset/", help="Path to save raw data.")
parser.add_argument('--save_path', type=str, default="data/demo/graphons/", help="Path to save raw data.")
parser.add_argument('--method', type=str, default="LP", help="Method to estimate graphon.")
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--domain_num', type=int, default=5,
                    help='how many domains')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')

args = parser.parse_args()
def construct_basis(datas,args):
    graphs = []
    for j in range(len(datas)):
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                    num_nodes=datas[j].num_nodes).toarray()
        graphs.append(matrix_graph)
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
        dis, alpha = alpha_fit(pre_funcs,len(pre_funcs),down_graphon)
        return np.argmax(alpha)
def choose_graphs(args):
    pre_path = args.file_path+"pre/"+args.pre_data
    down_path = args.file_path+"down/"+args.down_data+".pt"
    down_data = torch.load(down_path)
    down_func,down_graphon = estimate_generator_down(args,down_data,down_path)
    pre_datas=[]
    graph_ids = []
    topo_graphs = []
    cnt = 0
    if args.domain_num > 1:
        domain_feases = []
        domain_graphons = []
        domain_funcs = []
        for i in range(args.domain_num):
            datas = torch.load(pre_path + "/domain" + str(i) + ".pt")
            step_func,non_para_graphon = construct_basis(datas,args)
            domain_graphons.append(non_para_graphon)
            domain_funcs.append(step_func)
            feas = estimate_feasibility([step_func],down_graphon,alpha=False)
            domain_feases.append(feas)
            graph_ids += [i + cnt for i in range(len(datas))]
            pre_datas.append(datas)
            cnt += len(datas)
        most_feas_domain = np.argmin(domain_feases)
        print(most_feas_domain)
        max_domain_id = estimate_feasibility(domain_funcs,down_graphon)
        print(max_domain_id)
        datas = pre_datas[max_domain_id ]
    else:
        datas = torch.load(pre_path+".pt")
        # topo_graphs = datas
    for j in tqdm(range(len(datas))):
        topo_graphs.append( torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                    num_nodes=datas[j].num_nodes).toarray())

    part_num = int(math.log(len(topo_graphs)))-1
    # print(part_num)
    topo_feats = cal_topo_graphs(topo_graphs)
    for iter in range(part_num):
        # print(len(topo_feats))
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(topo_feats)
        y_kmeans = kmeans.predict(topo_feats)
        cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
        topo_graphons = []
        topo_funcs = []
        for k, deg_graphs in enumerate(cluster_graphs):
            # print(np.array(topo_graphs)[deg_graphs])
            step_func, non_para_graphon = estimate_graphon(np.array(topo_graphs)[deg_graphs], method=args.method, args=args)
            topo_funcs.append(step_func)
            topo_graphons.append(non_para_graphon)
        max_clu_id = estimate_feasibility(topo_funcs,down_graphon) #return maximum alpha index
        topo_graphs = np.array(topo_graphs)[cluster_graphs[max_clu_id]]
        topo_feats = np.array(topo_feats)[cluster_graphs[max_clu_id]]
        print(cluster_graphs[max_clu_id])
choose_graphs(args)
