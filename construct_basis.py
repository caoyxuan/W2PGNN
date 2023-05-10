import torch_geometric
from utils.learner import estimate_graphon
from utils.load_utils import MoleculeDataset
import argparse
import os
from cal_topo_statistcs import cal_topo_graphs
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import torch
parser = argparse.ArgumentParser(description='')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
parser.add_argument('--pre_data', type=str, default="zinc_standard_agent",
                    help='pretrain data')
parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
parser.add_argument('--file_path', type=str, default="data/dataset/", help="Path to save raw data.")
parser.add_argument('--save_path', type=str, default="data/graphons/pre/", help="Path to save graphons.")
parser.add_argument('--load_path', type=str, default="data/dataset/", help="Path to save split data.")
parser.add_argument('--method', type=str, default="LP", help="method to estimate graphon.")
parser.add_argument('--split_num', type=int, default=2, help='number of splits for pre-training datasets',
                    choices=[2, 3, 4])
parser.add_argument('--split_ids', type=list, default="12")
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
args = parser.parse_args()

def construct_basis(args):
    dataname = args.pre_data
    file_path = args.file_path
    file_path = "/data/srtpgroup/chem/dataset/"
    save_path = args.save_path
    load_path = args.load_path
    load_path = "/data/srtpgroup/chem/dataset/"
    method = args.method
    predata_splits = ""
    if args.pre_data == "zinc_standard_agent":
        dataset = MoleculeDataset(file_path + dataname, dataset=dataname)
        for i in range(args.split_num):
            predata_splits += args.split_ids[i]
        graph_ids = []
        for i in range(args.split_num):
            graph_ids+=np.load(load_path+dataname+"/split"+args.split_ids[i]+".npy").tolist()
        datas = dataset[graph_ids]
    else:
        datas = torch.load(args.file_path+args.dataname)
    graphs = []
    for j in tqdm(range(len(datas))):
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,num_nodes=datas[j].num_nodes).toarray()
        graphs.append(matrix_graph)
    splits = ["integer","topo","domain"]
    for split in (splits):
        if split=="integer":
            if not os.path.exists(save_path + "integer/" + dataname+ predata_splits):
                os.makedirs(save_path + "integer/" + dataname+ predata_splits)
            step_func,non_para_graphon  =estimate_graphon(np.array(graphs), method=method, args=args)
            np.save(save_path + "integer/" + dataname+ predata_splits + "/graphon.npy", non_para_graphon)
            np.save(save_path + "integer/" + dataname+ predata_splits + "/func.npy", step_func)
        elif split=="domain":
            for pre in predata_splits:
                if not os.path.exists(save_path + "domain/" + dataname + pre):
                    os.makedirs(save_path + "domain/" + dataname + pre)
                graph_ids=np.load(load_path+dataname+"/split"+pre+".npy").tolist()
                datas = dataset[graph_ids]
                graphs = []
                for j in range(len(datas)):
                    matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                                num_nodes=datas[j].num_nodes).toarray()
                    graphs.append(matrix_graph)
                step_func, non_para_graphon = estimate_graphon(np.array(graphs), method=method, args=args)
                np.save(save_path + "domain/" + dataname + pre + "/graphon.npy", non_para_graphon)
                np.save(save_path + "domain/" + dataname + pre + "/func.npy", step_func)
        elif split=="topo":
            if not os.path.exists(save_path + "topo/" + dataname+ predata_splits):
                os.makedirs(save_path + "topo/" + dataname+ predata_splits)
            topo_feats = cal_topo_graphs(graphs)
            print(topo_feats)
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(topo_feats)
            y_kmeans = kmeans.predict(topo_feats)
            cluster_graphs = [np.where(y_kmeans == i)[0].tolist() for i in range(5)]
            np.save(save_path + "topo/" + dataname + predata_splits + "/cluster_log.npy", cluster_graphs)
            for k, deg_graphs in enumerate(cluster_graphs):
                step_func, non_para_graphon = estimate_graphon(np.array(graphs)[deg_graphs], method=method, args=args)
                np.save(save_path + "topo/" + dataname + predata_splits + "/graphon" + str(k) + ".npy", non_para_graphon)
                np.save(save_path + "topo/" + dataname + predata_splits + "/func" + str(k) + ".npy", step_func)
            print("Kmeans topo Done!")
if __name__ == "__main__":
    construct_basis(args)