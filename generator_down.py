from utils.load_utils import MoleculeDataset
import torch_geometric
from utils.learner import estimate_graphon
from tqdm import tqdm
import os
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="bace",
                    help='pretrain data')
parser.add_argument('--file_path', type=str, default="data/dataset/", help="Path to save raw data.")
parser.add_argument('--downgraph_path', type=str, default=None, help="Path to save down_graphs.")
parser.add_argument('--save_path', type=str, default="data/demo/graphons/", help="Path to save graphons.")
parser.add_argument('--method', type=str, default="LP", help="method to estimate graphon.")
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
# parser.add_argument('--method', type=str, default=0,
#                     help='which gpu to use if any (default: 0)')
args = parser.parse_args()
def estimate_generator_down(args,datas=None,down_path=None):
    if down_path is None:
        file_path = args.file_path
        file_path = "/data/srtpgroup/chem/dataset/"
        dataname = args.dataset
        save_path = args.save_path
        method = args.method
        dataset = MoleculeDataset(file_path + dataname, dataset=dataname)
        datas=dataset
        torch.save(datas,"/home/caoyuxuan/kdd/graph_classification/data/demo/dataset/down/"+args.dataset+".pt")
    else:
        datas = torch.load(down_path)
    graphons, stepfuncs = [], []
    nxgraphs = []
    for j in tqdm(range(len(datas))):
        matrix_graph = torch_geometric.utils.to_scipy_sparse_matrix(datas[j].edge_index,
                                                                    num_nodes=datas[j].num_nodes).toarray()
        nxgraphs.append(matrix_graph)
    step_func, non_para_graphon = estimate_graphon(nxgraphs , method=args.method, args=args)
    # graphons.append(non_para_graphon)
    # stepfuncs.append(step_func)
    if not os.path.exists(args.save_path + "down/" + args.down_data):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(args.save_path + "down/" + args.down_data)
    np.save(args.save_path + "down/" + args.down_data + "/graphons.npy", non_para_graphon)
    np.save(args.save_path+ "down/" + args.down_data + "/func.npy", step_func)
    return step_func,non_para_graphon
if __name__ == "__main__":
    estimate_generator_down(args)