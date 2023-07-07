import torch.nn as nn
import torch
import argparse
import os
from utils.other_gw import entropic_gw
import numpy as np
import cv2
import pandas as pd
import torch.nn.functional as F
from utils import simulator
from tqdm import tqdm
from generator_down import estimate_generator_down
from construct_basis import construct_basis
from construct_pre_data import construct_pre_data
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--epoch', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--gpu', type=int,default=0,help='gpu')
parser.add_argument('--func', type=int, default="0")
parser.add_argument('--pre_data', type=str, default="zinc_standard_agent")
parser.add_argument('--down_data', type=str, default="bace")
parser.add_argument('--down_path', type=str, default="data/graphons/down/",
                    help='downstream data path')
parser.add_argument('--save_path', type=str, default="data/graphons/pre/",
                    help='downstream data path')
parser.add_argument('--load_path', type=str, default="data/graphons/pre/",
                    help='downstream data path')
parser.add_argument('--file_path', type=str, default="data/dataset/",
                    help='downstream data path')
parser.add_argument('--split_num', type=int, default=2, help='number of splits for pre-training datasets',
                    choices=[1,2, 3, 4])
parser.add_argument('--split_ids', type=list, default="12")
args = parser.parse_args()
assert args.gpu is not None and torch.cuda.is_available()
print("Use GPU: {} for training".format(args.gpu))
splits = ["integer","topo","domain"]
def mean_fit(pre_graphons_norm):
    return np.mean(pre_graphons_norm,axis = 0)
def alpha_fit(pre_graphons_norm,cluster_num,down_generator,lr,beta1,beta2,weight_decay,device,epochs,):
    alpha_graph = nn.Linear(1, cluster_num).to(torch.device(device))
    torch.nn.init.constant_(alpha_graph.weight, 1)
    optimizer_alpha = torch.optim.Adam(
        alpha_graph.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    loss_2,loss_3=[],[]
    min_loss = 100
    trigger_time = 0
    patience = 3
    for i in range(epochs):
        if i % 20 == 0:
            trigger_time = 0
        # print("epoch", i)
        final_graphon = 0
        normalized_alpha = F.softmax(alpha_graph.weight, dim=0).to(torch.device(device))
        for j in range(cluster_num):
            final_graphon += normalized_alpha[j].to(torch.device(device)) * torch.tensor(pre_graphons_norm[j]).to(
                torch.device(device))
        optimizer_alpha.zero_grad()
        loss2 = entropic_gw(final_graphon, torch.tensor(down_generator).to(torch.device(device)), device=device)
        # loss3 = simulator.gw_distance(final_graphon.cpu().detach().numpy(), down_generator)
        loss_2.append(loss2.data.item())
        # loss_3.append(loss3)
        loss2.backward()
        if min_loss > loss2.data.item():
            min_loss = min(min_loss, loss2.data.item())
            min_graphon = final_graphon
        if i > 30:
            if loss2 > min_loss:
                trigger_time += 1
                if trigger_time >= patience:
                    break
        optimizer_alpha.step()
    if min_graphon == None:
        gw_dis = simulator.gw_distance(final_graphon.cpu().detach().numpy(), down_generator)
    else:
        gw_dis = simulator.gw_distance(min_graphon.cpu().detach().numpy(), down_generator)
    print("final alpha:", normalized_alpha.cpu().detach().numpy())
    return gw_dis,normalized_alpha.cpu().detach().numpy()


def load_graphons(split,pre_splits,down_generator,dataset,args):
    pre_graphons = []
    if split == "topo":
        cluster_num = 5
    elif split == "domain":
        cluster_num = 2
    elif split == "integer":
        cluster_num = 1
    for i in range(cluster_num):
        if args.func:
            if split == "integer":
                load_path = args.load_path + split + "/"+dataset+ pre_splits + "/func.npy"
            elif split == "domain":
                load_path = args.load_path + split + "/"+dataset+ pre_splits[i] + "/func.npy"
            else:
                load_path = args.load_path + split +  "/"+dataset+ pre_splits + + "/func"+str(
                    i) + ".npy"
            graphon = np.load(load_path)
            cur_size = graphon.shape[0]
            max_size = max(cur_size, max_size)
            print(max_size)
        else:
            if split == "integer":
                load_path = args.load_path + split + "/"+dataset+ pre_splits + "/graphon.npy"
            elif split=="domain":
                load_path = args.load_path+ split +  "/"+dataset+ pre_splits[i]  + "/graphon.npy"
            else:
                load_path = args.load_path+ split +  "/"+dataset+ pre_splits + "/graphon"+ str(
                    i) + ".npy"
            graphon = np.load(load_path)
            max_size = graphon.shape[0]
        pre_graphons.append(graphon)
    max_size = max(down_generator.shape[0], max_size)
    pre_graphons_norm = []
    for i in range(cluster_num):
        pre_graphon = cv2.resize(pre_graphons[i], dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
        pre_graphons_norm.append(pre_graphon)
    print(max_size)
    print(down_generator.shape)
    down_generator = cv2.resize(down_generator, dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
    return pre_graphons_norm, cluster_num, down_generator
if __name__ == "__main__":
    construct_pre_data(args)
    construct_basis(args)
    down_generator = estimate_generator_down(args)
    down_path = args.down_path+args.down_data+"/graphons.npy"
    down_generator = np.load(down_path)
    # if args.pre_data == "zinc_standard_agent":
    pre_split=""
    for i in range(args.split_num):
        pre_split+=str(args.split_ids[i])
    alpha_min_dis = 1000
    mean_min_dis = 1000
    for k in range(len(splits)):
        split = splits[k]
        pre_graphons, cluster_num, down_generator = load_graphons(split,pre_split,down_generator,args.pre_data,args)
        mean_graphon = mean_fit(pre_graphons)
        mean_dis_gw = simulator.gw_distance(mean_graphon, down_generator)
        alpha_dis_gw = alpha_fit(pre_graphons,cluster_num,down_generator)
        print("mean_feasibility_{}:{}".format(split, mean_dis_gw))
        print("alpha_feasibility_{}:{}".format(split, alpha_dis_gw))
        alpha_min_dis = min(alpha_min_dis, alpha_dis_gw)
        mean_min_dis = min(mean_min_dis, mean_dis_gw)
    print("final_feasibility:{}".format(alpha_min_dis))

