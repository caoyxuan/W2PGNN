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
parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--epoch', type=int,default=100,help='number of trails to fit the final graphon')
parser.add_argument('--learning_rate', type=float,default=0.05,help='learning_rate')
parser.add_argument('--weight_decay', type=float,default=1e-5,help='weight_decay')
parser.add_argument('--beta1', type=float,default=0.9,help='learning_rate')
parser.add_argument('--beta2', type=float,default=0.99,help='weight_decay')
parser.add_argument('--gpu', type=int,default=0,help='gpu')
parser.add_argument('--func', type=int, default="0")
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
parser.add_argument('--down_path', type=str, default="data/graphons/down/",
                    help='downstream data path')
args = parser.parse_args()
assert args.gpu is not None and torch.cuda.is_available()
print("Use GPU: {} for training".format(args.gpu))
pre_datas=[]
splits=['domain','integer','topo']
def mean_fit(pre_generators_norm):
    return np.mean(pre_generators_norm,axis = 0)
def alpha_fit(pre_generators_norm,cluster_num, down_generator):
    alpha_graph = nn.Linear(1, cluster_num).to(torch.device(args.gpu))
    torch.nn.init.constant_(alpha_graph.weight, 1)
    optimizer_alpha = torch.optim.Adam(
        alpha_graph.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    loss_2,loss_3=[],[]

    min_loss = 100
    trigger_time = 0
    patience = 3
    for i in range(args.epoch):
        if i % 20 == 0:
            trigger_time = 0
        # print("epoch", i)
        final_graphon = 0
        normalized_alpha = F.softmax(alpha_graph.weight, dim=0).to(torch.device(args.gpu))
        for j in range(cluster_num):
            final_graphon += normalized_alpha[j].to(torch.device(args.gpu)) * torch.tensor(pre_generators_norm[j]).to(
                torch.device(args.gpu))
        optimizer_alpha.zero_grad()
        loss2 = entropic_gw(final_graphon, torch.tensor(down_generator).to(torch.device(args.gpu)), device=args.gpu)
        loss3 = simulator.gw_distance(final_graphon.cpu().detach().numpy(), down_generator)
        loss_2.append(loss2.data.item())
        loss_3.append(loss3)
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
    print("final alpha:", normalized_alpha)
    return gw_dis
def load_basis(split,dataname,down_generator):
    pre_generators = []
    pre_splits = np.split(dataname,"_")
    if split == "topo":
        cluster_num = 5
    elif split == "domain":
        cluster_num = len(pre_splits)
    elif split == "integer":
        cluster_num = 1
    for i in range(cluster_num):
        if args.func:
            if split == "integer":
                load_path = "data/graphons/" + split + dataname + "/func.npy"
            elif split == "domain":  
                load_path = "data/graphons/" + pre_splits[i] + "/func.npy"
            else:
                load_path = "data/graphons/" + split + dataname + "/func"+str(
                    i) + ".npy"
            graphon = np.load(load_path)
            cur_size = graphon.shape[0]
            max_size = max(cur_size, max_size)
            print(max_size)
        else:
            if split == "integer":
                load_path = "data/graphons/" + split + dataname + "/graphon.npy"
            elif split=="domain":
                load_path = "data/graphons/" + split +pre_splits[i] + "/graphon.npy"
            else:
                load_path = "data/graphons/" + split + dataname + "/graphon"+ str(
                    i) + ".npy"
            graphon = np.load(load_path)
            max_size = graphon.shape[0]
        pre_generators.append(graphon)
    max_size = max(down_generator.shape[0], max_size)
    pre_generators_norm = []
    for i in range(cluster_num):
        pre_generator = cv2.resize(pre_generators[i], dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
        pre_generators_norm.append(pre_generator)
    down_generator = cv2.resize(down_generator, dsize=(max_size, max_size), interpolation=cv2.INTER_LINEAR)
    return pre_generators_norm, cluster_num, down_generator


# construct_basis(args)
# down_generator = estimate_generator_down(args)
down_path = args.down_path+args.down_data+"/graphons.npy"
down_generator = np.load(down_path)
alpha_min_dis = 1000
mean_min_dis = 1000
for k in range(len(splits)):
    split = splits[k]
    pre_generators, cluster_num,down_generator= load_basis(split,args.pre_data,down_generator)
    mean_graphon = mean_fit(pre_generators)
    mean_dis_gw = simulator.gw_distance(mean_graphon, down_generator)
    alpha_dis_gw=alpha_fit(pre_generators,cluster_num,down_generator)
    print("mean_feasibility_{}:{}".format(split,mean_dis_gw))
    print("alpha_feasibility_{}:{}".format(split,alpha_dis_gw))
    alpha_min_dis = min(alpha_min_dis,alpha_dis_gw)
    mean_min_dis = min(mean_min_dis,mean_dis_gw)
print("final_feasibility:{}".format(alpha_min_dis))
        

