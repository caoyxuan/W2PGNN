import torch.nn as nn
import torch
from utils.other_gw import entropic_gw
import numpy as np
import torch.nn.functional as F
from utils import simulator
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
