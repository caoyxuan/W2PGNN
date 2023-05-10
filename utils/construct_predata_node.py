import dgl
import torch
import random
def merge_data(sam_num):
    data_name = ['livejournal', 'facebook', 'imdb', 'dblp-net', 'dblp-snap', 'academia']
    idx_list = [i for i in range(6)]
    c = dgl.data.utils.load_graphs("data/small.bin")[0]
    sam_list = random.sample(idx_list, sam_num)
    graph_list = [c[i] for i in sam_list]

    a = dgl.data.utils.load_graphs("data/small.bin")
    node_nb_list = []
    a[0].clear()

    for g in graph_list:
        num_nodes = len(g.nodes())
        a[0].append(g)
        node_nb_list.append(num_nodes)

    a[1]["graph_sizes"] = torch.Tensor(node_nb_list)
    graph_size = a[1]
    saved_filename = "data/pre_{}_{}.bin".format(data_name[sam_list[0]], data_name[sam_list[1]])
    dgl.data.utils.save_graphs(filename=saved_filename, g_list=a[0],
                               labels=graph_size)
  
merge_data(2)
merge_data(3)
