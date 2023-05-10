from tqdm import tqdm
import math
import networkx as nx
import numpy as np
import torch_geometric
def z_norm(sequence):
    return (np.array(sequence) - np.mean(sequence)) / np.std(sequence)
def cal_topo(nxgraph):
    density = nx.density(nxgraph)
    degrees = nx.degree_histogram(nxgraph)
    avg_degree = float(sum(degrees[i] * i for i in range(len(degrees))) / (nxgraph.number_of_nodes()))
    std = math.sqrt(
        sum(math.pow(i - avg_degree, 2) * degrees[i] for i in range(len(degrees))) / (nxgraph.number_of_nodes()))
    closeness_centrality = sum(nx.closeness_centrality(nxgraph).values()) / len(nxgraph.node)
    degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(nxgraph)
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(nxgraph)
    transitivity = nx.transitivity(nxgraph)
    avg_clu_co = nx.average_clustering(nxgraph)
    topo_vec = [ avg_degree,std,density, closeness_centrality, degree_pearson_correlation_coefficient,
                degree_assortativity_coefficient, transitivity, avg_clu_co]
    return topo_vec
def cal_topo_graphs(graphs):
    topos = []
    for i in tqdm(range(len(graphs))):
        if isinstance(graphs[0],np.ndarray):
            nxgraph = nx.from_numpy_array(graphs[i])
        elif isinstance(graphs[0],torch_geometric.data.Data ):
            nxgraph = torch_geometric.utils.to_networkx(graphs[i])
        if len(nxgraph.edges())==0:
            continue
        topo_vec1 = cal_topo(nxgraph)
        topos.append(topo_vec1)
    return topos
