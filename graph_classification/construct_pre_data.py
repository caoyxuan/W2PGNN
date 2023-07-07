import argparse
from utils.loader import MoleculeDataset
import itertools
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str, default="zinc_standard_agent",
                    help='pretrain data')
parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")

parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
args = parser.parse_args()
def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)
    scaffold_sets = rng.permutation(list(scaffolds.values()))
    len_per_env = len(scaffolds)//5
    envs=[]

    for i in tqdm(range(5)):
        if len_per_env*(i+1) < len(scaffolds):
            envs.append(list(itertools.chain(*(scaffold_sets[len_per_env*i:int(len_per_env*(i+0.5))].tolist()))))
        else:
            envs.append(list(itertools.chain(*(scaffold_sets[len_per_env*i:]))))
    return envs

def construct_pre_data(args):
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    if args.dataset == "zinc_standard_agent":
        data_pre = "data/"
        data_pre = "/data/chem/"
        save_path = "data/demo2/"
        dataset = MoleculeDataset(data_pre + "dataset/" + args.dataset, dataset=args.dataset)[:200]
        print(dataset)
        smiles_list = pd.read_csv(data_pre + 'dataset/' + args.dataset + '/processed/smiles.csv', header=None)[
            0].tolist()
        envs = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        for i in tqdm(range(len(envs))):
            np.save(save_path + "dataset/"+args.dataset+"/split"+str(i)+".npy",np.array(envs[i]))
            torch.save(dataset[np.array(envs[i]).tolist()],save_path + "dataset/"+args.dataset+"/domain"+str(i)+".pt")
if __name__ == "__main__":
    construct_pre_data(args)
# datas = torch.load(save_path + "dataset/"+args.dataset+"/split"+str(i)+".pt")
