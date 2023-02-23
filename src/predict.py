import numpy as np
import pandas as pd
import sys
import pickle as pkl
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pdb
import os
from threading import Thread, Lock
from rdkit.Chem import AllChem
import pgl

import paddle as pdl
from pgl.utils.data import Dataset
from pgl.utils.data.dataloader import Dataloader

pdl.seed(1024)
np.random.seed(1024)
random.seed(1024)
mutex = Lock()

from main import ADMET
from main import MyDataset

current_path = os.path.dirname(__file__)
current_path = os.path.dirname(current_path)


def collate_fn(data_batch):
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]
    
    atom_bond_graph_list = []
    bond_angle_graph_list = []
    smiles_list = []
    label_list = []

    for data_item in data_batch:
        graph = data_item['graph']
        ab_g = pgl.Graph(
                num_nodes=len(graph[atom_names[0]]),
                edges=graph['edges'],
                node_feat={name: graph[name].reshape([-1, 1]) for name in atom_names},
                edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_names + bond_float_names})
        ba_g = pgl.Graph(
                num_nodes=len(graph['edges']),
                edges=graph['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_angle_float_names})
        atom_bond_graph_list.append(ab_g)
        bond_angle_graph_list.append(ba_g)
        smiles_list.append(data_item['smiles'])
        label_list.append(data_item['label'])

    atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
    bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
    # TODO: reshape due to pgl limitations on the shape
    def _flat_shapes(d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    _flat_shapes(atom_bond_graph.node_feat)
    _flat_shapes(atom_bond_graph.edge_feat)
    _flat_shapes(bond_angle_graph.node_feat)
    _flat_shapes(bond_angle_graph.edge_feat)

    return atom_bond_graph, bond_angle_graph, np.array(label_list, dtype=np.float32), smiles_list

def evaluate(model, data_loader):
    model.eval()
    label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    label_smiles = pd.DataFrame(data=None, columns=['smile'])
    for (atom_bond_graph, bond_angle_graph, label_true_batch, smiles_batch) in data_loader:
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
        for i in range(len(smiles_batch)):
            label_smiles = label_smiles.append({'smile': smiles_batch[i]}, ignore_index=True)

    label_predict = label_predict.cpu().numpy()
    label_smiles.insert(loc=len(label_smiles.columns), column='predict', value=label_predict)
    return label_smiles

def get_data_loader(batch_size):
    data_list_train = pkl.load(open(current_path+'./data/lipo/intermediate/data_list_train.pkl', 'rb'))
    data_loader_train = Dataloader(MyDataset(data_list_train), batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    return data_loader_train

def predict(model_version):
    data_loader_predict = get_data_loader(batch_size=256)
    model = ADMET()
    model.set_state_dict(pdl.load(current_path+"./weight/lipo/" + model_version + ".pkl"))
    metric_train = evaluate(model, data_loader_predict)
    metric_train.to_csv( "a.csv")
    
if __name__ == '__main__':
    predict('1')
    print("All is well!")