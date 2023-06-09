import torch
import deepchem as dc
import json
from torch_geometric.data import Data, InMemoryDataset
import shutil
import os
from deepchem.data import NumpyDataset, DiskDataset

class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def load_dataset(adj,smiles,Y,fpf,fpc):
    
    d = []
    smiles_list = []
    graphconvfeaturizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    conv_mol_featurizer = dc.feat.ConvMolFeaturizer()
    for i in range(len(Y)):
        convmol = conv_mol_featurizer.featurize(smiles[i])[0]
        mol_graph = graphconvfeaturizer.featurize(smiles[i])[0]
        x = torch.Tensor(convmol.get_atom_features()).cuda()
        edge_index = torch.LongTensor(mol_graph.edge_index).cuda()
        edge_attr = torch.LongTensor(mol_graph.edge_features).cuda()
        data = Data(x=x, adj = adj[i], edge_index=edge_index, edge_attr=edge_attr, y=Y[i], fpf=fpf[i], fpc=fpc[i])
        d.append(data)
        smiles_list.append(smiles)
    set_ = TestDataset(d)
    return set_, smiles_list

class KFoldDataset(object):
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def create_dataset(self, dc_tvset,folds,splitter,seed):
        os.makedirs(f'{self.run_dir}/.tmpfolder/',exist_ok=True)
        directories = []
        for fold in range(folds):
            directories.extend([f'{self.run_dir}/.tmpfolder/{fold}-train',f'{self.run_dir}/.tmpfolder/{fold}-test'])
        for fold, (dc_trainset, dc_validset) in enumerate(splitter.k_fold_split(dc_tvset,k=folds,seed=seed,directories=directories)):
            pass

    def get_dataset(self, fold):
        trainset = NumpyDataset.from_DiskDataset(DiskDataset(data_dir=f'{self.run_dir}/.tmpfolder/{fold}-train'))
        testset = NumpyDataset.from_DiskDataset(DiskDataset(data_dir=f'{self.run_dir}/.tmpfolder/{fold}-test'))

        return trainset, testset

    def delete(self):
        shutil.rmtree(f'{self.run_dir}/.tmpfolder/')
