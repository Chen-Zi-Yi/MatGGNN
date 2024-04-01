#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project    : MatGGNN 
@File       : random_search.py
@IDE        : PyCharm 
@Author     : zychen@cnic.cn
@Date       : 2024/2/27 10:50 
@Description: 
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from ase.calculators.emt import EMT
from ase.constraints import ExpCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymongo import MongoClient
from scipy.stats import rankdata
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils import dense_to_sparse, add_self_loops
from tqdm import tqdm

from matdeeplearn import models, process, evaluate, write_results, training

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Config on graph generation from crystal
graph_processing = {'SM_descriptor': 'False',
                    'SOAP_descriptor': 'False',
                    'SOAP_lmax': 4,
                    'SOAP_nmax': 6,
                    'SOAP_rcut': 8.0,
                    'SOAP_sigma': 0.3,
                    'data_format': 'cif',
                    'data_path': 'dummy_data',
                    'dataset_type': 'inmemory',
                    'dictionary_path': 'atom_dict.json',
                    'dictionary_source': 'default',
                    'edge_features': 'True',
                    'graph_edge_length': 50,
                    'graph_max_neighbors': 12,
                    'graph_max_radius': 8.0,
                    'target_path': 'targets.csv',
                    'verbose': 'True',
                    'voronoi': 'False'}


def optimizer(atoms, savefile='opt.traj', fmax=0.05):
    atoms.calc = EMT()
    relax = ExpCellFilter(atoms)
    BFGS(relax, trajectory=savefile).run(fmax=fmax)


def traj2xyz(traj_file, xyz_file):
    traj = Trajectory(traj_file, mode='r')
    for i in traj:
        i.write(xyz_file, append=True)


# Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


# Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


# Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


# Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    # mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)
    feature_min = torch.tensor(0.0)
    feature_max = torch.tensor(7.9994)
    for data in dataset:
        data.edge_descriptor[descriptor_label] = (data.edge_descriptor[descriptor_label] - feature_min) / (feature_max - feature_min)


# Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


def process_ase_objects(atoms_objects, processing_args, edge_feature_min, edge_feature_max):
    # Load dictionary 100 dims
    atom_dictionary = get_dictionary(
        os.path.join(
            'matdeeplearn/process',
            "dictionary_default.json",
        )
    )

    # Load targets
    target_data = [[idx, 0] for idx, st in enumerate(atoms_objects)]

    # Process structure files and create structure graphs
    data_list = []
    for index in tqdm(range(0, len(target_data)), total=len(target_data)):
        structure_id = target_data[index][0]
        data = Data()
        data.ase = atoms_objects[index]
        ase_crystal = atoms_objects[index]
        # Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        # Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        # Adding self-loops
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
        )
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        distance_matrix_mask = (
                distance_matrix_trimmed.fill_diagonal_(1) != 0
        ).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        target = target_data[index][1:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y

        # pos = torch.Tensor(ase_crystal.get_positions())
        # data.pos = pos
        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data.structure_id = [[structure_id] * len(data.y)]
        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    # Generate node features
    for index in range(0, len(data_list)):
        atom_fea = np.vstack(
            [
                atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))
            ]
        ).astype(float)
        data_list[index].x = torch.Tensor(atom_fea)

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    ##Generate edge features
    ##Distance descriptor using a Gaussian basis
    distance_gaussian = GaussianSmearing(
        0, 1, processing_args["graph_edge_length"], 0.2
    )
    # print(GetRanges(data_list, 'distance'))
    NormalizeEdge(data_list, "distance")
    # print(GetRanges(data_list, 'distance'))
    for index in range(0, len(data_list)):
        data_list[index].edge_attr = distance_gaussian(
            data_list[index].edge_descriptor["distance"]
        )

    Cleanup(data_list, ["ase", "edge_descriptor"])

    data, slices = InMemoryDataset.collate(data_list)
    # torch.save((data, slices), os.path.join("data.pt"))
    return data, slices, data_list

def queryStructure():
    client = MongoClient('localhost', 27017)
    db = client['ICSD']
    conn = db['Structure']
    querySet = conn.find({'ElementCount': 2})
    # included = ['B', 'Al', 'Ga', 'In', 'N', 'P', 'As', 'Sb', 'Bi']
    # for item in querySet:
    #     need = True
    #     for com in item['Composition']:
    #         if com['AtomicSymbol'] not in included:
    #             need = False
    #             break
    #     if need:
    #         print(item['SimplestFormula'])
    structure = []
    for item in querySet:
        a, b, c, alpha, beta, gamma = item['CellParameters']['a'], item['CellParameters']['b'], item['CellParameters'][
            'c'], item['CellParameters']['alpha'], item['CellParameters']['beta'], item['CellParameters']['gamma']
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        position = []
        atoms = []
        for site in item['ReducedSites']:
            atoms.append(site['AtomicSymbol'])
            position.append(site['Position'])
        structure.append(Structure(lattice, atoms, position))
    print('len(structure) = ', len(structure))
    return structure


def return_megnet_model(dummy_dataset):
    model = models.MEGNet(dummy_dataset, dim1=170, dim2=170, dim3=190, gnn_count=7, lr=0.0001, pool='set2set',
                          post_fc_count=2, pre_fc_count=1, gc_count=4, gc_fc_count=1, dropout_rate=0)
    # weights = torch.load('models/train_oqmd50w_megnet_1000e_pretrain_imporve.pth', map_location=device)
    # weights = torch.load('models/train_oqmd50w_megnet_1000e_pretrain_1 copy.pth', map_location=device)
    weights = torch.load('models/train_oqmd50w_megnet_1000e_pretrain_2.pth', map_location=device)
    # weights = torch.load('models/train_oqmd50w_megnet_1000e_pretrain_imporve.pth', map_location=device)
    model.load_state_dict(weights['model_state_dict'])
    return model


if __name__ == '__main__':
    print('start')
    dummy_dataset = process.get_dataset('data/materialsproject/mp_all', 0, False, graph_processing)
    # # atoms = Atoms(['Al', 'N'],
    # #               cell=[3.956, 3.956, 3.956, 90, 90, 90],
    # #               positions=[[0, 0, 0], [1.978, 1.978, 1]],
    # #               pbc=True)
    # # traj_file = 'yyy.traj'
    # # xyz_file = 'yyy.xyz'
    # # optimizer(atoms, savefile=traj_file, fmax=0.05)
    # # traj2xyz(traj_file, xyz_file)
    # # print(1)
    energy_model = return_megnet_model(dummy_dataset)
    energy_model = energy_model.to(device)
    energy_model = energy_model.eval()
    # # model_summary(energy_model)
    lattice = Lattice.from_parameters(3.14,	3.14,	12.3,	90,	90,	120)  # -3.24448848
    structure1 = Structure(lattice=lattice, species=['Al', 'N'],
                           coords=[[0.3333333333333333,  0.6666666666666666,  0.25], [0.3333333333333333,  0.6666666666666666, 0.628] ])
    # structure1 = Structure.from_file(r'E:\ICSD-origin\download_ICSD\NumOfElements=2\YourCustomFileName_CollCode26875.cif')
    # structure1 = Structure.from_file(r'E:\ICSD-origin\download_ICSD\NumOfElements=3\cubic_bodyCentered\YourCustomFileName_CollCode167355.cif')
    # structure1 = Structure.from_file(r'E:\ICSD-origin\download_ICSD\NumOfElements=4\cubic\YourCustomFileName_CollCode4030.cif')
    # structure1 = Structure.from_file(r'data/oqmd_50w/100095.cif')
    structures = [structure1]
    # # structures = queryStructure()
    atoms_objects_list = [AAA.get_atoms(a) for a in structures]
    _,_,dataset = process_ase_objects(atoms_objects_list, graph_processing, edge_feature_min=0, edge_feature_max=8)

    # process.get_dataset(data_path='data/oqmd_data/test_data', target_index=0, processing_args=graph_processing)
    # transforms = GetY(index=0)
    # dataset = StructureDataset(
    #     'data/oqmd_data/test_data',
    #     'processed',
    #     transforms,
    # )
    # print(1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    _, energy_out = training.evaluate(loader, energy_model, 'l1_loss', device, out=True)
    energies = energy_out[:, -1].astype('float')
    print(energies)
    # print(len(energies.tolist()))
    # save = []
    # for index, struct in enumerate(structures):
    #     item = {
    #         'formula': struct.formula,
    #         'lattice': struct.lattice.matrix.tolist(),
    #         'composition': [{'symbol': site.label, 'pos': site.frac_coords.tolist()} for site in struct.sites],
    #         'predicted energy': energies[index]
    #     }
    #     save.append(item)
    # save = {'data': save}
    # json_data = json.dumps(save, indent=2)
    # with open('icsd_pred.json', 'w') as fp:
    #     fp.write(json_data)
    #     fp.close()

    # f = open('search_result/AlN_01.txt', 'w', encoding='utf8')
    # f.write('AlN\n')
    # f.write('parameters: 3.956, 3.956, 3.956, 90, 90, 90  \n')
    # f.write('   position(In)  position(Sb)  energy\n')
    # attempt = 0
    # position_list = list()
    # energy_list = list()
    # while attempt < 100:
    #     print('attempt = ', attempt)
    #     positions = np.random.randint(1, 100, (100, 3)) / 100
    #     positions = positions.tolist()
    #     structures = []
    #     for pos in positions:
    #         structure = Structure(lattice=lattice, species=['Al', 'N'],
    #                               coords=[[0, 0, 0], pos])
    #         structures.append(structure)
    #     atoms_objects_list = [AAA.get_atoms(a) for a in structures]
    #     data_, slices_, list_ = process_ase_objects(atoms_objects_list, graph_processing, edge_feature_min=0,
    #                                                 edge_feature_max=8)
    #     loader = DataLoader(list_, batch_size=100, shuffle=False)
    #     _, energy_out = training.evaluate(loader, energy_model, 'l1_loss', device, out=True)
    #     energies = energy_out[:, -1].astype('float')
    #
    #     # 排序
    #     new_position_list = position_list + positions
    #     new_energy_list = energy_list + energies.tolist()
    #     rank = np.argsort(new_energy_list)
    #     position_set = set()
    #     position_list = []
    #     energy_list = []
    #     for i in rank:
    #         a = len(position_set)
    #         position_set.add(tuple(new_position_list[i]))
    #         if len(position_set) > a:
    #             position_list.append(new_position_list[i])
    #             energy_list.append(new_energy_list[i])
    #         if len(position_set) >= 100:
    #             break
    #     assert len(position_list) == len(energy_list)
    #     f.write(f'attempt = {attempt}\n')
    #     for pos, energy in zip(position_list, energy_list):
    #         f.write(f'  [0, 0, 0]  {pos}   {energy}\n')
    #     attempt += 1
    #
    # f.close()


# Al As
# -9.0302
def test():
    data_path = ''
    dataset = process.get_dataset(data_path, 0, False)
    train_sampler = None
    train_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )
    rank = 'cuda'
    print('Load pretrained model')
    saved = torch.load(
        '/public/home/liufang395/czy/MatGGNN/jobs/train_oqmdadv_megnet_1000e_pretrain/train_oqmdadv_megnet_1000e_pretrain.pth',
        map_location=rank)
    model = models.MEGNet(dataset, dim1=170, dim2=170, dim3=190, gnn_count=7, lr=0.0001, pool='set2set',
                          post_fc_count=2, pre_fc_count=1, gc_count=4, gc_fc_count=1, dropout_rate=0.0)
    model.load_state_dict(saved['model_state_dict'])
    model = model.to(rank)

    optimizer = getattr(torch.optim, 'AdamW')(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler_args = {
        'factor': 0.8,
        'min_lr': 1e-06,
        'mode': 'min',
        'patience': 10,
        'threshold': 0.0002
    }
    scheduler = getattr(torch.optim.lr_scheduler, 'ReduceLROnPlateau')(optimizer, **scheduler_args)
    train_error, train_out = evaluate(
        train_loader, model, 'l1_loss', rank, out=True
    )
    print("Train Error: {:.5f}".format(train_error))
    write_results(
        train_out, "train_outputs.csv"
    )
