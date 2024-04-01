#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project    : MatGGNN 
@File       : process.py
@IDE        : PyCharm 
@Author     : zychen@cnic.cn
@Date       : 2023/11/7 16:25 
@Description: 
"""
import glob
import os
import sys

import ase
import ase.io
import numpy as np
import torch
import torch.nn.functional as F
from matminer.featurizers.composition import ElementProperty
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from scipy.stats import rankdata
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import dense_to_sparse, add_self_loops, degree

ep = ElementProperty.from_preset('magpie')


def split_data(
        dataset,
        train_ratio,
        val_ratio,
        test_ratio,
        seed=np.random.randint(1, int(1e6)),
        save=False
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")


def get_dataset(data_path, target_index, reprocess=False, processing_args=None):
    if processing_args is None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    # ?
    transforms = GetY(index=target_index)

    if not os.path.exists(data_path):
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess:
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")):
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")):
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "data.pt")):
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")):
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )
    return dataset


# Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


# Dataset class from pytorch/pytorch geometric
class StructureDataset_large(Dataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data


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


def NormalizeEdge(dataset, descriptor_label, feature_min=None, feature_max=None):
    if feature_min is None or feature_max is None:
        mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (data.edge_descriptor[descriptor_label] - feature_min) / (feature_max - feature_min)


class GaussianSmearing(torch.nn.Module):
    # 高斯模糊，用户平滑图像
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


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


def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


def distance_threshold(matrix, radius, neighbors, reverse=False):
    # 将距离 大于 半径的边 掩码
    mask = matrix > radius
    distance_matrix_constraint = np.ma.array(matrix, mask=mask)

    # 将距离排序
    if reverse:
        distance_matrix_constraint = rankdata(distance_matrix_constraint * -1, method='ordinal', axis=1)
    else:
        distance_matrix_constraint = rankdata(distance_matrix_constraint, method='ordinal', axis=1)

    # 掩码位置 置0
    distance_matrix_constraint = np.nan_to_num(np.where(mask, np.nan, distance_matrix_constraint))

    # 邻居距离rank超过最大数量的 置0
    distance_matrix_constraint[distance_matrix_constraint > neighbors + 1] = 0

    # 非0位置 设为 edge distance， 其他为0
    distance_matrix_constraint = np.where(
        distance_matrix_constraint == 0, distance_matrix_constraint, matrix
    )
    return distance_matrix_constraint


def process_data(file_path, processed_path, args):
    # ase_atoms = Atoms(numbers=list(struct.atomic_numbers), positions=struct.cart_coords, cell=struct.lattice.matrix, pbc=struct.pbc)
    data_list = []
    # 文件数据加载/数据库读取 遍历
    files = os.listdir(file_path)
    length = []
    elements = []
    for i in range(len(files)):
        data = Data()
        crystal = ase.io.read(os.path.join(file_path, files[i]))  # ase 模块读取文件
        data.ase = crystal
        if i == 0:
            length = [len(crystal)]
            elements = [list(set(crystal.get_chemical_symbols()))]
        else:
            length.append(len(crystal))
            elements.append(list(set(crystal.get_chemical_symbols())))

        # distance matrix
        distance_matrix = crystal.get_all_distances(mic=True)  # return shape [len(crystal), len(crystal) ]

        # 数量和距离约束 （只在一个单元晶胞内筛选）
        distance_matrix_constraint = distance_threshold(distance_matrix, args['graph_max_radius'],
                                                        args['graph_max_neighbors'])

        # 稀疏矩阵
        distance_matrix_constraint = torch.Tensor(distance_matrix_constraint)
        out = dense_to_sparse(distance_matrix_constraint)
        edge_index = out[0]
        edge_weight = out[1]

        # 自循环图, 权重 为 0
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(crystal), fill_value=0)
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        distance_matrix_mask = (distance_matrix_constraint.fill_diagonal_(1) != 0).int()

        data.edge_descriptor = {'distance': edge_weight, 'mask': distance_matrix_mask}

        # 结构的composition feature
        composition = torch.tensor(ep.featurize(AAA.get_structure(crystal).composition))
        data.composition = composition

        z = torch.LongTensor(crystal.get_atomic_numbers())
        data.z = z

        # 状态特征位
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        # data.structure
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print("Data processed: ", i + 1, " out of ", len(files))

        data_list.append(data)

    atoms_max_num = max(length)
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)
    if args["verbose"]:
        print(
            "Max structure size: ",
            atoms_max_num,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)

    # crystal_length = len()

    # Generates one-hot node features
    lb = LabelBinarizer()
    lb.fit(species)
    for i in range(0, len(data_list)):
        data_list[i].x = torch.Tensor(lb.transform(data_list[i].ase.get_chenical_symbols()))

    # Adds node degree to node features (appears to improve performance)
    for i in range(0, len(data_list)):
        data_list[i] = OneHotDegree(data_list[i], args['graph_max_neighbors'] + 1)

    # makes SOAP and SM features from dscribe
    if args["SOAP_descriptor"]:
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SOAP

        make_feature_SOAP = SOAP(
            species=species,
            r_cut=args["SOAP_rcut"],
            n_max=args["SOAP_nmax"],
            l_max=args["SOAP_lmax"],
            sigma=args["SOAP_sigma"],
            periodic=periodicity,
            sparse=False,
            average="inner",
            rbf="gto",
        )
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = torch.Tensor(features_SOAP)
            if args["verbose"] and index % 500 == 0:
                if index == 0:
                    print(
                        "SOAP length: ",
                        features_SOAP.shape,
                    )
                print("SOAP descriptor processed: ", index)

    elif args["SM_descriptor"]:
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SineMatrix, CoulombMatrix
        """
        SineMatrix 和 CoulombMatrix 是一些用于描述分子结构的特征表示方法，常用于机器学习模型中的分子性质预测或化合物筛选。

        SineMatrix（正弦矩阵）： SineMatrix 是一种通过计算分子中原子之间的相互作用的正弦矩阵来描述分子结构的方法。对于分子中的每一对原子，
        SineMatrix 使用正弦函数来编码它们之间的距离和角度信息。这种表示能够捕捉分子中原子之间的几何结构和相互作用。 【表现不好】

        CoulombMatrix（库仑矩阵）： CoulombMatrix 是一种基于物理学原理的分子表示方法，通过考虑分子中每对原子之间的库仑相互作用来编码分子结构。
        库仑相互作用是电荷之间的相互作用，CoulombMatrix 中的元素是原子间的库仑排斥或引力。这种表示有助于捕捉分子的电荷分布和原子之间的相互作用。

        这两种特征表示方法都是为了将分子的结构信息转化为机器学习模型能够处理的数值表示形式。它们在分子描述和分子性质预测等任务中都有广泛的应用。
        选择使用哪种方法通常取决于具体的问题和数据集。
        """
        if periodicity:
            make_feature_SM = SineMatrix(
                n_atoms_max=atoms_max_num,
                permutation="eigenspectrum",
                sparse=False
            )
        else:
            make_feature_SM = CoulombMatrix(
                n_atoms_max=atoms_max_num,
                permutation="eigenspectrum",
                sparse=False
            )

        for index in range(0, len(data_list)):
            features_SM = make_feature_SM.create(data_list[index].ase)
            data_list[index].extra_features_SM = torch.Tensor(features_SM)
            if args["verbose"] and index % 500 == 0:
                if index == 0:
                    print(
                        "SM length: ",
                        features_SM.shape,
                    )
                print("SM descriptor processed: ", index)

    # Generate edge features
    if args['edge_features']:
        distance_gaussian = GaussianSmearing(0, 1, args['graph_edge_length'], 0.2)
        NormalizeEdge(data_list, 'distance') # 归一化操作

        for i in range(0, len(data_list)):
            data_list[i].edge_attr = distance_gaussian(data_list[i].edge_descriptor['distance'])
            if args['verbose'] and ((i + 1) % 500 == 0 or (i + 1) == len(data_list)):
                print("Edge processed: ", i + 1, "out of", len(data_list))

    Cleanup(data_list, ['ase', 'edge_descriptor'])

    if args['dataset_type'] == 'inmemory':
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(processed_path, 'data.pt'))
    elif args['dataset_type'] == 'large':
        for i in range(len(data_list)):
            torch.save(data_list[i], os.path.join(processed_path, 'data_{}.pt'.format(i)))


class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data