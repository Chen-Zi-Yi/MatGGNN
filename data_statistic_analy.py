#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project    : MatGGNN 
@File       : data_statistic_analy.py
@IDE        : PyCharm 
@Author     : zychen@cnic.cn
@Date       : 2023/12/27 21:06 
@Description: 对mp_all, mp_stable,oqmd
"""
import json
import os
from pymatgen.core.structure import Structure
from fnmatch import fnmatch

from tqdm import tqdm

# structure.composition.to_data_dict['unit_cell_composition']
# {'Na': 24.0, 'Hf': 12.0, 'Si': 18.0, 'O': 72.0}
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

path = r'D:\CZY\OQMD_Mysql\oqmd_data'
list_ = os.listdir(path)
element_count = {}
atom_count = {}
for file in tqdm(list_, total=len(list_)):
    if fnmatch(file, '*.cif'):
        try:
            structure = Structure.from_file(os.path.join(path, file))
        except ValueError:
            continue
        composition = structure.composition.to_data_dict['unit_cell_composition']
        if len(composition) not in element_count.keys():
            element_count[len(composition)] = 1
        else:
            element_count[len(composition)] += 1
        for ele, number in composition.items():
            if ele not in atom_count:
                atom_count[ele] = number
            else:
                atom_count[ele] += number

print(element_count)
atom_count_sorted = {key: value for key, value in sorted(atom_count.items(), key=lambda item: item[1])}
print(atom_count_sorted)
record = {'element_count': element_count, 'atom_count': atom_count_sorted}

with open('oqmd_50w_formation_energy.json', 'w') as f:
    f.write(json.dumps(record, indent=2))
    f.close()
print('Over Save!')
