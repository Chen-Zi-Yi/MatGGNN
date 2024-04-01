import argparse
import csv
import os

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter

from darwin.optimize import search_and_optimize
from darwin.utils import GASearchParams, get_icsd_compounds


def config_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=2000)
    # parser.add_argument('--output', type=str, default='interpretability_compositions.pkl')
    parser.add_argument('--output', type=str, default='energy.csv')
    parser.add_argument('--generational', type=str, default='generational_outcomes.pkl')

    args = parser.parse_args()
    return args


def substitute():
    # 替换策略
    pass



print('start')
parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=2000)
# parser.add_argument('--output', type=str, default='interpretability_compositions.pkl')
parser.add_argument('--output', type=str, default='energy.csv')
parser.add_argument('--generational', type=str, default='generational_outcomes.pkl')
args = parser.parse_args()

params = GASearchParams(['energy'], [0])
params.population_size = 20
params.generations = 1000
params.attempts = 100
sizes = args.size

count = len(os.listdir(r'GA_new_structure'))
store_path = os.path.join(r'GA_new_structure', str(count))
if not os.path.exists(store_path):
    os.mkdir(store_path)
else:
    os.system("rm -rf " + store_path)
    os.mkdir(store_path)
print(store_path)

population, excluded_elements, included_elements = get_icsd_compounds()     # change this to the choice of initial population you want
print(len(population))
print('excluded_elements: ', excluded_elements)

generational_records = search_and_optimize(params, population, excluded_elements=excluded_elements, included_elements=included_elements)
# with open(os.path.join(store_path, args.generational), 'wb') as f:
#     pickle.dump(generational_records, f)
# generational_records = pickle.load(open(r'generational_outcomes.pkl', 'rb'))
print(generational_records['population'][:10])
print(generational_records['badness'][:10])

control_comps, treatment_comps = {}, {}

# for attempt in generational_records:
#     for generation in generational_records[attempt]:
#         structures = generational_records[attempt][generation]['population'].copy()
#         scores = generational_records[attempt][generation]['badness'].copy()
#         for i, (st, score) in enumerate(zip(structures, scores)):
#             if score < 0:
#                 treatment_comps[(attempt, generation, i)] = score

structures = generational_records['population'].copy()
scores = generational_records['badness'].copy()
for i, (st, score) in enumerate(zip(structures, scores)):
    if score < 0:
        treatment_comps[i] = score

# control_comps = [k for k, v in sorted(control_comps.items(), key=lambda item: item[1], reverse=True)][:sizes]
# treatment_comps = [(k, v) for k, v in sorted(treatment_comps.items(), key=lambda item: item[1])]
treatment_comps = [(k, v) for k, v in treatment_comps.items()]
# print('Size of control set:', len(control_comps))
print('Size of sorted treatment set: ', len(treatment_comps))
# control_comps = list(control_comps)
treatment_comps = list(treatment_comps)

indup_structures = []
Matcher = StructureMatcher()
count = 0
# structure_results = {'structure': [], 'energy': []}

structure_results = {}
for i, st in enumerate(treatment_comps):
    if count >= sizes:
        break
    structure = generational_records['population'][st[0]]
    same = False
    for struct in indup_structures:
        same = Matcher.fit(structure, struct)
        if same:
            break
    if same:
        continue
    indup_structures.append(structure)
    cif_writer = CifWriter(structure)
    cif_writer.write_file(os.path.join(store_path, str(count) + '.cif'))
    structure_results[str(count) + '.cif'] = st[1]
    count += 1
    # structure_results['structure'].append(generational_records[st[0][0]][st[0][1]]['population'][st[0][2]])
    # structure_results['energy'].append(st[1])

with open(os.path.join(store_path, args.output), 'w') as f:
    writer = csv.writer(f)
    for key, energy in structure_results.items():
        writer.writerow([key, energy])
    f.close()
# with open(os.path.join(store_path, args.output), 'wb') as f:
#     pickle.dump(structure_results, f)
# f.close()
