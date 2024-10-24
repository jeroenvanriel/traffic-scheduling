import pandas as pd
import numpy as np
import time
import glob, re
import csv
import math
import numpy as np
from tqdm import trange

from exact import solve
from threshold import threshold_heuristic
from automaton import evaluate
from util import lane_order


def eval_exact(data):
    def solve_instance(x, cut):
        start = time.time()
        y = solve(x['instance'], cutting_planes=cut)
        return y['y'], y['obj'], time.time() - start

    for cut in [0, 1, 2]:
        # apply branch-and-bound with different cutting planes active
        data[[f'y_opt_{cut}', f'y_opt_obj_{cut}', f'running_time_{cut}']] = \
            data.apply(lambda x: solve_instance(x, cut), axis=1, result_type='expand')

    # we verify the cutting planes by comparing objectives
    data['matching1'] = data.apply(lambda x: math.isclose(x['y_opt_obj_0'], x['y_opt_obj_1']), axis=1)
    data['matching2'] = data.apply(lambda x: math.isclose(x['y_opt_obj_0'], x['y_opt_obj_2']), axis=1)
    assert data['matching1'].all() and data['matching2'].all()

    # rename and remove columns
    data['y_opt'] = data['y_opt_0']
    data['y_opt_obj'] = data['y_opt_obj_0']
    data = data.drop([
            'y_opt_0', 'y_opt_obj_0',
            'y_opt_1', 'y_opt_obj_1',
            'y_opt_2', 'y_opt_obj_2',
            'matching1', 'matching2'
       ], axis=1)

    # store schedules with their lane order
    data['y_eta'] = data.apply(lambda x: lane_order(x['y_opt']), axis=1)

    return data

csv_data = pd.DataFrame()

# load all the test_{i}.pkl files
files = glob.glob("data/test_*.pkl")
for file in files:
    print(f"processing {file}")
    data = pd.read_pickle(file)

    if not 'y_opt' in data:
        print('branch-and-bound')
        data = eval_exact(data)

    if not 'y_threshold_obj' in data:
        print('heuristic')

        heuristic = lambda s: threshold_heuristic(s, tau=0.5)

        data['y_threshold_obj'] = data['instance'].apply(
                lambda instance: evaluate(instance, heuristic)['obj'])

    # save everything with pickle
    data.to_pickle(file)
    # save the objective values and running times for report
    res = data.drop(['instance', 'y_opt', 'y_eta'], axis=1, errors='ignore')
    res['set_id'] = int(re.findall(r'\d+', file)[0])
    csv_data = pd.concat([csv_data, res], ignore_index=True)

csv_data.to_csv(f"../report/data/results.csv", index=False)
