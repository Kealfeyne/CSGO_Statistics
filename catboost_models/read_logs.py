import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset_params_grid = [(1, 1, 1), (1, 5, 5), (1, 10, 10),
                       (2, 5, 5), (2, 10, 10),
                       (3, 5, 5), (3, 10, 10)]


all_logs = []
for dataset_params in dataset_params_grid:
    with open(f'{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_'
              f'{300}iterations_{5}depth_logs.json', 'r') as fp:
        logs = json.load(fp)
        all_logs.append(logs)

field = 'train_loss'
for logs in all_logs:
    plt.plot(logs[field], np.linspace(1, 300, 300))

