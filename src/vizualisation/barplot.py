import json
import numpy as np
from matplotlib import pyplot as plt

with open('src/vizualisation/best_params_steps_300.json') as f:
    best_params_steps_300 = json.load(f)
valid_means_300 = []
valid_stds_300 = []
for i, results in enumerate(best_params_steps_300):
    valid_means_300 += [float(best_params_steps_300[results]["valid_acc_mean"])]
    valid_stds_300 += [float(best_params_steps_300[results]["valid_acc_std"])]
test_values_300 = []
for i, results in enumerate(best_params_steps_300):
    test_values_300 += [float(best_params_steps_300[results]["test_acc"])]

with open('src/vizualisation/best_params_steps_200.json') as f:
    best_params_steps_200 = json.load(f)
valid_means_200 = []
valid_stds_200 = []
for i, results in enumerate(best_params_steps_200):
    valid_means_200 += [float(best_params_steps_200[results]["valid_acc_mean"])]
    valid_stds_200 += [float(best_params_steps_200[results]["valid_acc_std"])]
test_values_200 = []
for i, results in enumerate(best_params_steps_200):
    test_values_200 += [float(best_params_steps_200[results]["test_acc"])]

plt.figure(figsize=(15, 10))
ind = np.arange(len(best_params_steps_300))
bar_valid_200 = plt.bar(best_params_steps_300.keys(), valid_means_200, width=0.18, label='Valid acc 200 spd')
plt.errorbar(ind, valid_means_200, 0, valid_stds_200, barsabove=True, fmt='none')

bar_test_200 = plt.bar(ind + 0.18, test_values_200, width=0.18, label='Test acc 200 spd')

bar_valid_300 = plt.bar(ind + 0.42, valid_means_300, width=0.18, label='Valid acc 300 spd')
plt.errorbar(ind + 0.42, valid_means_300, 0, valid_stds_300, barsabove=True, fmt='none')
plt.xticks(ind + 0.28, best_params_steps_300.keys(), rotation='vertical')

bar_test_300 = plt.bar(ind + 0.6, test_values_300, width=0.2, label='Test acc 300 spd')
min300 = min(min(valid_means_300), min(test_values_300))
min200 = min(min(valid_means_200), min(test_values_200))

plt.ylim(min(min200, min300) - 0.05)

plt.legend(handles=[bar_valid_200, bar_test_200, bar_valid_300, bar_test_300, ], loc='upper right')
plt.tight_layout()
plt.savefig('views/models_results_best_params.png')
plt.close()
