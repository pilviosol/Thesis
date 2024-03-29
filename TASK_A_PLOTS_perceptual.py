import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functions import PLOT_PRINT_OPTIONS

save_path = "/nas/home/spol/Thesis/plots_thesis/"

df = pd.read_excel('/nas/home/spol/Thesis/risposte_TASK_A.xlsx')
datas = df.to_numpy(dtype=None, copy=False)

strings = np.concatenate((datas[:, 0], datas[:, 1], datas[:, 2], datas[:, 3], datas[:, 4]))
keyboards = np.concatenate((datas[:, 5], datas[:, 6], datas[:, 7], datas[:, 8], datas[:, 9]))
guitars = np.concatenate((datas[:, 10], datas[:, 11], datas[:, 12], datas[:, 13], datas[:, 14]))
organs = np.concatenate((datas[:, 15], datas[:, 16], datas[:, 17], datas[:, 18], datas[:, 19]))

all = [strings, keyboards, guitars, organs]
all_string = ['strings', 'keyboards', 'guitars', 'organs']

# ---------------------------------------------------------------
#  BOX PLOTS - ALL TOGETHER
# ---------------------------------------------------------------

fig1, ax1 = plt.subplots()
ax1.set_title('ALL')
ax1.boxplot(all, showmeans=True)
plt.show()

# ---------------------------------------------------------------
#  BOX PLOTS - SEABORN
# ---------------------------------------------------------------

sns.set_theme(style="darkgrid")
ax = sns.boxplot(data=all, showmeans=True, meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor":"white"})
ax.set_xticklabels(['STRING', 'KEYBOARD', 'GUITAR', 'ORGAN'])
ax.set(xlabel='Timbral class', ylabel='Score')
plt.savefig(save_path + 'task_a_perceptual', **PLOT_PRINT_OPTIONS)
plt.show()

print('debug')
