import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_excel('/nas/home/spol/Thesis/risposte_TASK_B.xlsx')
datas = df.to_numpy(dtype=None, copy=False)

point_2 = np.concatenate((datas[:, 0], datas[:, 3], datas[:, 6], datas[:, 9]))
point_3 = np.concatenate((datas[:, 1], datas[:, 4], datas[:, 7], datas[:, 10]))
point_4 = np.concatenate((datas[:, 2], datas[:, 5], datas[:, 8], datas[:, 11]))

all = [point_2, point_3, point_4]
"""
# ---------------------------------------------------------------
#  BOX PLOTS - POINT 2
# ---------------------------------------------------------------


fig1, ax1 = plt.subplots()
ax1.set_title('POINT 2')
ax1.boxplot(point_2, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - POINT 3
# ---------------------------------------------------------------


fig1, ax1 = plt.subplots()
ax1.set_title('POINT 3')
ax1.boxplot(point_3, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - POINT 4
# ---------------------------------------------------------------


fig1, ax1 = plt.subplots()
ax1.set_title('POINT 4')
ax1.boxplot(point_4, showmeans=True)
plt.show()

"""
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
ax = sns.boxplot(data= all, showmeans=True)
ax.set_xticklabels(['POINT 2', 'POINT 3', 'POINT 4'])
ax.set(xlabel='Interpolation points', ylabel='Score')
plt.show()


print('debug')
