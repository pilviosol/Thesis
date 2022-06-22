import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('/nas/home/spol/Thesis/risposte_.xlsx')
df_t = df.transpose()
a = df.to_numpy(dtype=None, copy=False)

point_2 = np.concatenate((a[:, 0], a[:, 3], a[:, 6], a[:, 9]))
point_3 = np.concatenate((a[:, 1], a[:, 4], a[:, 7], a[:, 10]))
point_4 = np.concatenate((a[:, 2], a[:, 5], a[:, 8], a[:, 11]))

all = [point_2, point_3, point_4]

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


# ---------------------------------------------------------------
#  BOX PLOTS - ALL TOGETHER
# ---------------------------------------------------------------

fig1, ax1 = plt.subplots()
ax1.set_title('ALL')
ax1.boxplot(all, showmeans=True)
plt.show()

print('debug')
