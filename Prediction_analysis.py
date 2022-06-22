import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
#  PATH and VARIABLES
# ---------------------------------------------------------------


ks_predictions = np.load('/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/KS_predictions.npy')
kg_predictions = np.load('/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/KG_predictions.npy')
go_predictions = np.load('/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/GO_predictions.npy')
so_predictions = np.load('/nas/home/spol/Thesis/INTERPOLATIONS/22_06_2022/SO_predictions.npy')


# ---------------------------------------------------------------
#  KS
# ---------------------------------------------------------------
ks_k = ks_predictions[:, 1]
ks_s = ks_predictions[:, 0]

ks_k_1 = []
ks_s_1 = []
ks_k_2 = []
ks_s_2 = []
ks_k_3 = []
ks_s_3 = []
ks_k_4 = []
ks_s_4 = []
ks_k_5 = []
ks_s_5 = []

for idx, k in enumerate(ks_k):
    if (idx % 5) == 0:
        ks_k_1.append(k)
    if (idx % 5) == 1:
        ks_k_2.append(k)
    if (idx % 5) == 2:
        ks_k_3.append(k)
    if (idx % 5) == 3:
        ks_k_4.append(k)
    if (idx % 5) == 4:
        ks_k_5.append(k)

for idx, s in enumerate(ks_s):
    if (idx % 5) == 0:
        ks_s_1.append(s)
    if (idx % 5) == 1:
        ks_s_2.append(s)
    if (idx % 5) == 2:
        ks_s_3.append(s)
    if (idx % 5) == 3:
        ks_s_4.append(s)
    if (idx % 5) == 4:
        ks_s_5.append(s)


# ---------------------------------------------------------------
#  KG
# ---------------------------------------------------------------
kg_k = kg_predictions[:, 1]
kg_g = kg_predictions[:, 2]

kg_k_1 = []
kg_g_1 = []
kg_k_2 = []
kg_g_2 = []
kg_k_3 = []
kg_g_3 = []
kg_k_4 = []
kg_g_4 = []
kg_k_5 = []
kg_g_5 = []

for idx, k in enumerate(kg_k):
    if (idx % 5) == 0:
        kg_k_1.append(k)
    if (idx % 5) == 1:
        kg_k_2.append(k)
    if (idx % 5) == 2:
        kg_k_3.append(k)
    if (idx % 5) == 3:
        kg_k_4.append(k)
    if (idx % 5) == 4:
        kg_k_5.append(k)

for idx, g in enumerate(kg_g):
    if (idx % 5) == 0:
        kg_g_1.append(g)
    if (idx % 5) == 1:
        kg_g_2.append(g)
    if (idx % 5) == 2:
        kg_g_3.append(g)
    if (idx % 5) == 3:
        kg_g_4.append(g)
    if (idx % 5) == 4:
        kg_g_5.append(g)


# ---------------------------------------------------------------
#  GO
# ---------------------------------------------------------------
go_g = go_predictions[:, 2]
go_o = go_predictions[:, 3]

go_g_1 = []
go_o_1 = []
go_g_2 = []
go_o_2 = []
go_g_3 = []
go_o_3 = []
go_g_4 = []
go_o_4 = []
go_g_5 = []
go_o_5 = []

for idx, g in enumerate(go_g):
    if (idx % 5) == 0:
        go_g_1.append(g)
    if (idx % 5) == 1:
        go_g_2.append(g)
    if (idx % 5) == 2:
        go_g_3.append(g)
    if (idx % 5) == 3:
        go_g_4.append(g)
    if (idx % 5) == 4:
        go_g_5.append(g)

for idx, o in enumerate(go_o):
    if (idx % 5) == 0:
        go_o_1.append(o)
    if (idx % 5) == 1:
        go_o_2.append(o)
    if (idx % 5) == 2:
        go_o_3.append(o)
    if (idx % 5) == 3:
        go_o_4.append(o)
    if (idx % 5) == 4:
        go_o_5.append(o)


# ---------------------------------------------------------------
#  SO
# ---------------------------------------------------------------
so_s = so_predictions[:, 0]
so_o = so_predictions[:, 3]

so_s_1 = []
so_o_1 = []
so_s_2 = []
so_o_2 = []
so_s_3 = []
so_o_3 = []
so_s_4 = []
so_o_4 = []
so_s_5 = []
so_o_5 = []

for idx, s in enumerate(so_s):
    if (idx % 5) == 0:
        so_s_1.append(s)
    if (idx % 5) == 1:
        so_s_2.append(s)
    if (idx % 5) == 2:
        so_s_3.append(s)
    if (idx % 5) == 3:
        so_s_4.append(s)
    if (idx % 5) == 4:
        so_s_5.append(s)

for idx, o in enumerate(so_o):
    if (idx % 5) == 0:
        so_o_1.append(o)
    if (idx % 5) == 1:
        so_o_2.append(o)
    if (idx % 5) == 2:
        so_o_3.append(o)
    if (idx % 5) == 3:
        so_o_4.append(o)
    if (idx % 5) == 4:
        so_o_5.append(o)
# ---------------------------------------------------------------
#  OVERALL
# ---------------------------------------------------------------

all_1_start = np.concatenate((ks_k_1, kg_k_1, go_g_1, so_s_1), axis=0)
all_1_end = np.concatenate((ks_s_1, kg_g_1, go_o_1, so_o_1), axis=0)
all_2_start = np.concatenate((ks_k_2, kg_k_2, go_g_2, so_s_2), axis=0)
all_2_end = np.concatenate((ks_s_2, kg_g_2, go_o_2, so_o_2), axis=0)
all_3_start = np.concatenate((ks_k_3, kg_k_3, go_g_3, so_s_3), axis=0)
all_3_end = np.concatenate((ks_s_3, kg_g_3, go_o_3, so_o_3), axis=0)
all_4_start = np.concatenate((ks_k_4, kg_k_4, go_g_4, so_s_4), axis=0)
all_4_end = np.concatenate((ks_s_4, kg_g_4, go_o_4, so_o_4), axis=0)
all_5_start = np.concatenate((ks_k_5, kg_k_5, go_g_5, so_s_5), axis=0)
all_5_end = np.concatenate((ks_s_5, kg_g_5, go_o_5, so_o_5), axis=0)

# ---------------------------------------------------------------
#  RESULTS
# ---------------------------------------------------------------

ks_k_1_MEAN = np.mean(ks_k_1)
ks_s_1_MEAN = np.mean(ks_s_1)
ks_k_2_MEAN = np.mean(ks_k_2)
ks_s_2_MEAN = np.mean(ks_s_2)
ks_k_3_MEAN = np.mean(ks_k_3)
ks_s_3_MEAN = np.mean(ks_s_3)
ks_k_4_MEAN = np.mean(ks_k_4)
ks_s_4_MEAN = np.mean(ks_s_4)
ks_k_5_MEAN = np.mean(ks_k_5)
ks_s_5_MEAN = np.mean(ks_s_5)

kg_k_1_MEAN = np.mean(kg_k_1)
kg_g_1_MEAN = np.mean(kg_g_1)
kg_k_2_MEAN = np.mean(kg_k_2)
kg_g_2_MEAN = np.mean(kg_g_2)
kg_k_3_MEAN = np.mean(kg_k_3)
kg_g_3_MEAN = np.mean(kg_g_3)
kg_k_4_MEAN = np.mean(kg_k_4)
kg_g_4_MEAN = np.mean(kg_g_4)
kg_k_5_MEAN = np.mean(kg_k_5)
kg_g_5_MEAN = np.mean(kg_g_5)

go_g_1_MEAN = np.mean(go_g_1)
go_o_1_MEAN = np.mean(go_o_1)
go_g_2_MEAN = np.mean(go_g_2)
go_o_2_MEAN = np.mean(go_o_2)
go_g_3_MEAN = np.mean(go_g_3)
go_o_3_MEAN = np.mean(go_o_3)
go_g_4_MEAN = np.mean(go_g_4)
go_o_4_MEAN = np.mean(go_o_4)
go_g_5_MEAN = np.mean(go_g_5)
go_o_5_MEAN = np.mean(go_o_5)

so_s_1_MEAN = np.mean(so_s_1)
so_o_1_MEAN = np.mean(so_o_1)
so_s_2_MEAN = np.mean(so_s_2)
so_o_2_MEAN = np.mean(so_o_2)
so_s_3_MEAN = np.mean(so_s_3)
so_o_3_MEAN = np.mean(so_o_3)
so_s_4_MEAN = np.mean(so_s_4)
so_o_4_MEAN = np.mean(so_o_4)
so_s_5_MEAN = np.mean(so_s_5)
so_o_5_MEAN = np.mean(so_o_5)


ks_k_1_VAR = np.var(ks_k_1)
ks_s_1_VAR = np.var(ks_s_1)
ks_k_2_VAR = np.var(ks_k_2)
ks_s_2_VAR = np.var(ks_s_2)
ks_k_3_VAR = np.var(ks_k_3)
ks_s_3_VAR = np.var(ks_s_3)
ks_k_4_VAR = np.var(ks_k_4)
ks_s_4_VAR = np.var(ks_s_4)
ks_k_5_VAR = np.var(ks_k_5)
ks_s_5_VAR = np.var(ks_s_5)

kg_k_1_VAR = np.var(kg_k_1)
kg_g_1_VAR = np.var(kg_g_1)
kg_k_2_VAR = np.var(kg_k_2)
kg_g_2_VAR = np.var(kg_g_2)
kg_k_3_VAR = np.var(kg_k_3)
kg_g_3_VAR = np.var(kg_g_3)
kg_k_4_VAR = np.var(kg_k_4)
kg_g_4_VAR = np.var(kg_g_4)
kg_k_5_VAR = np.var(kg_k_5)
kg_g_5_VAR = np.var(kg_g_5)

go_g_1_VAR = np.var(go_g_1)
go_o_1_VAR = np.var(go_o_1)
go_g_2_VAR = np.var(go_g_2)
go_o_2_VAR = np.var(go_o_2)
go_g_3_VAR = np.var(go_g_3)
go_o_3_VAR = np.var(go_o_3)
go_g_4_VAR = np.var(go_g_4)
go_o_4_VAR = np.var(go_o_4)
go_g_5_VAR = np.var(go_g_5)
go_o_5_VAR = np.var(go_o_5)

so_s_1_VAR = np.var(so_s_1)
so_o_1_VAR = np.var(so_o_1)
so_s_2_VAR = np.var(so_s_2)
so_o_2_VAR = np.var(so_o_2)
so_s_3_VAR = np.var(so_s_3)
so_o_3_VAR = np.var(so_o_3)
so_s_4_VAR = np.var(so_s_4)
so_o_4_VAR = np.var(so_o_4)
so_s_5_VAR = np.var(so_s_5)
so_o_5_VAR = np.var(so_o_5)


all_1_start_MEAN = np.mean(all_1_start)
all_2_start_MEAN = np.mean(all_2_start)
all_3_start_MEAN = np.mean(all_3_start)
all_4_start_MEAN = np.mean(all_4_start)
all_5_start_MEAN = np.mean(all_5_start)

all_1_end_MEAN = np.mean(all_1_end)
all_2_end_MEAN = np.mean(all_2_end)
all_3_end_MEAN = np.mean(all_3_end)
all_4_end_MEAN = np.mean(all_4_end)
all_5_end_MEAN = np.mean(all_5_end)

all_1_start_VAR = np.var(all_1_start)
all_2_start_VAR = np.var(all_2_start)
all_3_start_VAR = np.var(all_3_start)
all_4_start_VAR = np.var(all_4_start)
all_5_start_VAR = np.var(all_5_start)

all_1_end_VAR = np.var(all_1_end)
all_2_end_VAR = np.var(all_2_end)
all_3_end_VAR = np.var(all_3_end)
all_4_end_VAR = np.var(all_4_end)
all_5_end_VAR = np.var(all_5_end)


print('--------------------------')
print('ks_k_1_MEAN: ', ks_k_1_MEAN)
print('ks_s_1_MEAN: ', ks_s_1_MEAN)
print('ks_k_2_MEAN: ', ks_k_2_MEAN)
print('ks_s_2_MEAN: ', ks_s_2_MEAN)
print('ks_k_3_MEAN: ', ks_k_3_MEAN)
print('ks_s_3_MEAN: ', ks_s_3_MEAN)
print('ks_k_4_MEAN: ', ks_k_4_MEAN)
print('ks_s_4_MEAN: ', ks_s_4_MEAN)
print('ks_k_5_MEAN: ', ks_k_5_MEAN)
print('ks_s_5_MEAN: ', ks_s_5_MEAN)

print('--------------------------')
print('kg_k_1_MEAN: ', kg_k_1_MEAN)
print('kg_g_1_MEAN: ', kg_g_1_MEAN)
print('kg_k_2_MEAN: ', kg_k_2_MEAN)
print('kg_g_2_MEAN: ', kg_g_2_MEAN)
print('kg_k_3_MEAN: ', kg_k_3_MEAN)
print('kg_g_3_MEAN: ', kg_g_3_MEAN)
print('kg_k_4_MEAN: ', kg_k_4_MEAN)
print('kg_g_4_MEAN: ', kg_g_4_MEAN)
print('kg_k_5_MEAN: ', kg_k_5_MEAN)
print('kg_g_5_MEAN: ', kg_g_5_MEAN)

print('--------------------------')
print('go_g_1_MEAN: ', go_g_1_MEAN)
print('go_o_1_MEAN: ', go_o_1_MEAN)
print('go_g_2_MEAN: ', go_g_2_MEAN)
print('go_o_2_MEAN: ', go_o_2_MEAN)
print('go_g_3_MEAN: ', go_g_3_MEAN)
print('go_o_3_MEAN: ', go_o_3_MEAN)
print('go_g_4_MEAN: ', go_g_4_MEAN)
print('go_o_4_MEAN: ', go_o_4_MEAN)
print('go_g_5_MEAN: ', go_g_5_MEAN)
print('go_o_5_MEAN: ', go_o_5_MEAN)

print('--------------------------')
print('so_s_1_MEAN: ', so_s_1_MEAN)
print('so_o_1_MEAN: ', so_o_1_MEAN)
print('so_s_2_MEAN: ', so_s_2_MEAN)
print('so_o_2_MEAN: ', so_o_2_MEAN)
print('so_s_3_MEAN: ', so_s_3_MEAN)
print('so_o_3_MEAN: ', so_o_3_MEAN)
print('so_s_4_MEAN: ', so_s_4_MEAN)
print('so_o_4_MEAN: ', so_o_4_MEAN)
print('so_s_5_MEAN: ', so_s_5_MEAN)
print('so_o_5_MEAN: ', so_o_5_MEAN)


print('--------------------------')
print('ks_k_1_VAR ', ks_k_1_VAR)
print('ks_s_1_VAR ', ks_s_1_VAR)
print('ks_k_2_VAR ', ks_k_2_VAR)
print('ks_s_2_VAR ', ks_s_2_VAR)
print('ks_k_3_VAR ', ks_k_3_VAR)
print('ks_s_3_VAR ', ks_s_3_VAR)
print('ks_k_4_VAR ', ks_k_4_VAR)
print('ks_s_4_VAR ', ks_s_4_VAR)
print('ks_k_5_VAR ', ks_k_5_VAR)
print('ks_s_5_VAR ', ks_s_5_VAR)

print('--------------------------')
print('kg_k_1_VAR ', kg_k_1_VAR)
print('kg_g_1_VAR ', kg_g_1_VAR)
print('kg_k_2_VAR ', kg_k_2_VAR)
print('kg_g_2_VAR ', kg_g_2_VAR)
print('kg_k_3_VAR ', kg_k_3_VAR)
print('kg_g_3_VAR ', kg_g_3_VAR)
print('kg_k_4_VAR ', kg_k_4_VAR)
print('kg_g_4_VAR ', kg_g_4_VAR)
print('kg_k_5_VAR ', kg_k_5_VAR)
print('kg_g_5_VAR ', kg_g_5_VAR)

print('--------------------------')
print('go_g_1_VAR ', go_g_1_VAR)
print('go_o_1_VAR ', go_o_1_VAR)
print('go_g_2_VAR ', go_g_2_VAR)
print('go_o_2_VAR ', go_o_2_VAR)
print('go_g_3_VAR ', go_g_3_VAR)
print('go_o_3_VAR ', go_o_3_VAR)
print('go_g_4_VAR ', go_g_4_VAR)
print('go_o_4_VAR ', go_o_4_VAR)
print('go_g_5_VAR ', go_g_5_VAR)
print('go_o_5_VAR ', go_o_5_VAR)

print('--------------------------')
print('so_s_1_VAR ', so_s_1_VAR)
print('so_o_1_VAR ', so_o_1_VAR)
print('so_s_2_VAR ', so_s_2_VAR)
print('so_o_2_VAR ', so_o_2_VAR)
print('so_s_3_VAR ', so_s_3_VAR)
print('so_o_3_VAR ', so_o_3_VAR)
print('so_s_4_VAR ', so_s_4_VAR)
print('so_o_4_VAR ', so_o_4_VAR)
print('so_s_5_VAR ', so_s_5_VAR)
print('so_o_5_VAR ', so_o_5_VAR)

print('--------------------------')
print('--------------------------')
print('all_1_start_MEAN: ', all_1_start_MEAN)
print('all_1_end_MEAN', all_1_end_MEAN)
print('all_2_start_MEAN: ', all_2_start_MEAN)
print('all_2_end_MEAN', all_2_end_MEAN)
print('all_3_start_MEAN: ', all_3_start_MEAN)
print('all_3_end_MEAN', all_3_end_MEAN)
print('all_4_start_MEAN: ', all_4_start_MEAN)
print('all_4_end_MEAN', all_4_end_MEAN)
print('all_5_start_MEAN: ', all_5_start_MEAN)
print('all_5_end_MEAN', all_5_end_MEAN)
print('--------------------------')
print('--------------------------')

print('--------------------------')
print('--------------------------')
print('all_1_start_VAR: ', all_1_start_VAR)
print('all_1_end_VAR', all_1_end_VAR)
print('all_2_start_VAR: ', all_2_start_VAR)
print('all_2_end_VAR', all_2_end_VAR)
print('all_3_start_VAR: ', all_3_start_VAR)
print('all_3_end_VAR', all_3_end_VAR)
print('all_4_start_VAR: ', all_4_start_VAR)
print('all_4_end_VAR', all_4_end_VAR)
print('all_5_start_VAR: ', all_5_start_VAR)
print('all_5_end_VAR', all_5_end_VAR)
print('--------------------------')
print('--------------------------')


print('debig')

