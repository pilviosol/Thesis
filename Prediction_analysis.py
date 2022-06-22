import numpy as np

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

ks_k_2 = []
ks_s_2 = []
ks_k_3 = []
ks_s_3 = []
ks_k_4 = []
ks_s_4 = []

for idx, k in enumerate(ks_k):
    if (idx % 5) == 1:
        ks_k_2.append(k)
    if (idx % 5) == 2:
        ks_k_3.append(k)
    if (idx % 5) == 3:
        ks_k_4.append(k)

for idx, s in enumerate(ks_s):
    if (idx % 5) == 1:
        ks_s_2.append(s)
    if (idx % 5) == 2:
        ks_s_3.append(s)
    if (idx % 5) == 3:
        ks_s_4.append(s)


# ---------------------------------------------------------------
#  KG
# ---------------------------------------------------------------
kg_k = kg_predictions[:, 1]
kg_g = kg_predictions[:, 2]

kg_k_2 = []
kg_g_2 = []
kg_k_3 = []
kg_g_3 = []
kg_k_4 = []
kg_g_4 = []

for idx, k in enumerate(kg_k):
    if (idx % 5) == 1:
        kg_k_2.append(k)
    if (idx % 5) == 2:
        kg_k_3.append(k)
    if (idx % 5) == 3:
        kg_k_4.append(k)

for idx, g in enumerate(kg_g):
    if (idx % 5) == 1:
        kg_g_2.append(g)
    if (idx % 5) == 2:
        kg_g_3.append(g)
    if (idx % 5) == 3:
        kg_g_4.append(g)


# ---------------------------------------------------------------
#  GO
# ---------------------------------------------------------------
go_g = go_predictions[:, 2]
go_o = go_predictions[:, 3]

go_g_2 = []
go_o_2 = []
go_g_3 = []
go_o_3 = []
go_g_4 = []
go_o_4 = []

for idx, g in enumerate(go_g):
    if (idx % 5) == 1:
        go_g_2.append(g)
    if (idx % 5) == 2:
        go_g_3.append(g)
    if (idx % 5) == 3:
        go_g_4.append(g)

for idx, o in enumerate(go_o):
    if (idx % 5) == 1:
        go_o_2.append(o)
    if (idx % 5) == 2:
        go_o_3.append(o)
    if (idx % 5) == 3:
        go_o_4.append(o)


# ---------------------------------------------------------------
#  SO
# ---------------------------------------------------------------
so_s = go_predictions[:, 0]
so_o = go_predictions[:, 3]

so_s_2 = []
so_o_2 = []
so_s_3 = []
so_o_3 = []
so_s_4 = []
so_o_4 = []

for idx, s in enumerate(so_s):
    if (idx % 5) == 1:
        so_s_2.append(s)
    if (idx % 5) == 2:
        so_s_3.append(s)
    if (idx % 5) == 3:
        so_s_4.append(s)

for idx, o in enumerate(so_o):
    if (idx % 5) == 1:
        so_o_2.append(o)
    if (idx % 5) == 2:
        so_o_3.append(o)
    if (idx % 5) == 3:
        so_o_4.append(o)
# ---------------------------------------------------------------
#  OVERALL
# ---------------------------------------------------------------

all_2_start = np.concatenate((ks_k_2, kg_k_2, go_g_2, so_s_2), axis=0)
all_2_end = np.concatenate((ks_s_2, kg_g_2, go_o_2, so_o_2), axis=0)
all_3_start = np.concatenate((ks_k_3, kg_k_3, go_g_3, so_s_3), axis=0)
all_3_end = np.concatenate((ks_s_3, kg_g_3, go_o_3, so_o_3), axis=0)
all_4_start = np.concatenate((ks_k_4, kg_k_4, go_g_4, so_s_4), axis=0)
all_4_end = np.concatenate((ks_s_4, kg_g_4, go_o_4, so_o_4), axis=0)

# ---------------------------------------------------------------
#  RESULTS
# ---------------------------------------------------------------

ks_k_2_MEAN = np.mean(ks_k_2)
ks_s_2_MEAN = np.mean(ks_s_2)
ks_k_3_MEAN = np.mean(ks_k_3)
ks_s_3_MEAN = np.mean(ks_k_3)
ks_k_4_MEAN = np.mean(ks_k_4)
ks_s_4_MEAN = np.mean(ks_k_4)
kg_k_2_MEAN = np.mean(kg_k_2)
kg_g_2_MEAN = np.mean(kg_g_2)
kg_k_3_MEAN = np.mean(kg_k_3)
kg_g_3_MEAN = np.mean(kg_g_3)
kg_k_4_MEAN = np.mean(kg_k_4)
kg_g_4_MEAN = np.mean(kg_g_4)
go_g_2_MEAN = np.mean(go_g_2)
go_o_2_MEAN = np.mean(go_o_2)
go_g_3_MEAN = np.mean(go_g_3)
go_o_3_MEAN = np.mean(go_o_3)
go_g_4_MEAN = np.mean(go_g_4)
go_o_4_MEAN = np.mean(go_o_4)
so_s_2_MEAN = np.mean(so_s_2)
so_o_2_MEAN = np.mean(so_o_2)
so_s_3_MEAN = np.mean(so_s_3)
so_o_3_MEAN = np.mean(so_o_3)
so_s_4_MEAN = np.mean(so_s_4)
so_o_4_MEAN = np.mean(so_o_4)


ks_k_2_VAR = np.var(ks_k_2)
ks_s_2_VAR = np.var(ks_s_2)
ks_k_3_VAR = np.var(ks_k_3)
ks_s_3_VAR = np.var(ks_k_3)
ks_k_4_VAR = np.var(ks_k_4)
ks_s_4_VAR = np.var(ks_k_4)
kg_k_2_VAR = np.var(kg_k_2)
kg_g_2_VAR = np.var(kg_g_2)
kg_k_3_VAR = np.var(kg_k_3)
kg_g_3_VAR = np.var(kg_g_3)
kg_k_4_VAR = np.var(kg_k_4)
kg_g_4_VAR = np.var(kg_g_4)
go_g_2_VAR = np.var(go_g_2)
go_o_2_VAR = np.var(go_o_2)
go_g_3_VAR = np.var(go_g_3)
go_o_3_VAR = np.var(go_o_3)
go_g_4_VAR = np.var(go_g_4)
go_o_4_VAR = np.var(go_o_4)
so_s_2_VAR = np.var(so_s_2)
so_o_2_VAR = np.var(so_o_2)
so_s_3_VAR = np.var(so_s_3)
so_o_3_VAR = np.var(so_o_3)
so_s_4_VAR = np.var(so_s_4)
so_o_4_VAR = np.var(so_o_4)


all_2_start_MEAN = np.mean(all_2_start)
all_3_start_MEAN = np.mean(all_3_start)
all_4_start_MEAN = np.mean(all_4_start)
all_2_end_MEAN = np.mean(all_2_end)
all_3_end_MEAN = np.mean(all_3_end)
all_4_end_MEAN = np.mean(all_4_end)

all_2_start_VAR = np.var(all_2_start)
all_3_start_VAR = np.var(all_3_start)
all_4_start_VAR = np.var(all_4_start)
all_2_end_VAR = np.var(all_2_end)
all_3_end_VAR = np.var(all_3_end)
all_4_end_VAR = np.var(all_4_end)


print('debug')
