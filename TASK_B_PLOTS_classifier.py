import matplotlib.pyplot as plt
from Prediction_analysis import all_2_start, all_2_end, all_3_start, all_3_end, all_4_start, all_4_end,\
ks_k_2, ks_s_2, ks_k_3, ks_s_3, ks_k_4, ks_s_4, kg_k_2, kg_g_2, kg_k_3, kg_g_3, kg_k_4, kg_g_4,\
go_g_2, go_o_2, go_g_3, go_o_3, go_g_4, go_o_4, so_s_2, so_o_2, so_s_3, so_o_3, so_s_4, so_o_4


# ---------------------------------------------------------------
#  BOX PLOTS -OVERALL
# ---------------------------------------------------------------

overall_data = [all_2_start, all_2_end, all_3_start, all_3_end, all_4_start, all_4_end]

fig1, ax1 = plt.subplots()
ax1.set_title('OVERALL')
ax1.boxplot(overall_data, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - KEYBOARD TO STRING
# ---------------------------------------------------------------

KS_data = [ks_k_2, ks_s_2, ks_k_3, ks_s_3, ks_k_4, ks_s_4]

fig1, ax1 = plt.subplots()
ax1.set_title('KEYBOARD TO STRING')
ax1.boxplot(KS_data, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - KEYBOARD TO GUITAR
# ---------------------------------------------------------------

KG_data = [kg_k_2, kg_g_2, kg_k_3, kg_g_3, kg_k_4, kg_g_4]

fig1, ax1 = plt.subplots()
ax1.set_title('KEYBOARD TO GUITAR')
ax1.boxplot(KG_data, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - GUITAR TO ORGAN
# ---------------------------------------------------------------

GO_data = [go_g_2, go_o_2, go_g_3, go_o_3, go_g_4, go_o_4]

fig1, ax1 = plt.subplots()
ax1.set_title('GUITAR TO ORGAN')
ax1.boxplot(GO_data, showmeans=True)
plt.show()


# ---------------------------------------------------------------
#  BOX PLOTS - STRING TO ORGAN
# ---------------------------------------------------------------

SO_data = [so_s_2, so_o_2, so_s_3, so_o_3, so_s_4, so_o_4]

fig1, ax1 = plt.subplots()
ax1.set_title('STRING TO ORGAN')
ax1.boxplot(SO_data, showmeans=True)
plt.show()










