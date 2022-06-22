import matplotlib.pyplot as plt

from Prediction_analysis import all_2_start, all_2_end, all_3_start, all_3_end, all_4_start, all_4_end, \
    ks_k_2, ks_s_2, ks_k_3, ks_s_3, ks_k_4, ks_s_4, kg_k_2, kg_g_2, kg_k_3, kg_g_3, kg_k_4, kg_g_4, \
    go_g_2, go_o_2, go_g_3, go_o_3, go_g_4, go_o_4, so_s_2, so_o_2, so_s_3, so_o_3, so_s_4, so_o_4, \
    ks_k_1_MEAN, ks_s_1_MEAN, ks_k_2_MEAN, ks_s_2_MEAN, ks_k_3_MEAN, ks_s_3_MEAN, ks_k_4_MEAN, ks_s_4_MEAN, ks_k_5_MEAN, ks_s_5_MEAN, \
    kg_k_1_MEAN, kg_g_1_MEAN, kg_k_2_MEAN, kg_g_2_MEAN, kg_k_3_MEAN, kg_g_3_MEAN, kg_k_4_MEAN, kg_g_4_MEAN, kg_k_5_MEAN, kg_g_5_MEAN, \
    go_g_1_MEAN, go_o_1_MEAN, go_g_2_MEAN, go_o_2_MEAN, go_g_3_MEAN, go_o_3_MEAN, go_g_4_MEAN, go_o_4_MEAN, go_g_5_MEAN, go_o_5_MEAN, \
    so_s_1_MEAN, so_o_1_MEAN, so_s_2_MEAN, so_o_2_MEAN, so_s_3_MEAN, so_o_3_MEAN, so_s_4_MEAN, so_o_4_MEAN, so_s_5_MEAN, so_o_5_MEAN

"""
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
"""

# ---------------------------------------------------------------
#  ALTRI GRAFICI
# ---------------------------------------------------------------


# ---------------------------------------------------------------
#  ALTRI GRAFICI - KEYBOARD TO STRING
# ---------------------------------------------------------------

ks_k = [ks_k_1_MEAN, ks_k_2_MEAN, ks_k_3_MEAN, ks_k_4_MEAN, ks_k_5_MEAN]
ks_s = [ks_s_1_MEAN, ks_s_2_MEAN, ks_s_3_MEAN, ks_s_4_MEAN, ks_s_5_MEAN]

x1 = [1, 2, 3, 4, 5]
y1 = ks_k
y2 = ks_s
plt.plot(x1, y1, label="keyboard probability")
plt.plot(x1, y2, label="string probability")

plt.xlabel('Interpolation Points')
plt.ylabel('Probability')
plt.title('Keyboard to String')
plt.legend()
plt.show()

# ---------------------------------------------------------------
#  ALTRI GRAFICI - KEYBOARD TO GUITAR
# ---------------------------------------------------------------

kg_k = [kg_k_1_MEAN, kg_k_2_MEAN, kg_k_3_MEAN, kg_k_4_MEAN, kg_k_5_MEAN]
kg_g = [kg_g_1_MEAN, kg_g_2_MEAN, kg_g_3_MEAN, kg_g_4_MEAN, kg_g_5_MEAN]

x1 = [1, 2, 3, 4, 5]
y1 = kg_k
y2 = kg_g
plt.plot(x1, y1, label="keyboard probability")
plt.plot(x1, y2, label="guitar probability")

plt.xlabel('Interpolation Points')
plt.ylabel('Probability')
plt.title('Keyboard to Guitar')
plt.legend()
plt.show()

# ---------------------------------------------------------------
#  ALTRI GRAFICI - GUITAR TO ORGAN
# ---------------------------------------------------------------

go_g = [go_g_1_MEAN, go_g_2_MEAN, go_g_3_MEAN, go_g_4_MEAN, go_g_5_MEAN]
go_o = [go_o_1_MEAN, go_o_2_MEAN, go_o_3_MEAN, go_o_4_MEAN, go_o_5_MEAN]

x1 = [1, 2, 3, 4, 5]
y1 = go_g
y2 = go_o
plt.plot(x1, y1, label="guitar probability")
plt.plot(x1, y2, label="organ probability")

plt.xlabel('Interpolation Points')
plt.ylabel('Probability')
plt.title('Guitar to Organ')
plt.legend()
plt.show()

# ---------------------------------------------------------------
#  ALTRI GRAFICI - STRING TO ORGAN
# ---------------------------------------------------------------

so_s = [so_s_1_MEAN, so_s_2_MEAN, so_s_3_MEAN, so_s_4_MEAN, so_s_5_MEAN]
so_o = [so_o_1_MEAN, so_o_2_MEAN, so_o_3_MEAN, so_o_4_MEAN, so_o_5_MEAN]

x1 = [1, 2, 3, 4, 5]
y1 = so_s
y2 = so_o
plt.plot(x1, y1, label="string probability")
plt.plot(x1, y2, label="organ probability")

plt.xlabel('Interpolation Points')
plt.ylabel('Probability')
plt.title('String to Organ')
plt.legend()
plt.show()

print('debug')