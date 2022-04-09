import numpy as np

a = np.array([0, 1, 2, 3, np.nan])
print(a)

for element in a:
    if not(np.isnan(element)):
        print(element)