
'''
Python script example for plotting Tidy3D simulation results using GUI exported data file.
Please modify according to the specific dataset you download.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path of the data file
fname = './data.csv'

# Print header information
with open(fname, 'r') as f:
    for i in range(7):
        print(f.readline()[0:-1])

# Import data to DataFrame object
df = pd.read_csv(fname, skiprows=7)
columns = df.columns

# Plot (2D data array example)
x = df[columns[0]].drop_duplicates() # Select the 'x' column for horizontal axis
y = df[columns[1]].drop_duplicates() # Select the 'z' column for vertical axis
value = np.asarray(df[columns[2]].values.reshape((x.shape[0], y.shape[0])), complex).real
fig, ax = plt.subplots(1, 1)
im = ax.pcolormesh(x, y, np.transpose(value))
plt.colorbar(im)
plt.show()
