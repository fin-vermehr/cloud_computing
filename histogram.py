import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

dict_ = dict()

for file in os.listdir('/Users/nif/documents/website/cloud_computing/scored_companies'):
    if '.csv' in file:
        df = pd.read_csv('/Users/nif/documents/website/cloud_computing/scored_companies/' + file)
        x = df['score']
        username = file.strip('.csv')
        dict_[username] = np.mean(x)

print(dict_)
# n, bins, patches = plt.hist(x, 30, facecolor='blue', alpha=0.5)
# plt.show()
