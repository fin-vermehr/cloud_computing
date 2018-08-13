import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

username_l = []
score_l = []
for file in os.listdir('/Users/nif/documents/website/cloud_computing/scored_companies'):
    if '.csv' in file:
        df1 = pd.read_csv('/Users/nif/documents/website/cloud_computing/scored_companies/' + file)

        username_l.append(file.strip('.csv'))
        score = np.mean(df1['score'])
        score_l.append(score)

df = pd.DataFrame({'username': username_l, 'score': score_l})
# df['username'] = username_l
# df['score'] = score_l

df = df.sort_values('score')
print(df)
# n, bins, patches = plt.hist(x, 30, facecolor='blue', alpha=0.5)
# plt.show()
