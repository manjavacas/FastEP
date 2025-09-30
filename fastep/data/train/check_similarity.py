import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import os
import re

"""
Script to check the dissimilarity between datasets (higher value = more different)
"""

dirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'df_\d+', d)]
dirs.sort(key=lambda x: int(x.split('_')[1]))

dfs_stats = []
for d in dirs:
    df = pd.read_csv(os.path.join(d, 'observations.csv'))
    stats = df.describe().loc[['mean','std']].values.flatten()
    dfs_stats.append(stats)

dfs_stats = np.array(dfs_stats)

dist_matrix = squareform(pdist(dfs_stats, metric='euclidean'))

plt.figure(figsize=(12,10))
sns.heatmap(dist_matrix, annot=False, cmap='viridis')
plt.title('Dissimilarity matrix')
plt.xlabel('Dataset')
plt.ylabel('Dataset')
plt.savefig('dist_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
