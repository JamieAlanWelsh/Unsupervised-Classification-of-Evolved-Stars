import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics.cluster import rand_score

# =============================================================================
# Loading Dataset
# =============================================================================

df = pd.read_csv(r'Datasets\mainDataset\photometric_dataset_v01.csv',index_col=0)

groundTruth = df['source_num']

# feat_cols will need to be adjusted if using a different dataset
feat_cols = ['W12','W23','W34','bp_rp_mag','JHmag','HKmag','RJ','KW1']

# # Normalising all values between 0 and 1 
# scaler = MinMaxScaler()
# for column in feat_cols:
#     scaler.fit(df[[column]])
#     df[column] = scaler.transform(df[[column]])

X = df[feat_cols].values

# =============================================================================
# Plotting ground truth in 2 dimensions with PCA
# =============================================================================

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X)

df['pca_one'] = pca_result[:,0]
df['pca_two'] = pca_result[:,1] 
# df['pca_three'] = pca_result[:,2]

x = 'pca_one'
y = 'pca_two'

sns.lmplot(x = x, y = y, data=df, hue='source_type', 
           fit_reg=False, size=8, scatter_kws={"s": 1.25})

# =============================================================================
# DBSCAN
# =============================================================================

# Calculating optimal epsilon
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)
distances = np.sort(distances[:,10], axis=0)

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

optimalEpsilon = distances[knee.knee]

clustering = DBSCAN(eps= optimalEpsilon+0.2, min_samples=40).fit(X)

dbscan_cluster = clustering.labels_

df['DBSCAN'] = dbscan_cluster

sns.lmplot(x = x, y = y, data=df, hue='DBSCAN', 
           fit_reg=False, size=10, scatter_kws={"s": 2})

dbscan = df['DBSCAN']

# =============================================================================
# print cluster distribution and accuracy with rand score
# =============================================================================
def cluster_distribution(X, column):
    # getting unique clusters and source_type from column
    clusters = set(X[column].tolist())
    source_types = set(X['source_type'].tolist())
    for cluster in clusters:
        # generate subset of df for each cluster and count rows
        subCluster = X.loc[X[column] == cluster]
        clusterRows = subCluster.shape[0]
        print(f"cluster {cluster}:")
        print(f"{clusterRows} sources")
        # generate subset of sub-df for each source_type and count rows
        for source_type in source_types:
            subSource = subCluster.loc[subCluster['source_type'] == source_type]
            sourceRows = subSource.shape[0]
            # calculate distrbituion and print results
            result = sourceRows/clusterRows*100
            # round to 2 decimal places
            result = "{:.2f}".format(result)
            print(f"{result}% {source_type}")
        print("\n")
        
        
cluster_distribution(df, 'DBSCAN')

# computing accuracy
accuracy = rand_score(groundTruth, dbscan)
print(f"Rand index score: {accuracy}")

# removing outliers and recalculating accuracy
df2 = df.drop(df[df.HDBSCAN == -1].index)
accuracy_noOutliers = rand_score(df2['source_num'], df2['HDBSCAN'])
print(f"Rand index score without outliers: {accuracy_noOutliers}")