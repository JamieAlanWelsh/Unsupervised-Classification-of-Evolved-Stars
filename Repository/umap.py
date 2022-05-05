import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

# =============================================================================
# PREPARING DATASET
# =============================================================================

df = pd.read_csv(r'Datasets\mainDataset\photometric_dataset_v01.csv',index_col=0)

# # Normalising all values between 0 and 1 
# scaler = MinMaxScaler()
# for column in ['W12','W23','W34','bp_rp_mag','JHmag','HKmag','RJ','KW1']:
#     scaler.fit(df[[column]])
#     df[column] = scaler.transform(df[[column]])

# setting the columns that will be accounted with UMAP
X = df[['W12','W23','W34','bp_rp_mag','JHmag','HKmag','RJ','KW1']]

dimensions = 6
dimension_range = range(dimensions)

# =============================================================================
# UMAP
# =============================================================================

review_umapped = umap.UMAP(n_components = dimensions, 
                           #densmap=True,
                           # metric = "euclidean",
                           n_neighbors = 10,
                           min_dist = 0.0,
                           random_state = 43).fit_transform(X)

for num in dimension_range:
    df[f'umap_{num}'] = review_umapped[:,num]
    
# =============================================================================
# PLOTTING IN 2 DIMENSIONS
# =============================================================================

# gets all columns that include substring umap
Y = df[df.columns[df.columns.to_series().str.contains('umap')]]

pca = PCA(n_components=2)

pca_result = pca.fit_transform(Y.values)

df['pca_one'] = pca_result[:,0]
df['pca_two'] = pca_result[:,1] 

# ground truth
sns.lmplot(x='pca_one',y='pca_two', data=df, hue='source_type', 
           fit_reg=False, size=8, scatter_kws={"s": 1.25})