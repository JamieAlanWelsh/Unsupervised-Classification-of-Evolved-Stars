import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

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

m = TSNE(learning_rate=100,
         perplexity=400)

tsne_features = m.fit_transform(X)

df['x'] = tsne_features[:,0]

df['y'] = tsne_features[:,1]

sns.lmplot(x='x',y='y', data=df, hue='source_type', 
            fit_reg=False, size=10, scatter_kws={"s": 4})