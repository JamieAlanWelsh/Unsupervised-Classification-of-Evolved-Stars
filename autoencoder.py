import pandas as pd
import keras
from keras import layers
from keras.models import Model
from keras.layers import Dense, Input
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA

# =============================================================================
# PREPARING DATASET
# =============================================================================

df = pd.read_csv(r'Datasets\mainDataset\photometric_dataset_v01.csv',index_col=0)

# # Normalising all values between 0 and 1 
# scaler = MinMaxScaler()
# for column in ['W12','W23','W34','bp_rp_mag','JHmag','HKmag','RJ','KW1']:
#     scaler.fit(df[[column]])
#     df[column] = scaler.transform(df[[column]])

# Isolating colours
colors_df = df[['W12','W23','W34','bp_rp_mag','JHmag','HKmag','RJ','KW1']]

# =============================================================================
# ARCHITECTURE
# =============================================================================

d = 8

inputs_dim = colors_df.shape[1]
encoder = Input(shape = (inputs_dim, ))
e = Dense(d, activation = "relu")(encoder)
e = Dense(d, activation = "relu")(e)
e = Dense(d, activation = "relu")(e)

## bottleneck layer
n_bottleneck = 4
## defining it with a name to extract it later
bottleneck_layer = "bottleneck_layer"
# can also be defined with an activation function, relu for instance
bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)
## define the decoder (in reverse)
decoder = Dense(d, activation = "relu")(bottleneck)
decoder = Dense(d, activation = "relu")(decoder)
decoder = Dense(d, activation = "relu")(decoder)
## output layer
output = Dense(inputs_dim)(decoder)
## model
model = Model(inputs = encoder, outputs = output)

model.summary()

# extracting the bottleneck layer we are interested in the most
encoder = Model(inputs = model.input, outputs = bottleneck)

# =============================================================================
# TRAINING
# =============================================================================

model.compile(loss = "mse", optimizer = "adam")
history = model.fit(
    colors_df,
    colors_df,
    batch_size = 32,
    epochs = 10,
    verbose = 1
    #,validation_data = (colors_df,colors_df)
)

review_encoded = encoder.predict(colors_df)

# =============================================================================
# ADDING TO DATAFRAME
# =============================================================================

# adding columns
df['d1'] = review_encoded[:,0]
df['d2'] = review_encoded[:,1]
df['d3'] = review_encoded[:,2]
df['d4'] = review_encoded[:,3]

# setting the columns that will be accounted for in pca
feat_cols = ['d1','d2','d3','d4']

# we want to reduce to 2 components
pca = PCA(n_components=2)

pca_result = pca.fit_transform(df[feat_cols].values)

df['autoencoder_d1'] = pca_result[:,0]
df['autoencoder_d2'] = pca_result[:,1] 

## =============================================================================
## PLOTS
## =============================================================================

sns.lmplot(x='autoencoder_d1',y='autoencoder_d2', data=df, hue='source_type', 
            fit_reg=False, size=10, scatter_kws={"s": 2})
ax = plt.gca()
ax.set_title("Autoencoder")

# below can be used to save dataset

#autoencoder_df.to_csv('autoencoder_experimentation.csv')