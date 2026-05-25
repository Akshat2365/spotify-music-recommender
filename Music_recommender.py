import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import os
print(os.getcwd())
df = pd.read_excel("/Users/akshatsharma/Documents/spotifyML_project/spotify_dataset.xlsx")

#Check for null values.
df.isnull().sum()

#Data preprocessing (as the data set is huge, we can drop null values. 
df = df.dropna().reset_index(drop=True)
df.isnull().sum()
print(df)

#All possible plots to derive meaningfully insight
subset = ['danceability','energy','loudness','speechiness','acousticness',
        'instrumentalness','liveness','valence','tempo','duration_ms']

sns.pairplot(df[subset], diag_kind="kde", corner=True,)
plt.suptitle("Pairplot of Selected Spotify Features", y=1.02)
plt.show()

#correlation of features 
numeric_df = df.select_dtypes(include=['float64','int64'])

corr = numeric_df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix of Spotify Features", fontsize=14)
plt.show()

# Distribution of songs by playlist_genre

if 'playlist_genre' in df.columns:
    plt.figure(figsize=(12,6))
    order = df['playlist_genre'].value_counts().index
    sns.countplot(data=df, x='playlist_genre', order=order, palette="tab10")
    plt.title("Number of Songs by Playlist Genre")
    plt.xlabel("Playlist Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Distribution of songs by playlist_name

if 'playlist_name' in df.columns:
    top_playlists = df['playlist_name'].value_counts().nlargest(15).index
    plt.figure(figsize=(12,6))
    sns.countplot(data=df[df['playlist_name'].isin(top_playlists)], 
                  y='playlist_name', order=top_playlists, palette="tab20")
    plt.title("Number of Songs in Top 15 Playlists")
    plt.xlabel("Count")
    plt.ylabel("Playlist Name")
    plt.tight_layout()
    plt.show()

#KMeans clustering Model and Recommendation System
features = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness',
    'valence','tempo','duration_ms'
]
features = [f for f in features if f in df.columns]  # keep only available ones

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[features])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("Cluster sizes:\n", df['cluster'].value_counts())

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'],
                palette="tab10", s=30, alpha=0.7)
plt.title("Clusters of Songs (PCA 2D projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

cluster_nn_models = {}
for c in sorted(df['cluster'].unique()):
    cluster_indices = df[df['cluster'] == c].index
    nn = NearestNeighbors(n_neighbors=6, metric='cosine')  # itself + 5 recs
    nn.fit(X_scaled[cluster_indices])
    cluster_nn_models[c] = (nn, cluster_indices)

def recommend_song(song_name, n_recs=5):
    """Recommend similar songs based on cluster + nearest neighbors."""
    if 'track_name' not in df.columns:
        raise ValueError("Dataset must have a 'track_name' column")

    # Find song (case-insensitive)
    matches = df[df['track_name'].str.lower() == song_name.lower()]
    if matches.empty:
        matches = df[df['track_name'].str.lower().str.contains(song_name.lower(), na=False)]
    if matches.empty:
        print("Song not found in dataset.")
        return

    idx = matches.index[0]
    cluster_id = df.loc[idx, 'cluster']

    # Get NN model for this cluster
    nn, cluster_indices = cluster_nn_models[cluster_id]
    cluster_pos = list(cluster_indices).index(idx)
    distances, indices = nn.kneighbors(
        X_scaled[cluster_indices][cluster_pos].reshape(1, -1),
        n_neighbors=n_recs+1
    )

    rec_indices = [cluster_indices[i] for i in indices[0][1:]]  # skip itself
    recs = df.loc[rec_indices].copy()
    recs['similarity'] = 1 - distances[0][1:]

    cols = ['track_name','artist_name','playlist_genre','playlist_name','cluster','similarity']
    recs = recs[[c for c in cols if c in recs.columns]]
    return recs.reset_index(drop=True)

example_song = df['track_name'].iloc[0]  # pick first song in dataset
print(f"Recommendations for: {example_song}\n")
print(recommend_song(example_song, n_recs=5))

