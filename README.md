# Spotify Songs Genre Segmentation

## Project Overview
This project demonstrates how Spotify-like music recommendations can be built using **clustering techniques**.  
The dataset contains audio features of songs (e.g., danceability, energy, tempo, etc.), along with playlist metadata.  
We group songs into clusters based on these features and then build a **basic recommendation system**.

---

## Steps Performed

### 1. Data Preprocessing
- Loaded dataset (`spotify_dataset.xlsx`).
- Removed null values (`dropna`) and handled missing data using `SimpleImputer`.
- Scaled features using `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)
- Plotted histograms and distributions of features.
- Created **pairplots** and **correlation matrix heatmap**.
- Visualized playlist genres and names.

### 3. Clustering
- Selected key audio features:
  - `danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms`
- Applied **KMeans** clustering to group songs into clusters.
- Visualized clusters in **2D using PCA**.

### 4. Recommendation System
- Built a **content-based recommender**:
  - A song’s cluster is identified.
  - Nearest neighbors are found within that cluster.
  - Recommended top N most similar songs.

---

## Results
- Songs were grouped into clusters that capture their **musical similarity**.
- PCA visualization showed clear cluster separation.
- The recommendation system was able to suggest **similar songs** to a given input.

---

## Usage

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the notebook
Open the Jupyter Notebook:
```bash
jupyter notebook spotify_segmentation.ipynb
```

### 3. Example Recommendation
```python
recommend_song("Shape of You", n_recs=5)
```

Output:
```
Recommendations for: Shape of You
----------------------------------------------------
Perfect              | Ed Sheeran | pop | cluster 2
Thinking Out Loud    | Ed Sheeran | pop | cluster 2
Love Yourself        | Justin Bieber | pop | cluster 2
...
```

---
