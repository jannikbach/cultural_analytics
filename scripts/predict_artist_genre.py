import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from utils import get_base_path
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

def get_color_names():
    with open(Path(get_base_path()) / "color_buckets.pkl", "rb") as f:
        color_map = pickle.load(f)
        return list(color_map.values())

def visualize_example(artist_id, release_id):
    df = pd.read_csv("./.fetched_data/discogs_with_single_colors.csv")

    color_names = get_color_names()

    color_map = {
        'Red': 'Rot',
        'Orange': 'Orange',
        'Yellow': 'Gelb',
        'Chartreuse': 'Gelbgrün',
        'Green': 'Grün',
        'Spring Green': 'Frühlingsgrün',
        'Cyan': 'Cyan',
        'Azure': 'Azurblau',
        'Blue': 'Blau',
        'Violet': 'Violett',
        'Magenta': 'Magenta',
        'Rose': 'Rosa',
        'Black': 'Schwarz',
        'White': 'Weiß',
        'Gray': 'Grau'
    }

    df[color_names] = df[color_names].replace("", pd.NA)
    df = df.dropna(subset=color_names)
    artist_data = df[df["Artist-Id"] == artist_id][color_names]
    
    artist_avg = artist_data.mean()
    
    cover_data = df[df["Id"] == release_id][color_names]
    if cover_data.empty:
        print(f"No data!")
        return
    cover_values = cover_data.iloc[0]

    labels = np.array([color_map[c] for c in color_names])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    
    artist_avg_values = artist_avg.tolist() + [artist_avg.iloc[0]]
    cover_values = cover_values.tolist() + [cover_values.iloc[0]]
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    ax.plot(angles, artist_avg_values, label=f"Durchschnitt Künstler", linestyle="solid", linewidth=2)
    ax.fill(angles, artist_avg_values, alpha=0.2)
    
    ax.plot(angles, cover_values, label=f"Release-Cover", linestyle="dashed", linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13, rotation=45)
    
    #plt.title(f"Vergleich: Künstler {artist_id} vs. Cover {release_id}", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=13)

    plt.show()
    
def visualize_tsne(X_scaled, y_encoded, le:LabelEncoder):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_encoded, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(le.classes_))
    #plt.title("t-SNE Visualisierung der Genres")
    plt.show()

def main():
    df = pd.read_csv("./.fetched_data/discogs_with_all_colors.csv")

    color_names = get_color_names()

    # remove entries without color-values
    df[color_names] = df[color_names].replace("", pd.NA)
    df = df.dropna(subset=color_names)

    artist_counts = df["Artist-Id"].value_counts()

    # remove artists with <5 releases
    min_covers = 5
    valid_artists = artist_counts[artist_counts >= min_covers].index

    df = df[df["Artist-Id"].isin(valid_artists)]

    # calculate means for artists
    artist_means = df.groupby("Artist-Id")[color_names].mean()

    # calculate diff of each release to artist mean
    df[color_names] = df.apply(lambda row: row[color_names] - artist_means.loc[row["Artist-Id"]], axis=1)

    # releases with multiple styles to multiple entries with one style
    expanded_rows = []
    for _, row in df.iterrows():
        genres = row['Subgenres'].split(";")
        for genre in genres:
            new_row = row.copy()
            new_row['genre'] = genre.strip()
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    #counts = expanded_df['genre'].value_counts()

    # valid_genres = counts[counts >= 1000].index
    # df_filtered = expanded_df[expanded_df['genre'].isin(valid_genres)]

    # min_count = df_filtered['genre'].value_counts().min()

    # balanced_df = df_filtered.groupby('genre', group_keys=False).apply(lambda x: x.sample(min_count))

    max_count = expanded_df['genre'].value_counts().max()

    balanced_df = expanded_df.groupby('genre', group_keys=False).apply(lambda x: x.sample(max_count, replace=True))

    X = balanced_df[color_names].values
    y = balanced_df['genre'].values

    # scale the color-diffs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Label-Encoding for genre (get numeric)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    visualize_tsne(X_scaled, y_encoded, le)

if __name__ == "__main__":
    #visualize_example(4321662, 16097207)
    main()
    