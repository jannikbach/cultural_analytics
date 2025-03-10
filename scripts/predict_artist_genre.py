import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from utils import get_base_path
import pickle
from pathlib import Path

def get_color_names():
    with open(Path(get_base_path()) / "color_buckets.pkl", "rb") as f:
        color_map = pickle.load(f)
        return list(color_map.values())

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

    X = expanded_df[color_names].values
    y = expanded_df['genre'].values

    # scale the color-diffs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Label-Encoding for genre (get numeric)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()