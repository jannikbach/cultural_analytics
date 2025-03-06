import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_base_path, HARD_STYLES_WITHOUT_HARD


@dataclass(frozen=True)
class Config:
    base_path: Path = Path(get_base_path())
    data_path: Path = Path("./.fetched_data/")
    csv: Path = data_path / "discogs_with_colors.csv"
    subgenres = HARD_STYLES_WITHOUT_HARD
    prefix = 'Hard '
    file_types = ['svg', 'png']
    artist_seperator_regex = [r",", r"&", r"vs", r"feat", r"ft", r"featuring"]


def return_defining_style(df_subgenre):
    """Return the defining style of a release"""

    for style in df_subgenre.split(','):
        for genre in config.subgenres:
            # if style contains (config.prefix + genre) print
            if (config.prefix + genre) in style:
                return config.prefix + genre

    for style in df_subgenre.split(','):
        for genre in config.subgenres:
            if genre in style:
                return genre

    return 'N/A'


def split_artists(s):
    artists = [s]
    for separator in config.artist_seperator_regex:
        for artist in artists:
            if re.search(separator, artist, re.IGNORECASE):
                artists.remove(artist)
                text = re.sub(r"[^a-zA-Z0-9\s(),&]", '', artist)
                result = re.split(separator, text, flags=re.IGNORECASE)
                for substring in result:
                    # remove leading and trailing whitespaces
                    artists.append(substring.strip())
    return artists


config = Config()


def main():
    """Main processing pipeline"""

    df = pd.read_csv(config.csv, sep=",", quotechar='"')
    df['Defining Style'] = df['Subgenres'].apply(return_defining_style)

    # Algorithm:
    # For a subgenre and its hard companion create a set of all artist
    # look for artists in both sets
    # create a df of all the releases of the artists in both sets

    shared_artists = set()

    for genre in config.subgenres:
        style = df[df['Defining Style'] == genre]
        hard_style = df[df['Defining Style'] == (config.prefix + genre)]

        artists = set()
        hard_artists = set()

        def get_artists(origin, set_to_add):
            release = origin['Release Name']
            release = release.apply(lambda x: x.split(' - ')[0])

            def apply_artists_to_set(x):
                artists = split_artists(x)
                for artist in artists:
                    set_to_add.add(artist)

            release.apply(apply_artists_to_set)

        get_artists(style, artists)
        get_artists(hard_style, hard_artists)

        shared_artists = artists.intersection(hard_artists)

        print(shared_artists)


if __name__ == "__main__":
    main()
