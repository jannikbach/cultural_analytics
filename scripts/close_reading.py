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

        ####
        # Split into artists and song parts, then extract artists
        def get_artists(origin):
            artists = (
                origin['Release Name']
                .str.split(' - ', n=1).str[0]  # Split on hyphen and keep the artist part
                .str.split(
                    r',|[fF][eE][aA][tT]\.*|[fF][tT]\.*|&|[Vv][Ss]|[fF][eE][aA][tT][uU][rR][iI][nN][gG]')  # Split on commas with optional spaces
                .explode()  # Break list of artists into separate rows
                .str.strip()  # Remove any leading/trailing whitespace
                .str.strip('*')  # Remove any trailing asterisks
            )

            # Convert to a set of unique artists
            artists_set = set(artists.unique())
            if '' in artists_set:
                artists_set.remove('')  # Remove empty strings
            if 'Various' in artists_set:
                artists_set.remove('Various')  # Remove 'Various' artists
            if 'Unknown Artist' in artists_set:
                artists_set.remove('Unknown Artist')  # Remove 'Various Artists' artists
            if 'S' in artists_set:
                artists_set.remove('S')  # Remove 'S' artists
            return artists_set

        artists = get_artists(style)
        hard_artists = get_artists(hard_style)

        shared_artists = shared_artists.union(artists.intersection(hard_artists))

    ###
    # Filter the dataframe for shared artists
    # Split to get the artist part (before the hyphen)
    artist_part = df['Release Name'].str.split('-', n=1).str[0]

    # make shared artists a regex with subsequent optional white spaces or comma
    # Define prefix and suffix
    prefix = r"\b(?=\w)" # start of word
    suffix = r"\b(?<=\w)" # end of word

    # Modify strings and escape special characters
    escaped_artists = [re.escape(s) for s in shared_artists]
    escaped_pattern = [prefix + s + suffix for s in escaped_artists]
    # Escape special characters in shared_artists and join into a regex pattern
    pattern = '|'.join(escaped_pattern)
    regex = re.compile(pattern)

    # Check if any artist in shared_artists appears as a substring in artist_part
    mask = artist_part.str.contains(regex, regex=True)

    # Filter the DataFrame
    genre_fluid_artist_tracks = df[mask]
    print('Shared artists:', genre_fluid_artist_tracks)



if __name__ == "__main__":
    main()
