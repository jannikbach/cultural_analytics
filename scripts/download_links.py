import random
import discogs_client
import csv
from pathlib import Path
import discogs_client.models
import dotenv
import os
from tqdm import tqdm
import pandas as pd
from utils import TOP_STYLES

dotenv.load_dotenv()

def load_releases(styles:list[str]):
    api_key = os.getenv('DISCOGS_API_KEY')

    # Initialize Discogs client
    d = discogs_client.Client('album-cover-project', user_token=api_key)

    genre = 'Electronic'

    styles_count = {}

    for style in styles:
        styles_count[style] = d.search(genre=genre, style=style, type='release')

    print('aggregate releases')
    # extract 20 random pages from each style and put them in a list
    all_sampled_releases = {}
    num_random_pages = 20
    for key, value in styles_count.items():
        random_page_indices = random.sample(range(1, value.pages + 1), k=num_random_pages)

        all_sampled_releases[key] = []
        for page_number in random_page_indices:
            page_data = value.page(page_number)
            releases_list = list(page_data)
            all_sampled_releases[key] = all_sampled_releases[key] + releases_list

    Path('.fetched_data').mkdir(parents=True, exist_ok=True)

    # Write headers
    with open('.fetched_data/discogs_releases.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Release Name', 'Genre', 'Subgenres', 'Cover URL'])

    for key, value in all_sampled_releases.items():
        # Open the file once for writing the actual data
        with open('.fetched_data/discogs_releases.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Create one progress bar to track all releases
            with tqdm(total=len(value), desc=str(key)) as pbar:
                for release in value:
                    try:
                        release_name = release.title
                        genres = ', '.join(release.genres) if release.genres else 'N/A'
                        subgenres = ', '.join(release.styles) if release.styles else 'N/A'
                        cover_url = release.images[0]['uri'] if release.images else 'N/A'
                        writer.writerow([release.id, release_name, genres, subgenres, cover_url])
                    except discogs_client.exceptions.HTTPError as e:
                        # just skip the entry
                        # if e.status_code != 404:
                        #     raise
                        print(f'Not found while loading release {release.id}')

                    # Update progress bar after each release
                    pbar.update(1)


def load_releases():
    api_key = os.getenv('DISCOGS_API_KEY')

    # Initialize Discogs client
    d = discogs_client.Client('album-cover-project', user_token=api_key)

    genre = 'Electronic'

    results = d.search(genre=genre, type='release')

    print('aggregate releases')
    # extract 20 random pages from each style and put them in a list
    all_releases = []
    num_random_pages = 20

    random_page_indices = random.sample(range(1, results.pages + 1), k=num_random_pages)

    for page_number in random_page_indices:
        page_data = results.page(page_number)
        releases_list = list(page_data)
        all_releases = all_releases + releases_list

    Path('.fetched_data').mkdir(parents=True, exist_ok=True)

    # Write headers
    with open('.fetched_data/discogs_releases.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Release Name', 'Genre', 'Subgenres', 'Cover URL'])

    # Open the file once for writing the actual data
    with open('.fetched_data/discogs_releases.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Create one progress bar to track all releases
        with tqdm(total=len(all_releases), desc='releases done') as pbar:
            for release in all_releases:
                try:
                    release_name = release.title
                    genres = ', '.join(release.genres) if release.genres else 'N/A'
                    subgenres = ', '.join(release.styles) if release.styles else 'N/A'
                    cover_url = release.images[0]['uri'] if release.images else 'N/A'
                    writer.writerow([release.id, release_name, genres, subgenres, cover_url])
                except discogs_client.exceptions.HTTPError as e:
                    # just skip the entry
                    # if e.status_code != 404:
                    #     raise
                    print(f'Not found while loading release {release.id}')

                # Update progress bar after each release
                pbar.update(1)

def load_releases_with_different_styles():
    api_key = os.getenv('DISCOGS_API_KEY')

    # Initialize Discogs client
    d = discogs_client.Client('album-cover-project', user_token=api_key)

    results = d.search(type='artist')

    # Path('.fetched_data').mkdir(parents=True, exist_ok=True)

    # with open('.fetched_data/discogs_releases_different.csv', mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Id', 'Release Name', 'Artist', 'Genre', 'Subgenres', 'Cover URL'])

    with tqdm(total=results.count, desc='artists loaded') as pbar:
        df = pd.read_csv('.fetched_data/discogs_releases_different.csv', sep=",", quotechar='"')
        for i in range(results.pages):
            for artist in list(results.page(i)):
                pbar.update(1)

                try:
                    if df["Artist"].eq(artist.name).any():
                        continue
                    releases_to_save = []
                    # releases = artist.releases
                    #releases = d.search(type='release', genre='Electronic', artist=artist.name)

                    for year in range(2015, 2025):
                        for style in TOP_STYLES:
                            releases = d.search(type='release', genre='Electronic', year=str(year), style=style, artist=artist.name)

                            for j in range(releases.pages):
                                for release in list(releases.page(j)):
                                    try:
                                        # if release.year is None or int(release.year) < 2015:
                                        #     continue

                                        # if not 'Electronic' in release.genres:
                                        #     continue

                                        if len(release.styles) != 1:
                                            continue

                                        # if not release.styles[0] in TOP_STYLES:
                                        #     continue

                                        releases_to_save.append(release)
                                    except Exception as e:
                                        print(f"Error in release {release.title}: {e}")

                    styles = {release.styles[0] for release in releases_to_save}

                    with open('.fetched_data/discogs_releases_different.csv', mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)

                        if len(styles) <= 1:
                            writer.writerow(['N/A','N/A',artist.name,'N/A','N/A','N/A'])
                            continue
                        
                        for release in releases_to_save:
                            release_name = release.title
                            genres = ', '.join(release.genres) if release.genres else 'N/A'
                            subgenres = ', '.join(release.styles) if release.styles else 'N/A'
                            cover_url = release.images[0]['uri'] if release.images else 'N/A'
                            writer.writerow([release.id, release_name, artist.name, genres, subgenres, cover_url])
                except Exception as e:
                    print(f"Error in artist {artist.name}: {e}")

if __name__ == "__main__":
    load_releases_with_different_styles()