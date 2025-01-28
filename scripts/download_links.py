import random
import discogs_client
import csv
from pathlib import Path
import dotenv
import os
from tqdm import tqdm

dotenv.load_dotenv()
api_key = os.getenv('DISCOGS_API_KEY')

# Initialize Discogs client
d = discogs_client.Client('album-cover-project', user_token=api_key)

genre = 'Electronic'
styles = ['House', 'Hard House', 'Techno', 'Hard Techno', 'Trance', 'Hard Trance']

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
