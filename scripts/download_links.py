import discogs_client
import csv
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('DISCOGS_API_KEY')

d = discogs_client.Client('album-cover-project', user_token=api_key)

genre = 'Electronic'
results = d.search(genre=genre, type='release')

Path('.fetched_data').mkdir(parents=True, exist_ok=True)

with open('.fetched_data/discogs_releases.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'Release Name', 'Genre', 'Subgenres', 'Cover URL'])

for i in range(results.pages):
    j = 0
    for release in results.page(i):
        j += 1
        print(f'processing release {i * results.per_page + j} of {results.pages * results.per_page}...')
        id = release.id
        try:
            release_name = release.title
            genres = ', '.join(release.genres) if release.genres else 'N/A'
            subgenres = ', '.join(release.styles) if release.styles else 'N/A'
            cover_url = release.images[0]['uri'] if release.images else 'N/A'

            with open('.fetched_data/discogs_releases.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([id, release_name, genres, subgenres, cover_url])
        except discogs_client.exceptions.HTTPError as exception:
            if exception.status_code != 404:
                raise
            print(f'not found while loading release {id}')

        
    
    if i == 3:
        break
