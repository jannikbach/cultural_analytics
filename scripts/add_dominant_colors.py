import numpy
import pandas as pd
import numpy as np
from skimage import io
import pickle
import requests
from io import BytesIO
from tqdm import tqdm

VERBOSE = False

COLOR_LUT_256 = None
COLOR_MAP_IDX = None


def calculate_dominant_color(row):
    url = row["Cover URL"]

    # load image dynamially
    def load_image_from_url(url):
        # Pretend to be a regular browser
        headers = {"User-Agent": "Mozilla/5.0"}

        # 1. Fetch image data (raise_for_status() ensures an error if not 200 OK)
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 2. Convert to a file-like object
        file_like = BytesIO(response.content)

        # 3. Now read it with skimage
        image = io.imread(file_like)

        return image

    try:
        image = load_image_from_url(url)
    except Exception as e:
        if VERBOSE:
            id = row["Id"]
            print(f"ID {id}: Failed to load {url}: {e}")
        return 'N/A'

    color_map_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # image[i,j] is the rgb value of a pixel in the jpg
            # color_lut_256 is the color of that pixel
            rgb = image[i, j]
            color = COLOR_LUT_256[rgb[0], rgb[1], rgb[2]]
            color_map_image[i, j] = color

    flat = color_map_image.ravel()
    counts = np.bincount(flat)
    top3_colors_idx = np.argsort(counts)[-3:][::-1]
    top3_colors = ",".join(COLOR_MAP_IDX[x] for x in top3_colors_idx)

    return top3_colors

def add_dominant_colors():
    df = pd.read_csv('.fetched_data/discogs_releases.csv', sep=",", quotechar='"')

    COLOR_LUT_256 = numpy.load('../lut_hsv.npy')

    file = open("../color_buckets.pkl", 'rb')
    COLOR_MAP_IDX = pickle.load(file)
    file.close()


    # start
    print("Calculating dominant colors...")
    tqdm.pandas()
    df["dominant_colors"] = df.progress_apply(calculate_dominant_color, axis=1)
    if VERBOSE:
        print("Done!")

    df.to_csv('dicogs_with_colors.csv', index=False)

if __name__ == "__main__":
    add_dominant_colors()
