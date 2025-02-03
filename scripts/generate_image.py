import csv
import requests
from PIL import Image
import random
import io
import pandas as pd
from tqdm import tqdm

# Configuration
CSV_FILE = '.fetched_data/discogs_with_colors.csv'
OUTPUT_FILE = '../figures/collage.jpg'
CANVAS_WIDTH = 6000
CANVAS_HEIGHT = 6000
MIN_SIZE = 500
MAX_SIZE = 1000
IMAGE_COUNT = 400
ROTATION_RANGE = (-30, 30)
BACKGROUND_COLOR = (255, 255, 255)  # White


def process_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert('RGBA')  # Keep alpha channel

        # Resize with aspect ratio preservation
        original_width, original_height = image.size
        max_dimension = max(original_width, original_height)
        if max_dimension == 0:
            return None

        target_size = random.randint(MIN_SIZE, MAX_SIZE)
        scale_factor = target_size / max_dimension
        new_size = (
            int(original_width * scale_factor),
            int(original_height * scale_factor)
        )
        resized_image = image.resize(new_size)

        # Apply random rotation with transparent background
        angle = random.uniform(*ROTATION_RANGE)
        rotated_image = resized_image.rotate(
            angle,
            expand=True,
            fillcolor=(0, 0, 0, 0)  # Transparent fill for expanded areas
        )

        return rotated_image
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

# Read URLs from CSV

df = pd.read_csv(CSV_FILE, sep=",", quotechar='"')
urls = df["Cover URL"].sample(n=IMAGE_COUNT).tolist()
print(f"Loaded {len(urls)} URLs")


# Process all images
processed_images = []
for url in tqdm(urls, desc="Processing images", unit="image"):
    img = process_image(url)
    if img is not None:
        processed_images.append(img)

# Shuffle images for random overlap order
random.shuffle(processed_images)

# Create RGBA canvas with white background
canvas = Image.new('RGBA', (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND_COLOR)

# Position images randomly with transparency
for img in tqdm(processed_images, desc="Placing images", unit="image"):
    img_width, img_height = img.size
    x = random.randint(-img_width, CANVAS_WIDTH)
    y = random.randint(-img_height, CANVAS_HEIGHT)

    # Create temporary canvas for each image to handle transparency
    temp_canvas = Image.new('RGBA', canvas.size)
    temp_canvas.paste(img, (x, y))
    canvas = Image.alpha_composite(canvas, temp_canvas)

# Flatten the image onto a white background before saving as JPEG
background = Image.new('RGB', canvas.size, (255, 255, 255))  # White background
background.paste(canvas, mask=canvas.split()[-1])  # Paste with transparency mask
background.save(OUTPUT_FILE, quality=90)  # Save as JPEG

print(f"Collage saved as {OUTPUT_FILE}")
