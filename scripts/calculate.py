from download_links import load_releases
from create_color_bin_LUT import create_buckets
from add_dominant_colors import add_dominant_colors

ANALYZE_HARDS = False

def calculate_hards():
    styles = ['House', 'Hard House', 'Techno', 'Hard Techno', 'Trance', 'Hard Trance']
    load_releases(styles)
    create_buckets()
    add_dominant_colors()
    
def calculate_all():
    load_releases()
    create_buckets()
    add_dominant_colors()

if __name__ == "__main__":
    if ANALYZE_HARDS:
        calculate_hards()
    else:
        calculate_all()