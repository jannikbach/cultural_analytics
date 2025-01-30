import pickle
import numpy as np

# Constants
NUM_BINS = 256
COLOR_BIT_DEPTH = 8
BIN_WIDTH = 1  # 2**COLOR_BIT_DEPTH // NUM_BINS

# Color mappings
HUE_TO_COLOR = {
    0: "Red",
    30: "Orange",
    60: "Yellow",
    90: "Chartreuse",
    120: "Green",
    150: "Spring Green",
    180: "Cyan",
    210: "Azure",
    240: "Blue",
    270: "Violet",
    300: "Magenta",
    330: "Rose",
    360: "Red"
}

COLOR_BUCKETS = {
    0: "Red",
    1: "Orange",
    2: "Yellow",
    3: "Chartreuse",
    4: "Green",
    5: "Spring Green",
    6: "Cyan",
    7: "Azure",
    8: "Blue",
    9: "Violet",
    10: "Magenta",
    11: "Rose",
    12: "Black",
    13: "White",
    14: "Gray"
}

# Pre-calculated values
SATURATION_THRESHOLD = 2 ** COLOR_BIT_DEPTH // 10
VALUE_THIRD = (2 ** COLOR_BIT_DEPTH) / 3
VALUE_TWO_THIRDS = 2 * VALUE_THIRD


def create_rgb_lut():
    """Generate RGB lookup table using vectorized operations."""
    indices = np.indices((NUM_BINS, NUM_BINS, NUM_BINS))
    return np.moveaxis(indices * BIN_WIDTH, 0, -1).astype(np.float32)


def rgb_to_hsv(rgb):
    """Convert RGB array to HSV color space."""
    rgb_normalized = rgb.astype(np.float32) / 255.0

    cmax = np.max(rgb_normalized, axis=-1)
    cmin = np.min(rgb_normalized, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    saturation = np.zeros_like(cmax)

    # Hue calculation
    mask = delta != 0
    r, g, b = np.moveaxis(rgb_normalized, -1, 0)

    red_mask = mask & (cmax == r)
    green_mask = mask & (cmax == g)
    blue_mask = mask & (cmax == b)

    hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = ((b[green_mask] - r[green_mask]) / delta[green_mask]) + 2
    hue[blue_mask] = ((r[blue_mask] - g[blue_mask]) / delta[blue_mask]) + 4

    hue_degrees = hue * 60.0
    saturation = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax != 0)

    return np.stack([
        hue_degrees,
        saturation * 255,
        cmax * 255
    ], axis=-1)


def create_hue_map(hsv_data):
    """Generate hue classification map from HSV data."""
    hue_map = np.zeros(hsv_data.shape[:3], dtype=np.uint8)
    h, s, v = np.moveaxis(hsv_data, -1, 0)

    # Calculate masks
    low_sat_mask = s < SATURATION_THRESHOLD
    black_mask = low_sat_mask & (v < VALUE_THIRD)
    white_mask = low_sat_mask & (v > VALUE_TWO_THIRDS)
    gray_mask = low_sat_mask & ~black_mask & ~white_mask

    # Handle chromatic colors
    hue_rounded = np.round(h / 30) * 30
    hue_buckets = ((hue_rounded % 360) // 30).astype(np.uint8)

    # Apply masks
    hue_map[black_mask] = 12
    hue_map[white_mask] = 13
    hue_map[gray_mask] = 14
    hue_map[~low_sat_mask] = hue_buckets[~low_sat_mask]

    return hue_map


def save_data(hue_map, sat_map, val_map):
    """Save generated lookup tables."""
    np.save("../lut_hue.npy", hue_map)
    np.save("../lut_sat.npy", sat_map)
    np.save("../lut_val.npy", val_map)

    with open("../color_buckets.pkl", "wb") as f:
        pickle.dump(COLOR_BUCKETS, f)


def test_color_mapping(lut_rgb, lut_hsv, hue_map, test_cases):
    """Test color classification for given test cases."""
    for (r, g, b), name in test_cases:
        print(f"{name}:", (r, g, b))
        print("RGB:", lut_rgb[r, g, b])
        print("HSV:", lut_hsv[r, g, b])
        print("Color by Map:", COLOR_BUCKETS[hue_map[r, g, b]])
        print()


# Main execution
if __name__ == "__main__":
    # Generate LUTs
    lut_rgb = create_rgb_lut()
    lut_hsv = rgb_to_hsv(lut_rgb)

    # Create classification maps
    hue_map = create_hue_map(lut_hsv)
    sat_map = lut_hsv[..., 1].astype(np.uint8)
    val_map = lut_hsv[..., 2].astype(np.uint8)

    # Test cases
    test_cases = [
        ((0, 0, 31), "Green"),
        ((0, 20, 0), "Blue"),
        ((15, 0, 0), "Red"),
        ((15, 15, 0), "Yellow"),
        ((0, 0, 0), "Black"),
        ((31, 31, 31), "White"),
        ((12, 12, 12), "Gray"),
        ((255, 68, 0), "Should Be Orange")
    ]

    test_color_mapping(lut_rgb, lut_hsv, hue_map, test_cases)
    save_data(hue_map, sat_map, val_map)