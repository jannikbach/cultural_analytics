import pickle
import numpy as np
from utils import get_base_path

def rgb_to_hsv(rgb):
    """
        rgb: NumPy array of shape (a, a, a, 3), with dtype in [0..255].
        returns: HSV array of shape (a, a, a, 3) where:
            HSV[..., 0] (Hue) is in [0..360]
            HSV[..., 1] (Saturation) is in [0..255]
            HSV[..., 2] (Value) is in [0..255]
        """

    # Convert to float in [0..1]
    rgb_float = rgb.astype(np.float32) / 255.0

    # Separate R, G, B channels
    r = rgb_float[..., 0]
    g = rgb_float[..., 1]
    b = rgb_float[..., 2]

    # Compute cmax, cmin, and delta
    cmax = np.max(rgb_float, axis=-1)
    cmin = np.min(rgb_float, axis=-1)
    delta = cmax - cmin

    # Initialize Hue, Saturation, Value arrays
    hue = np.zeros_like(cmax)
    sat = np.zeros_like(cmax)
    val = cmax  # Value is the max channel

    # Mask where delta != 0 (to avoid division by zero)
    mask_delta = (delta != 0)

    # ----- HUE calculation -----
    # if R is max => Hue = (G - B) / delta % 6
    # if G is max => Hue = (B - R) / delta + 2
    # if B is max => Hue = (R - G) / delta + 4

    idx = (cmax == r) & mask_delta
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6

    idx = (cmax == g) & mask_delta
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2

    idx = (cmax == b) & mask_delta
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4

    # Now hue is in [0..6], convert to [0..360]
    hue_degrees = hue * 60.0

    # ----- SATURATION calculation -----
    # Sat = delta / cmax (but if cmax = 0 => sat = 0)
    mask_cmax = (cmax != 0)
    sat[mask_cmax] = delta[mask_cmax] / cmax[mask_cmax]

    # Convert saturation and value to [0..255]
    sat_255 = sat * 255.0
    val_255 = val * 255.0

    # Finally stack them into an HSV array of shape (a,a,a,3)
    # Hue in [0..360], Sat in [0..255], Val in [0..255]
    hsv = np.stack([hue_degrees, sat_255, val_255], axis=-1)

    return hsv

def create_buckets():
    # Create a 3D LUT
    num_bins = 256
    color_bit_depth = 8
    bin_width = 1#2**color_bit_depth // num_bins

    # Initialize the LUT: (num_bins, num_bins, num_bins, 3)
    lut_rgb = np.zeros((num_bins, num_bins, num_bins, 3), dtype=np.float32)
    lut_hsv = np.zeros((num_bins, num_bins, num_bins, 3), dtype=np.float32)


    # Populate the LUT
    for r_idx in range(num_bins):
        for g_idx in range(num_bins):
            for b_idx in range(num_bins):
                # OpenCV expects BGR order
                lut_rgb[r_idx,g_idx,b_idx] = np.array([r_idx, g_idx, b_idx], dtype=np.uint8) * bin_width

    lut_hsv = rgb_to_hsv(lut_rgb)

    hue_to_color = {
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
        360: "Red"  # 360° wraps around to Red
    }
    color_buckets = {
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
    color_to_bucket = {color: bucket for bucket, color in color_buckets.items()}

    saturation_threshold = 2 ** color_bit_depth // 10 #colors with less saturation than this will be considered gray
    hue_map = np.zeros((num_bins, num_bins, num_bins), dtype=np.uint8)
    for index in np.ndindex(hue_map.shape):
        r, g, b = index
        hsv = lut_hsv[r, g, b]
        if hsv[1] < saturation_threshold:
            if hsv[2] < (2**color_bit_depth / 3):
                hue_map[r, g, b] = 12 # black
            elif hsv[2] > (2*(2**color_bit_depth) / 3):
                hue_map[r, g, b] = 13 # white
            else:
                hue_map[r, g, b] = 14 # gray
        else:
            # find the closest hue
            closest_key = min(hue_to_color.keys(), key=lambda k: abs(k - hsv[0]))
            closest_hue = hue_to_color[closest_key]
            bucket = color_to_bucket.get(closest_hue)
            hue_map[r, g, b] = bucket

    sat_map = np.zeros((num_bins, num_bins, num_bins), dtype=np.uint8)
    for index in np.ndindex(sat_map.shape):
        r, g, b = index
        hsv = lut_hsv[r, g, b]
        sat_map[r, g, b] = hsv[1]

    val_map = np.zeros((num_bins, num_bins, num_bins), dtype=np.uint8)
    for index in np.ndindex(val_map.shape):
        r, g, b = index
        hsv = lut_hsv[r, g, b]
        val_map[r, g, b] = hsv[2]


    # Test:
    r, g, b = 0, 0, 31
    print("Green:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")

    r, g, b = 0, 20, 0
    print("Blue:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")


    r, g, b = 15, 0, 0
    print("Red:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")


    r, g, b = 15, 15, 0
    print("Yellow:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")

    r, g, b = 0, 0, 0
    print("Black:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")

    r, g, b = 31, 31, 31
    print("White:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")

    r, g, b = 12, 12, 12
    print("Gray:", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")

    r, g, b = 255, 68, 0
    print("Should Be Orange not Red (Hue 16):", (r, g, b))
    print("RGB:", lut_rgb[r, g, b])
    print("HSV:", lut_hsv[r, g, b])
    print("Color by Map :", color_buckets[hue_map[r, g, b]])
    print("")


    # Save the LUTs
    np.save(f"{get_base_path}lut_hue.npy", hue_map)
    np.save(f"{get_base_path}lut_sat.npy", sat_map)
    np.save(f"{get_base_path}lut_val.npy", val_map)

    # Save to 'color_buckets.pkl'
    with open(f"{get_base_path}color_buckets.pkl", "wb") as f:   # note 'wb' for write-binary
        pickle.dump(color_buckets, f)

if __name__ == "__main__":
    create_buckets()