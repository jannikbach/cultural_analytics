from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


VERBOSE = False
LAME_PLOT = False

df = pd.read_csv('.fetched_data/discogs_with_colors.csv', sep=",", quotechar='"')

file = open("../color_buckets.pkl", 'rb')
color_buckets = pickle.load(file)
file.close()

# get the unique Subgenres from the dataset
subgenres = ['House', 'Hard House', 'Techno', 'Hard Techno', 'Trance', 'Hard Trance']

# create a nested dict with the unique subgenres as keys and the color histogram as values
subgenre_color_histogram = {}
for subgenre in subgenres:
    # create a color histogram
    subgenre_color_histogram[subgenre] = {color: 0 for _, color in color_buckets.items()}

    # get the rows that contain the current subgenre if it has the hard prefix
    # if the hard prefix is missing just get the rows that contain the subgenre but not the hard prefix
    if 'Hard' in subgenre:
        subgenre_rows = df[df['Subgenres'].str.contains(subgenre)].copy()
    else:
        subgenre_rows = df[
            df['Subgenres'].str.contains(subgenre) & ~df['Subgenres'].str.contains('Hard ' + subgenre)].copy()

    # get the unique colors from the current subgenre, only get the first color
    subgenre_rows['Most dominant Color'] = df['Dominant Colors'].str.split(',').str[0]

    # count the colors
    for color in subgenre_rows['Most dominant Color']:
        # skip the entry if the color is not in the color buckets or if the color is 'N/A'
        try:
            subgenre_color_histogram[subgenre][color] += 1
        except KeyError:
            if VERBOSE:
                print('N/A: skipped')
            pass

    # count the total number of releases
    if VERBOSE:
        total_releases = len(subgenre_rows)
        print(f"{subgenre}: {total_releases}")
        print(subgenre_color_histogram[subgenre])
        print("")

plt_color_map = {
    'Red': 'red',
    'Orange': 'orange',
    'Yellow': 'yellow',
    'Chartreuse': 'chartreuse',
    'Green': 'green',
    'Spring Green': 'springgreen',
    'Cyan': 'cyan',
    'Azure': 'deepskyblue',  # No direct "azure", closest match
    'Blue': 'blue',
    'Violet': 'blueviolet',  # Closest match to violet
    'Magenta': 'magenta',
    'Rose': 'deeppink',  # No direct "rose", closest match
    'Black': 'black',
    'White': 'white',
    'Gray': 'gray'
}

if LAME_PLOT:
    # plot the six histograms in one figure
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Color Histograms of Electronic Subgenres')

    for i, subgenre in enumerate(subgenres):
        plt_colors = []
        ax = axs[i // 2, i % 2]
        colors = list(subgenre_color_histogram[subgenre].keys())

        for color in colors:
            plt_colors.append(plt_color_map[color])  # Map colors using your color mapping

        counts = list(subgenre_color_histogram[subgenre].values())

        # Normalize counts
        max_count = max(counts)  # Get max count for normalization
        normalized_counts = [count / max_count for count in counts]  # Normalize counts

        # Plot bars and apply an outline for white
        for j, (color, norm_count) in enumerate(zip(colors, normalized_counts)):
            if color == 'White':
                ax.bar(j, norm_count, color='white', edgecolor='black', linewidth=1.5)  # Black outline for white
            else:
                ax.bar(j, norm_count, color=plt_color_map[color])  # Normal color for others

        ax.set_title(subgenre)
        # ax.set_xlabel('Color')
        ax.set_xticks(range(len(colors)))
        ax.set_xticklabels(colors, rotation=45)

        # Normalize y-axis and set specific ticks
        ax.set_ylim(0, 1.1)  # Ensure y-axis runs from 0 to 1
        ax.set_yticks([0.25, 0.5, 0.75, 1])

    plt.tight_layout()
    plt.show()


def create_hist_axes(fig, ax_position, lw_bars, lw_grid, lw_border, histogram):
    """
    Create a polar Axes containing the histogram radar plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw into.
    ax_position : (float, float, float, float)
        The position of the created Axes in figure coordinates as
        (x, y, width, height).
    lw_bars : float
        The linewidth of the bars.
    lw_grid : float
        The linewidth of the grid.
    lw_border : float
        The linewidth of the Axes border.
    histogram : dict
        The histogram to plot. The keys are the colors and the values are the
        frequencies.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The created Axes.
    """
    with plt.rc_context({'axes.edgecolor': '#111111',
                         'axes.linewidth': lw_border}):
        ax = fig.add_axes(ax_position, projection='polar')
        ax.set_axisbelow(True)

        N = len(histogram)
        arc = 2. * np.pi
        theta = np.arange(0.0, arc, arc / N)
        radii_value = np.array(list(histogram.values()))
        # take the square root of the radii to make the plot more readable
        # this matches values of the histogram with the area of the bars
        radii = np.sqrt(radii_value)
        # normalize the radii
        radii = radii / np.max(radii)

        colors = list(histogram.keys())

        # Function to convert color to pastel by blending with white
        def pastel_color(color, mix=0.5):
            c = plt_color_map[color]
            rgb = np.array(mcolors.to_rgb(c))  # Convert to RGB
            white = np.array([1, 1, 1])  # White color in RGB
            return (1 - mix) * rgb + mix * white  # Blend with white

        # Create pastel colors
        pastel_colors = [pastel_color(c, mix=0.35) for c in colors]
        colors = pastel_colors

        width = 2 * np.pi / N * np.ones(N)
        bars = ax.bar(theta, radii, width=width, bottom=0.0, align='edge',
                      edgecolor='0.3', lw=lw_bars)
        for color, bar in zip(colors, bars):
            color_plt = color  # map to matplotlib color
            bar.set_facecolor(color_plt)

        ax.tick_params(labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

        ax.grid(lw=lw_grid, color='0.9')
        ax.set_rmax(1.05)
        y_ticks = np.sqrt(np.linspace(0, 1, 5)[1:])
        ax.set_yticks(y_ticks)

        # the actual visible background - extends a bit beyond the axis
        ax.add_patch(Rectangle((0, 0), arc, 9.58,
                               facecolor='white', zorder=0,
                               clip_on=False, in_layout=False))
        return ax


def make_circle_histogram(height_px, lw_bars, lw_grid, lw_border, historagm, title):
    """
    Create a full figure with the Color Circle Histogram. (A Histogram in polar coordinates)

    Parameters
    ----------
    height_px : int
        Height of the figure in pixel.
    lw_bars : float
        The linewidth of the bar border.
    lw_grid : float
        The linewidth of the grid.
    lw_border : float
        The linewidth of icon border.
    histogram : dict
            The histogram to plot. The keys are the colors and the values are the
            frequencies.
    title : str
        The title of the plot.
    """
    dpi = 100
    height = height_px / dpi
    figsize = (height, height * 1.2)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0)
    fig.suptitle(title)

    ax_pos = (0.03, 0.03, .94, .94)
    ax = create_hist_axes(fig, ax_pos, lw_bars, lw_grid, lw_border, historagm)

    return fig, ax


file_types = ['svg', 'png']

for file_type in file_types:
    Path(f'../figures/{file_type}').mkdir(parents=True, exist_ok=True)

for genre, hist in subgenre_color_histogram.items():
    fig, ax = make_circle_histogram(height_px=500, lw_bars=0.7, lw_grid=0.5, lw_border=1,
                                    historagm=hist, title=genre)
    for file_type in file_types:
        plt.savefig(f'../figures/{file_type}/{genre.lower().replace(" ", "_")}_radial_hist.{file_type}')
    if VERBOSE:
        plt.show()
