import pandas as pd
import pickle

from matplotlib import pyplot as plt

VERBOSE = False

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
        subgenre_rows = df[df['Subgenres'].str.contains(subgenre) & ~df['Subgenres'].str.contains('Hard ' + subgenre)].copy()


    # get the unique colors from the current subgenre, only get the first color
    subgenre_rows['Most dominant Color'] = df['Dominant Colors'].str.split(',').str[0]

    # count the colors
    for color in subgenre_rows['Most dominant Color']:
        # skip the entry if the color is not in the color buckets or if the color is 'N/A'
        try :
            subgenre_color_histogram[subgenre][color] += 1
        except KeyError:
            if VERBOSE:
                print('N/A: skipped')
            pass

    # count the total number of releases
    total_releases = len(subgenre_rows)
    print(f"{subgenre}: {total_releases}")
    print(subgenre_color_histogram[subgenre])
    print("")

# plot the six histograms in one figure
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
fig.suptitle('Color Histograms of Electronic Subgenres')

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




