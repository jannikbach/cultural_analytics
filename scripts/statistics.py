import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu


@dataclass(frozen=True)
class Config:
    base_path: Path = Path("../")
    data_path: Path = Path("./.fetched_data/")
    csv: Path = data_path / "discogs_with_colors.csv"
    subgenres = ['House','Techno', 'Trance']
    prefix = 'Hard '
    parameters = ['saturation', 'value']

def get_conditions(df, styles):
    """Get conditions for np.select based on unique values in dataframe"""
    conditions = [df['Subgenres'].apply(lambda x: style in x) for style in styles]
    return conditions

def return_defining_style(df_subgenre):
    """Return the defining style of a release"""
    # if df subgenre starts with techno print
    if df_subgenre.startswith('Techno'):
        print(df_subgenre)

    for style in df_subgenre.split(','):
        for genre in config.subgenres:
            # if style contains (config.prefix + genre) print
            if (config.prefix + genre) in style:
                print(config.prefix + genre)
                return config.prefix + genre


    for style in df_subgenre.split(','):
        for genre in config.subgenres:
            if genre in style:
                print(genre)
                return genre

    print('N/A')
    return 'N/A'


config = Config()

def ascii_histogram(row, max_width=50, bar_char='█'):
    """
    Prints an ASCII histogram for the given row.

    Parameters:
    - row (pd.Series): The row from the DataFrame.
    - max_width (int): The maximum width of the histogram in characters.
    - bar_char (str): The character to use for the histogram bars.
    """
    # Extract the values and column names
    values = row.values
    labels = row.index.tolist()

    # Determine the scaling factor based on the maximum value
    max_value = max(values)
    scale = max_width / max_value if max_value > 0 else 1

    # Print each label with its corresponding bar
    for label, value in zip(labels, values):
        bar_length = int(math.floor(value * scale))
        bar = bar_char * bar_length
        print(f"{label:12}: {bar} ({value})")
def main():
    """Main processing pipeline"""

    df = pd.read_csv(config.csv, sep=",", quotechar='"')
    df = df.dropna(subset=config.parameters)
    df['Defining Style']=df['Subgenres'].apply(return_defining_style)

    results = []

    for param in config.parameters:
        for genre in config.subgenres:
            style = df[df['Defining Style'] == genre][param]
            hard_style = df[df['Defining Style'] == (config.prefix + genre)][param]
            if len(style) < 2 or len(hard_style) < 2:
                print(f"Skipping {genre} vs {config.prefix + genre} for {param}: insufficient data")
                continue

            # Check normality (Shapiro-Wilk)
            _, p1 = shapiro(style)
            _, p2 = shapiro(hard_style)
            normal_data = (p1 > 0.05) and (p2 > 0.05)  # Null hypothesis: data is normal

            # Check homogeneity of variances (Levene’s test)
            if normal_data:
                _, p_levene = levene(style, hard_style)
                equal_var = p_levene > 0.05  # Null hypothesis: equal variances


            # Choose test
            if normal_data:
                test_name = "t-test"
                stat, p = ttest_ind(style, hard_style, equal_var=equal_var)
            else:
                test_name = "Mann-Whitney U"
                stat, p = mannwhitneyu(style, hard_style, alternative='two-sided')

            # Calculate effect size
            if test_name == "t-test":
                pooled_std = np.sqrt(((len(style) - 1) * style.var() + (len(hard_style) - 1) * hard_style.var()) / (
                            len(style) + len(hard_style) - 2))
                cohen_d = abs((style.mean() - hard_style.mean()) / pooled_std)
                effect_size = f"Cohen's d = {cohen_d:.2f}"
            else:
                n1, n2 = len(style), len(hard_style)
                rank_biserial = 1 - (2 * stat) / (n1 * n2)
                effect_size = f"Rank-biserial r = {rank_biserial:.2f}"

            # Append results
            results.append({
                'Parameter': param,
                'Comparison': f"{genre} vs {(config.prefix + genre)}",
                'Test': test_name,
                'Statistic': stat,
                'p-value': p,
                'Effect Size': effect_size,
                'Significant (α=0.0083)': p < 0.0083  # Bonferroni-adjusted threshold
            })

    results_df = pd.DataFrame(results)
    print(results)
    print(results_df.head(6))





if __name__ == "__main__":
    main()