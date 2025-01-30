from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
import requests
from skimage import io
from tqdm import tqdm
from io import BytesIO
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    base_path: Path = Path("../")
    data_path: Path = Path("./.fetched_data/")
    lut_names: Tuple[str, ...] = ("hue", "sat", "val")
    input_csv: Path = data_path / "discogs_releases_6000.csv"
    output_csv: Path = data_path / "discogs_with_colors.csv"


class ImageProcessingError(Exception):
    """Custom exception for image processing failures"""


class ColorAnalyzer:
    """Analyzes color characteristics of images using precomputed LUTs"""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.luts = self._load_luts()
        self.color_map = self._load_color_map()

    def _load_luts(self) -> Dict[str, np.ndarray]:
        """Load lookup tables from numpy files"""
        return {
            name: np.load(self.config.base_path / f"lut_{name}.npy")
            for name in self.config.lut_names
        }

    def _load_color_map(self) -> Dict[int, str]:
        """Load color name mapping using context manager"""
        with open(self.config.base_path / "color_buckets.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_image(url: str) -> Optional[np.ndarray]:
        """Fetch image from URL with proper error handling"""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            response.raise_for_status()
            return io.imread(BytesIO(response.content))
        except Exception as e:
            logger.debug(f"Image load failed for {url}: {str(e)}")
            return None

    def analyze(self, image: np.ndarray) -> Dict[str, float | str]:
        """Analyze image and return color metrics"""
        try:
            color_map = self.luts["hue"][image[..., 0], image[..., 1], image[..., 2]]
            sat_map = self.luts["sat"][image[..., 0], image[..., 1], image[..., 2]]
            val_map = self.luts["val"][image[..., 0], image[..., 1], image[..., 2]]

            return {
                "colors": self._top_colors(color_map),
                "saturation": sat_map.mean(),
                "value": val_map.mean()
            }
        except IndexError as e:
            raise ImageProcessingError(f"Invalid image dimensions: {e}") from e

    def _top_colors(self, color_map: np.ndarray) -> str:
        """Extract top 3 colors from color map"""
        counts = np.bincount(color_map.ravel())
        top_indices = np.argsort(counts)[-3:][::-1]
        return ",".join(self.color_map[idx] for idx in top_indices)


def process_data(df: pd.DataFrame, analyzer: ColorAnalyzer) -> pd.DataFrame:
    """Process DataFrame with image analysis"""

    def analyze_row(row: pd.Series) -> Dict[str, float | str]:
        if (image := analyzer.load_image(row["Cover URL"])) is not None:
            try:
                return analyzer.analyze(image)
            except ImageProcessingError as e:
                logger.debug(f"Processing failed for {row.name}: {e}")
        return {"colors": "N/A", "saturation": np.nan, "value": np.nan}

    logger.info("Analyzing album covers...")
    tqdm.pandas(desc="Processing images")
    results = df.progress_apply(analyze_row, axis=1, result_type="expand")
    return pd.concat([df, results], axis=1)


def main():
    """Main processing pipeline"""
    config = Config()
    analyzer = ColorAnalyzer(config)

    try:
        df = pd.read_csv(config.input_csv, sep=",", quotechar='"')
        processed_df = process_data(df, analyzer)
        processed_df.to_csv(config.output_csv, index=False)
        logger.info(f"Successfully processed {len(df)} entries")
    except FileNotFoundError as e:
        logger.error(f"File operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()