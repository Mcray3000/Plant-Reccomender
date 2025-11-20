import os

import polars as pl


class DataLoader:
    """
    Handles loading of plant data from processed parquet files.
    """

    def __init__(self, data_path: str = "processed_data/trustworthy_plants.parquet"):
        self.data_path = data_path
        self._data = None

    def load_data(self) -> pl.DataFrame:
        """
        Loads the plant data from parquet.
        """
        if self._data is None:
            if not os.path.exists(self.data_path):
                # Fallback to absolute path if running from different cwd
                # Assuming standard project structure
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.data_path = os.path.join(base_path, "processed_data", "trustworthy_plants.parquet")

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")

            self._data = pl.read_parquet(self.data_path)

            # Clean has_flowers column
            # Convert to 0/1: If "0" -> 0, else 1
            self._data = self._data.with_columns(
                pl.when(pl.col("has_flowers") == "0").then(0).otherwise(1).alias("has_flowers")
            )

        return self._data
