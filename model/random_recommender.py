from typing import Any

import polars as pl

from model.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """
    Recommends plants randomly from the dataset.
    """

    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate random plant recommendations.

        Args:
            inputs: Dictionary (not used, but kept for API consistency).
            top_k: Number of recommendations to return (default: 3).

        Returns:
            Polars DataFrame containing randomly selected plants.
        """
        # Simply sample random plants from the dataset
        if self.data.height > top_k:
            return self.data.sample(top_k)
        else:
            return self.data
