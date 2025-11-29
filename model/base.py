from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseRecommender(ABC):
    """
    Abstract base class for all plant recommender models.
    """

    def __init__(self, data: pl.DataFrame):
        self.data = data

    @abstractmethod
    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate recommendations based on inputs.

        Args:
            inputs: Dictionary containing user inputs (e.g., {'light': 'North', 'care': 'Low'}
                    or {'plants': ['Monstera', 'Pothos']}).
            top_k: Number of recommendations to return.

        Returns:
            Polars DataFrame containing recommended plants.
        """
        pass
