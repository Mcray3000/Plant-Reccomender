from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from model.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Recommends plants using Content-Based Filtering (Cosine Similarity).
    """

    def __init__(self, data: pl.DataFrame):
        super().__init__(data)
        self.features = ["light_level", "water_need", "humidity_need", "temp_tolerance"]
        self.X = self.data.select(self.features).to_numpy()
        self.sim_matrix = cosine_similarity(self.X)

    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate plant recommendations using content-based filtering.

        Args:
            inputs: Dictionary with keys 'plants' (list of plant names), 'flowers', 'toxic'.
            top_k: Number of recommendations to return (default: 3).

        Returns:
            Polars DataFrame containing recommended plants.
        """
        plant_names = inputs.get("plants", [])
        if not plant_names:
            return pl.DataFrame()

        # Find indices of user's plants
        # Using lowercase matching
        df_lower = self.data.with_columns(
            [
                pl.col("common_name").str.to_lowercase().alias("common_lower"),
                pl.col("scientific_name").str.to_lowercase().alias("sci_lower"),
            ]
        )

        indices = []
        for p in plant_names:
            # Find index
            p_lower = p.lower()
            matches = df_lower.with_row_index().filter(
                (pl.col("common_lower").str.contains(p_lower)) | (pl.col("sci_lower").str.contains(p_lower))
            )
            if matches.height > 0:
                indices.append(matches["index"][0])

        if not indices:
            print(f"Warning: No matching plants found for {plant_names}")
            return pl.DataFrame()

        # Sum similarity vectors for all user plants
        user_sim_vector = np.sum(self.sim_matrix[indices], axis=0)

        # Get top K indices (excluding user's plants)
        # Sort by similarity (descending)
        sorted_indices = np.argsort(user_sim_vector)[::-1]

        recommended_indices = []

        # Create a list of all candidate indices (excluding user's plants)
        candidate_indices = [idx for idx in sorted_indices if idx not in indices]

        # Get the candidate dataframe in order
        candidates_df = self.data[candidate_indices]

        # Flowers Filter
        flowers = inputs.get("flowers")
        if flowers == "Yes":
            candidates_df = candidates_df.filter(pl.col("has_flowers") == 1)
        elif flowers == "No":
            candidates_df = candidates_df.filter(pl.col("has_flowers") == 0)

        # Toxic Filter
        toxic = inputs.get("toxic")
        if toxic == "No":
            candidates_df = candidates_df.filter(pl.col("is_toxic") == 0)

        return candidates_df.head(top_k)
