from typing import Any

import polars as pl

from model.base import BaseRecommender


class HeuristicRecommender(BaseRecommender):
    """
    Recommends plants based on simple heuristic filters.
    """

    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate plant recommendations using heuristic filters.

        Args:
            inputs: Dictionary with keys 'light', 'care', 'room', 'flowers', 'toxic'.
            top_k: Number of recommendations to return (default: 3).

        Returns:
            Polars DataFrame containing recommended plants.
        """
        light = inputs.get("light")
        care = inputs.get("care")
        room = inputs.get("room")

        filtered_df = self.data

        light_map = {
            "South": (1.0, 2.0),
            "East": (1.5, 3.0),
            "West": (1.5, 3.0),
            "North": (2.5, 4.0),
            "Grow Light": (1.0, 4.0),
        }
        if light == "North":
            filtered_df = filtered_df.filter(pl.col("light_level") >= 3.0)
        elif light == "South":
            filtered_df = filtered_df.filter(pl.col("light_level") <= 2.0)
        elif light == "Grow Light":
            filtered_df = filtered_df.filter(pl.col("light_level") <= 2.5)

        care_map = {"Low": (2.0, 3.0), "Medium": (1.5, 2.5), "High": (1.0, 3.0)}
        if care == "Low":
            filtered_df = filtered_df.filter(pl.col("water_need") >= 2)
        elif care == "Medium":
            filtered_df = filtered_df.filter(pl.col("water_need") >= 1)
        # High care includes everything

        if inputs.get("room") == "Bathroom":
            filtered_df = filtered_df.filter(pl.col("humidity_need") >= 2.0)

        flowers = inputs.get("flowers")
        if flowers == "Yes":
            filtered_df = filtered_df.filter(pl.col("has_flowers") == 1)
        elif flowers == "No":
            filtered_df = filtered_df.filter(pl.col("has_flowers") == 0)

        toxic = inputs.get("toxic")
        if toxic == "No":
            filtered_df = filtered_df.filter(pl.col("is_toxic") == 0)

        # If we have more than top_k, sample randomly or take top
        if filtered_df.height > top_k:
            return filtered_df.sample(top_k)

        return filtered_df
