from typing import Any

import polars as pl
from sklearn.neighbors import NearestNeighbors

from model.base import BaseRecommender


class KNNRecommender(BaseRecommender):
    """
    Recommends plants using K-Nearest Neighbors.
    Finds plants closest to the average feature vector of the user's current plants.
    """

    def __init__(self, data: pl.DataFrame):
        super().__init__(data)
        self.model = NearestNeighbors(n_neighbors=5, algorithm="auto")
        self._train_model()

    def _train_model(self):
        """
        Fits the KNN model on plant features.
        Features: Light, Water, Humidity.
        """
        self.features = ["light_level", "water_need", "humidity_need", "temp_tolerance"]
        self.X = self.data.select(self.features).to_numpy()
        self.model.fit(self.X)

    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate plant recommendations using K-Nearest Neighbors.

        Args:
            inputs: Dictionary with keys 'plants' (list of plant names), 'flowers', 'toxic'.
            top_k: Number of recommendations to return (default: 3).

        Returns:
            Polars DataFrame containing recommended plants.
        """
        plant_names = inputs.get("plants", [])
        if not plant_names:
            return pl.DataFrame()

        user_plants_df = self.data.filter(
            pl.col("common_name").str.to_lowercase().is_in([p.lower() for p in plant_names])
            | pl.col("scientific_name").str.to_lowercase().is_in([p.lower() for p in plant_names])
        )

        if user_plants_df.height == 0:
            print(f"Warning: No matching plants found for {plant_names}")
            return pl.DataFrame()

        avg_features = user_plants_df.select(self.features).mean().to_numpy()

        distances, indices = self.model.kneighbors(
            avg_features, n_neighbors=min(self.data.height, top_k * 5 + len(plant_names))
        )

        recommended_indices = indices[0]

        recommendations = self.data[recommended_indices]

        recommendations = recommendations.filter(~pl.col("common_name").is_in(user_plants_df["common_name"]))

        flowers = inputs.get("flowers")
        if flowers == "Yes":
            recommendations = recommendations.filter(pl.col("has_flowers") == 1)
        elif flowers == "No":
            recommendations = recommendations.filter(pl.col("has_flowers") == 0)

        toxic = inputs.get("toxic")
        if toxic == "No":
            recommendations = recommendations.filter(pl.col("is_toxic") == 0)

        return recommendations.head(top_k)
