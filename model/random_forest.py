from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from model.base import BaseRecommender


class RandomForestRecommender(BaseRecommender):
    """
    Recommends plants using a Random Forest Classifier.
    It clusters plants into 'Care Levels' and predicts the suitable cluster for the user.
    """

    def __init__(self, data: pl.DataFrame):
        super().__init__(data)
        self.model = None
        self.le_light = LabelEncoder()
        self.le_care = LabelEncoder()  # Not used for input features directly but for mapping
        self._train_model()

    def _train_model(self):
        """
        Trains the RF model.
        """

        df = self.data.to_pandas()

        conditions = [
            (df["water_need"] <= 1) & (df["humidity_need"] <= 2),
            (df["water_need"] <= 2),
        ]
        choices = [0, 1]
        df["difficulty_cluster"] = np.select(conditions, choices, default=2)

        self.train_df = df

        X = df[["light_level", "water_need", "humidity_need", "temp_tolerance"]]
        y = df["difficulty_cluster"]

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def recommend(self, inputs: dict[str, Any], top_k: int = 3) -> pl.DataFrame:
        """
        Generate plant recommendations using Random Forest classification.

        Args:
            inputs: Dictionary with keys 'light', 'care', 'room', 'flowers', 'toxic'.
            top_k: Number of recommendations to return (default: 3).

        Returns:
            Polars DataFrame containing recommended plants.
        """
        light_map = {
            "South": 1.0,
            "South-East": 1.5,
            "South-West": 1.5,
            "East": 2.0,
            "West": 2.0,
            "Grow Light": 2.5,
            "North-East": 3.0,
            "North-West": 3.0,
            "North": 3.5,
        }
        light_val = light_map.get(inputs.get("light"), 2.5)

        care_map = {"Low": 1, "Medium": 2, "High": 3}
        care_val = care_map.get(inputs.get("care"), 2)

        room_val = 2.5 if inputs.get("room") == "Bathroom" else 1.5

        temp_val = 2.0

        input_vector = np.array([[light_val, care_val, room_val, temp_val]])

        predicted_cluster = self.model.predict(input_vector)[0]

        cluster_plants = self.train_df[self.train_df["difficulty_cluster"] == predicted_cluster]

        result_pl = pl.from_pandas(cluster_plants)

        flowers = inputs.get("flowers")
        if flowers == "Yes":
            result_pl = result_pl.filter(pl.col("has_flowers") == 1)
        elif flowers == "No":
            result_pl = result_pl.filter(pl.col("has_flowers") == 0)

        toxic = inputs.get("toxic")
        if toxic == "No":
            result_pl = result_pl.filter(pl.col("is_toxic") == 0)

        if result_pl.height > top_k:
            return result_pl.sample(top_k)

        return result_pl
