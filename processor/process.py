from pathlib import Path

import polars as pl


def clean_light_col(col_name):
    return (
        pl.col(col_name)
        .cast(pl.String)
        # Fix: Excel converted ranges (e.g., 3-4) into dates
        .str.replace(r"2022-04-03.*", "3_4")
        .str.replace(r"2022-03-01.*", "1_3")
        # Fix: Normalize descriptive text to numeric strings
        .str.replace(r"^1-3.*", "1_3")
        .str.replace(r"(?i).*Bright indirect.*", "2")
        .str.replace(r"2\(.*", "2")
        # Format: Standardize delimiters to underscore for splitting
        .str.replace_all(r"[, \-]+", "_")
        # Calc: Convert string range to average float
        .str.split("_")
        .list.eval(pl.element().cast(pl.Float64, strict=False))
        .list.mean()
    )


def clean_humidity_col(col_name):
    return (
        pl.col(col_name)
        .cast(pl.String)
        # Fix: Normalize descriptive text to numeric strings
        .str.replace(r"Average:.*", "2")
        # Calc: Convert string range to average float
        .str.split("_")
        .list.eval(pl.element().cast(pl.Float64, strict=False))
        .list.mean()
    )


def clean_temp_col(col_name):
    return (
        pl.col(col_name)
        .cast(pl.String)
        # Fix: Normalize descriptive text to numeric strings
        .str.replace(r"Average:.*", "2")
        # Format: Standardize delimiters to underscore for splitting
        .str.replace_all(r"[, \-]+", "_")
        # Calc: Convert string range to average float
        .str.split("_")
        .list.eval(pl.element().cast(pl.Float64, strict=False))
        .list.mean()
    )


def main():
    script_dir = Path(__file__).parent
    src = script_dir.parent / "raw_data"
    dst = script_dir.parent / "processed_data"

    trust_path = src / "trustworthy_plants.xlsx"
    sketch_path = src / "sketch_plants.csv"

    # Plants primarily grown for their blooms
    FLOWERING_SIGNALS = [
        "Orchidaceae",
        "Gesneriaceae",
        "Begoniaceae",
        "Malvaceae",
        "Bromeliaceae",
        "Anthurium",
        "Spathiphyllum",
        "Hibiscus",
        "Kalanchoe",
        "Lily",
        "Hydrangea",
        "Pelargonium",
        "Cyclamen",
        "Azalea",
        "Allamanda",
        "Aphelandra",
        "Beloperone",
        "Bougainvillea",
        "Calceolaria",
        "Carissa",
        "Catharanthus",
        "Chrysanthemum",
        "Citrofortunella",
        "Clerodendrum",
        "Colummea",
        "Crossandra",
        "Fuchsia",
        "Guzmania",
        "Hippeastrum",
        "Hoya",
        "Hyacinthus",
        "Impatiens",
        "Ixora",
        "Jatropha",
        "Justicia",
        "Manettia",
        "Nautilocalyx",
        "Pachystachys",
        "Pentas",
        "Rhododendron",
        "Ruellia",
        "Schlumbergera",
        "Sinningia",
        "Stephanotis",
        "Tillandsia",
        "Trillandsia",
        "Vriesea",
        "Zygocactus",
        "Aechmea",
        "Billbergia",
        "Cryptanthus",
        "Neoregelia",
        "Nidularium",
        "Malvaviscus",
        "Ardissa",
    ]

    # Common names that imply flowers
    FLOWERING_COMMON = ["Flower", "Bloom", "Lily", "Orchid", "Rose", "Violet"]

    # Plants primarily grown for foliage (technically can flower)
    FOLIAGE_SIGNALS = [
        "Araceae",
        "Arecaceae",
        "Pteridaceae",
        "Aspleniaceae",
        "Polypodiaceae",
        "Asparagaceae",
        "Moraceae",
        "Piperaceae",
        "Araliaceae",
        "Ficus",
        "Philodendron",
        "Monstera",
        "Epipremnum",
        "Dracaena",
        "Sansevieria",
        "Hedera",
        "Syngonium",
        "Dieffenbachia",
        "Peperomia",
        "Acorus",
        "Adromischus",
        "Aloe",
        "Araucaria",
        "Ardisia",
        "Astrophytum",
        "Brassaia",
        "Calathea",
        "Callisia",
        "Cereus",
        "Ceropegia",
        "Chlorophytum",
        "Cissus",
        "Codiaeum",
        "Coffea",
        "Coleus",
        "Cordyline",
        "Crassula",
        "Cyperus",
        "Dizygotheca",
        "Dyckia",
        "Echeveria",
        "Echinocereus",
        "Euphorbia",
        "Fatsia",
        "Fittonia",
        "Gasteria",
        "Graptopetalum",
        "Gynura",
        "Haworthia",
        "Hemigraphis",
        "Mammillaria",
        "Maranta",
        "Mikania",
        "Opuntia",
        "Oxalis",
        "Pachira",
        "Pachyphytum",
        "Pellionia",
        "Pilea",
        "Plectranthus",
        "Podocarpus",
        "Polyscias",
        "Saxifraga",
        "Schefflera",
        "Scheflera",
        "Sedum",
        "Sempervivum",
        "Setcreasea",
        "Soleirolia",
        "Stapelia",
        "Strobilanthes",
        "Tolmiea",
        "Tradescantia",
        "Yucca",
        "Zebrina",
        "Scindapsus",
        "Nephrolepis",
    ]

    # Common names that imply foliage
    FOLIAGE_COMMON = ["Fern", "Palm", "Ivy", "Pothos", "Fig", "Rubber"]

    foliage_list_safe = [x for x in FOLIAGE_SIGNALS if x != "Euphorbia"]

    re_flowering = r"(?i)" + "|".join(FLOWERING_SIGNALS)
    re_flowering_common = r"(?i)" + "|".join(FLOWERING_COMMON)
    re_foliage = r"(?i)" + "|".join(foliage_list_safe)
    re_foliage_common = r"(?i)" + "|".join(FOLIAGE_COMMON)

    # Load and Select Data
    trust_df = pl.read_excel(trust_path)
    sketch_df = pl.read_csv(sketch_path, encoding="iso-8859-1")

    trust_df = trust_df.select(
        [
            pl.col("name").alias("scientific_name"),
            pl.col("commonName").alias("common_name"),
            pl.col("Family"),
            pl.col("Toxicity").alias("is_toxic"),
            pl.col("brightness").alias("light_level"),
            pl.col("watering").alias("water_need"),
            pl.col("solHumidity").alias("humidity_need"),
            pl.col("temperature").alias("temp_tolerance"),
            pl.col("Flower").alias("has_flowers"),
            pl.col("description"),
            pl.col("General care"),
        ]
    )

    # Clean and Patch Numeric Columns
    trust_df = trust_df.with_columns(
        [clean_light_col("light_level"), clean_humidity_col("humidity_need"), clean_temp_col("temp_tolerance")]
    )
    # Patches for missing values
    trust_df = trust_df.with_columns(
        pl.when(pl.col("scientific_name") == "Philodendron hederaceum")
        .then(2.5)
        .when(pl.col("scientific_name") == "Hydrangea hortensia")
        .then(1.5)
        .otherwise(pl.col("light_level"))
        .alias("light_level"),
        pl.when(pl.col("scientific_name") == "Hydrangea hortensia")
        .then(1.0)
        .when(pl.col("scientific_name") == "Pachira aquatiac")
        .then(1.0)
        .when(pl.col("scientific_name") == "Philodendron hederaceum")
        .then(2.0)
        .otherwise(pl.col("humidity_need"))
        .alias("humidity_need"),
        pl.when(pl.col("scientific_name") == "Hydrangea hortensia")
        .then(1.0)
        .when(pl.col("scientific_name") == "Pachira aquatiac")
        .then(2.0)
        .when(pl.col("scientific_name") == "Philodendron hederaceum")
        .then(2.0)
        .otherwise(pl.col("temp_tolerance"))
        .alias("temp_tolerance"),
    )

    # Logic: Toxicity & Water
    trust_df = trust_df.with_columns(
        pl.when(pl.col("is_toxic").fill_null("Safe").str.contains(r"(?i)toxic|poisonous"))
        .then(1)
        .otherwise(0)
        .alias("is_toxic")
    )

    trust_df = trust_df.with_columns(
        pl.when(
            pl.col("scientific_name").str.contains(
                r"(?i)Sansevieria|Haworthia|Sedum|Sempervivum|Dyckia|Euphorbia|Pachyphytum|Stapelia"
            )
        )
        .then(3)
        .when(pl.col("scientific_name").str.contains(r"(?i)Ficus|Monstera|Pachira|Pellionia|Philodendron"))
        .then(2)
        .when(pl.col("scientific_name").str.contains(r"(?i)Pellaea|Hydrangea|Paphiopedilum"))
        .then(1)
        .otherwise(pl.col("water_need"))
        .alias("water_need")
    )

    # Logic: Flowering
    trust_df = trust_df.with_columns(
        pl.when(pl.col("has_flowers").is_not_null())
        .then(pl.col("has_flowers"))  # Keep existing data if present
        # -- EXCEPTIONS --
        # Euphorbia Split: Poinsettia/Milii are flowers, generic Euphorbia is foliage
        .when(pl.col("scientific_name").str.contains(r"(?i)Euphorbia pulcherrima|Euphorbia milii"))
        .then(1)
        .when(pl.col("scientific_name").str.contains(r"(?i)Euphorbia"))
        .then(0)
        # -- FLOWERING LISTS --
        .when(pl.col("scientific_name").str.contains(re_flowering) | pl.col("Family").str.contains(re_flowering))
        .then(1)
        # -- FOLIAGE LISTS --
        .when(pl.col("scientific_name").str.contains(re_foliage) | pl.col("Family").str.contains(re_foliage))
        .then(0)
        # -- COMMON NAMES --
        .when(pl.col("common_name").str.contains(re_flowering_common))
        .then(1)
        .when(pl.col("common_name").str.contains(re_foliage_common))
        .then(0)
        .otherwise(pl.col("has_flowers"))
        .alias("has_flowers")
    )

    trust_df = trust_df.drop("Family")

    trust_df.write_parquet(dst / "trustworthy_plants.parquet")
    trust_df.write_csv(dst / "trustworthy_plants.csv")
    sketch_df.write_parquet(dst / "sketch_plants.parquet")


if __name__ == "__main__":
    main()
