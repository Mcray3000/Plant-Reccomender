import argparse
import os
import shutil
import sys

from model.content_based import ContentBasedRecommender
from model.data_loader import DataLoader
from model.heuristic import HeuristicRecommender
from model.knn import KNNRecommender
from model.random_forest import RandomForestRecommender
from model.random_recommender import RandomRecommender


def clear_recommended_images():
    """Clear the recommended_images folder."""
    rec_images_dir = "recommended_images"
    if os.path.exists(rec_images_dir):
        shutil.rmtree(rec_images_dir)
    os.makedirs(rec_images_dir, exist_ok=True)


def copy_recommended_images(recommendations):
    """Copy images for recommended plants to recommended_images folder."""
    images_source_dir = "processed_data/images"
    rec_images_dir = "recommended_images"

    if not os.path.exists(images_source_dir):
        print(f"Warning: Images directory '{images_source_dir}' not found.")
        return

    copied_count = 0
    for scientific_name in recommendations["scientific_name"]:
        safe_name = scientific_name.replace(" ", "_")
        source_path = os.path.join(images_source_dir, f"{safe_name}.jpg")
        dest_path = os.path.join(rec_images_dir, f"{safe_name}.jpg")

        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            print(f"Warning: Image not found for {scientific_name}")

    print(f"Copied {copied_count} image(s) to {rec_images_dir}/")


def main():
    clear_recommended_images()

    parser = argparse.ArgumentParser(description="Plant Recommender System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_quest = subparsers.add_parser("questionnaire", help="Recommend based on questionnaire")
    parser_quest.add_argument(
        "-l",
        "--light",
        type=str,
        choices=[
            "North",
            "North-West",
            "West",
            "South-West",
            "South",
            "South-East",
            "East",
            "North-East",
            "Grow Light",
        ],
        default="Grow Light",
        help="Light availability",
    )
    parser_quest.add_argument(
        "-f",
        "--flowers",
        type=str,
        choices=["Yes", "No", "Dont Care"],
        default="Dont Care",
        help="Preference for flowering plants",
    )
    parser_quest.add_argument(
        "-t",
        "--toxic",
        type=str,
        choices=["Yes", "No", "Dont Care"],
        default="No",
        help="Are toxic plants acceptable (e.g., due to pets)?",
    )
    parser_quest.add_argument(
        "-c", "--care", type=str, choices=["Low", "Medium", "High"], default="Low", help="Care capacity"
    )
    parser_quest.add_argument(
        "-r",
        "--room",
        type=str,
        choices=["Living Room", "Bathroom", "Bedroom", "Kitchen"],
        default="Living Room",
        help="Room type",
    )
    parser_quest.add_argument(
        "-m", "--model", type=str, choices=["heuristic", "rf", "random"], default="heuristic", help="Model to use"
    )

    parser_item = subparsers.add_parser("item", help="Recommend based on existing plants")
    parser_item.add_argument("-p", "--plants", type=str, required=True, help="Comma-separated list of current plants")
    parser_item.add_argument(
        "-f",
        "--flowers",
        type=str,
        choices=["Yes", "No", "Dont Care"],
        default="Dont Care",
        help="Preference for flowering plants",
    )
    parser_item.add_argument(
        "-t",
        "--toxic",
        type=str,
        choices=["Yes", "No", "Dont Care"],
        default="Dont Care",
        help="Are toxic plants acceptable?",
    )
    parser_item.add_argument(
        "-m", "--model", type=str, choices=["knn", "content", "random"], default="content", help="Model to use"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    print("Loading data...")
    loader = DataLoader()
    try:
        data = loader.load_data()
        print(f"Data loaded: {data.shape[0]} plants available.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if args.command == "questionnaire":
        print(f"Running Questionnaire Recommender ({args.model})...")
        inputs = {
            "light": args.light,
            "care": args.care,
            "room": args.room,
            "flowers": args.flowers,
            "toxic": args.toxic,
        }

        if args.model == "heuristic":
            recommender = HeuristicRecommender(data)
        elif args.model == "rf":
            recommender = RandomForestRecommender(data)
        elif args.model == "random":
            recommender = RandomRecommender(data)

        recommendations = recommender.recommend(inputs)
        print("\nTop Recommendations:")
        print(recommendations.select(["scientific_name", "common_name", "light_level", "water_need"]))

        copy_recommended_images(recommendations)

    elif args.command == "item":
        print(f"Running Item-based Recommender ({args.model})...")
        plant_list = [p.strip() for p in args.plants.split(",")]
        inputs = {"plants": plant_list, "flowers": args.flowers, "toxic": args.toxic}

        if args.model == "knn":
            recommender = KNNRecommender(data)
        elif args.model == "content":
            recommender = ContentBasedRecommender(data)
        elif args.model == "random":
            recommender = RandomRecommender(data)

        recommendations = recommender.recommend(inputs)
        print("\nTop Recommendations:")
        print(recommendations.select(["scientific_name", "common_name", "light_level", "water_need"]))

        copy_recommended_images(recommendations)


if __name__ == "__main__":
    main()
