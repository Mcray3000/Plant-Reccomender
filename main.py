import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from model.data_loader import DataLoader
from model.heuristic import HeuristicRecommender
from model.knn import KNNRecommender
from model.random_recommender import RandomRecommender

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize data loader
loader = DataLoader()
data = loader.load_data()

# Map plant scientific names to image files
def get_plant_image_url(scientific_name):
    """Get the image URL for a plant based on its scientific name."""
    safe_name = scientific_name.replace(" ", "_")
    image_path = Path(f"static/plant_images/{safe_name}.jpg")
    if image_path.exists():
        return f"/static/plant_images/{safe_name}.jpg"
    return "/static/placeholder.jpg"


def format_plant_data(plant_row):
    """Format a plant row into a dictionary for template rendering."""
    return {
        "scientific_name": plant_row.get("scientific_name", ""),
        "common_name": plant_row.get("common_name", ""),
        "image_url": get_plant_image_url(plant_row.get("scientific_name", "")),
        "light_level": f"{plant_row.get('light_level', 0):.1f}" if plant_row.get('light_level') else "N/A",
        "water_need": f"{plant_row.get('water_need', 0):.1f}" if plant_row.get('water_need') else "N/A",
        "humidity_need": f"{plant_row.get('humidity_need', 0):.1f}" if plant_row.get('humidity_need') else "N/A",
        "temp_tolerance": f"{plant_row.get('temp_tolerance', 0):.1f}" if plant_row.get('temp_tolerance') else "N/A",
        "is_toxic": "Yes" if plant_row.get("is_toxic") == 1 else "No",
        "flowers": "Yes" if plant_row.get("has_flowers", "0") == "1" or plant_row.get("has_flowers") == 1 else "No",
        "general_care": plant_row.get("General care") or "No specific care information available",
        "description": plant_row.get("description") or "",
    }


@app.route("/")
def index():
    """Home page with navigation to both flows."""
    return render_template("index.html")


@app.route("/questionnaire")
def questionnaire():
    """Questionnaire form page."""
    return render_template("questionnaire.html")


@app.route("/item-based")
def item_based():
    """Item-based recommendation form page."""
    # Get list of available plants for autocomplete
    plants = data.select(["scientific_name", "common_name"]).to_dicts()
    plant_names = [(p["scientific_name"], p["common_name"]) for p in plants]
    return render_template("item_based.html", available_plants=plant_names)


@app.route("/api/recommend/questionnaire", methods=["POST"])
def recommend_questionnaire():
    """API endpoint for questionnaire-based recommendations."""
    try:
        req_data = request.get_json()

        inputs = {
            "light": req_data.get("light", "Grow Light"),
            "care": req_data.get("care", "Low"),
            "room": req_data.get("room", "Living Room"),
            "flowers": req_data.get("flowers", "Dont Care"),
            "toxic": req_data.get("toxic", "No"),
        }

        model_choice = req_data.get("model", "heuristic")

        if model_choice == "heuristic":
            recommender = HeuristicRecommender(data)
        elif model_choice == "random":
            recommender = RandomRecommender(data)
        else:
            recommender = HeuristicRecommender(data)

        recommendations = recommender.recommend(inputs, top_k=3)
        plants = [format_plant_data(row) for row in recommendations.to_dicts()]

        return jsonify({"success": True, "plants": plants})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/recommend/item-based", methods=["POST"])
def recommend_item_based():
    """API endpoint for item-based recommendations."""
    try:
        req_data = request.get_json()

        plant_list = req_data.get("plants", [])
        if not plant_list:
            return jsonify({"success": False, "error": "Please enter at least one plant"}), 400

        inputs = {
            "plants": plant_list,
            "flowers": req_data.get("flowers", "Dont Care"),
            "toxic": req_data.get("toxic", "Dont Care"),
        }

        model_choice = req_data.get("model", "knn")

        if model_choice == "knn":
            recommender = KNNRecommender(data)
        elif model_choice == "random":
            recommender = RandomRecommender(data)
        else:
            recommender = KNNRecommender(data)

        recommendations = recommender.recommend(inputs, top_k=3)
        plants = [format_plant_data(row) for row in recommendations.to_dicts()]

        return jsonify({"success": True, "plants": plants})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/plants/search", methods=["GET"])
def search_plants():
    """API endpoint to search for plants by name."""
    query = request.args.get("q", "").lower()

    if not query or len(query) < 2:
        return jsonify([])

    plants = data.to_dicts()
    matches = [
        {
            "scientific": p["scientific_name"],
            "common": p["common_name"],
            "label": f"{p['common_name']} ({p['scientific_name']})",
        }
        for p in plants
        if (p.get("scientific_name") and p.get("common_name") and 
            (query in p["scientific_name"].lower() or query in p["common_name"].lower()))
    ]

    return jsonify(matches[:10])


@app.route("/api/plants/all", methods=["GET"])
def get_all_plants():
    """API endpoint to get all plants with their images for selection."""
    plants = data.to_dicts()
    plant_list = [
        {
            "scientific_name": p["scientific_name"],
            "common_name": p["common_name"],
            "image_url": get_plant_image_url(p.get("scientific_name", "")),
            "light_level": f"{p.get('light_level', 0):.1f}" if p.get('light_level') else "N/A",
            "water_need": f"{p.get('water_need', 0):.1f}" if p.get('water_need') else "N/A",
        }
        for p in plants
        if p.get("scientific_name") and p.get("common_name")
    ]
    return jsonify(plant_list)


if __name__ == "__main__":
    # Create static directories if they don't exist
    os.makedirs("static/plant_images", exist_ok=True)

    # Copy plant images to static folder
    images_source_dir = "processed_data/images"
    images_dest_dir = "static/plant_images"

    if os.path.exists(images_source_dir):
        for img_file in os.listdir(images_source_dir):
            if img_file.endswith(".jpg"):
                src = os.path.join(images_source_dir, img_file)
                dst = os.path.join(images_dest_dir, img_file)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)

    app.run(debug=True, host="127.0.0.1", port=8000)
