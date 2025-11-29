import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from bs4 import BeautifulSoup


def scrape_single_image(name, output_dir, headers):
    """
    Scrapes a single image for the given plant name from Google Images.
    """
    safe_name = name.replace(" ", "_")
    image_path = os.path.join(output_dir, f"{safe_name}.jpg")

    search_url = f"https://www.google.com/search?q={name}+houseplant&tbm=isch&tbs=isz:l"

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        images = soup.find_all("img")

        # Iterate through images (skipping the first one which is usually a logo)
        # We try up to 5 candidates
        candidates = images[1:6]

        for i, img in enumerate(candidates):
            img_url = img.get("src")
            if not img_url or not img_url.startswith("http"):
                continue

            try:
                img_data = requests.get(img_url, timeout=5).content

                # Check size
                with open(image_path, "wb") as handler:
                    handler.write(img_data)
                return f"Saved image for {name} (Size: {len(img_data) // 1024}KB)"

            except Exception:
                continue

        return f"No suitable image found for {name}"

    except Exception as e:
        return f"Error scraping {name}: {e}"


def scrape_images(
    parquet_path="processed_data/trustworthy_plants.parquet", output_dir="processed_data/images", max_workers=10
):
    """
    Reads scientific names from a Parquet file, scrapes images in parallel.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(parquet_path):
        print(f"Parquet file not found at {parquet_path}")
        return

    df = pl.read_parquet(parquet_path)

    if "scientific_name" not in df.columns:
        print("Column 'scientific_name' not found.")
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    plant_names = df["scientific_name"].unique().to_list()
    print(f"Starting scrape for {len(plant_names)} plants with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(scrape_single_image, name, output_dir, headers): name for name in plant_names}

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")


if __name__ == "__main__":
    if os.path.exists("../processed_data/trustworthy_plants.parquet"):
        scrape_images(
            parquet_path="../processed_data/trustworthy_plants.parquet", output_dir="../processed_data/images"
        )
    else:
        scrape_images(parquet_path="processed_data/trustworthy_plants.parquet", output_dir="processed_data/images")
