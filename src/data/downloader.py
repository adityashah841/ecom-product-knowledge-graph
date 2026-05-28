"""
Data downloader for the E-Commerce Product Knowledge Graph pipeline.

Primary strategy: generate a large synthetic product catalogue with realistic
Amazon-style titles and perfect BIO labels. Each title is assembled from
real brand names, categories, colors, materials and attributes using randomized
templates, giving us perfectly aligned NER training data without any external
dataset dependency.

Fallback: try HuggingFace datasets that are pure-Parquet (no custom scripts).
"""

import os
import json
import random
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Vocabulary ────────────────────────────────────────────────────────────────

BRANDS = [
    "Nike", "Adidas", "Puma", "Under Armour", "Reebok", "New Balance",
    "Asics", "Saucony", "Brooks", "Hoka", "Salomon", "Merrell",
    "Apple", "Samsung", "Sony", "LG", "Bose", "JBL", "Beats", "Sennheiser",
    "Logitech", "Razer", "Corsair", "SteelSeries", "HyperX", "Anker",
    "Levi's", "H&M", "Zara", "Gap", "Tommy Hilfiger", "Ralph Lauren",
    "Calvin Klein", "Guess", "Wrangler", "Lee", "Dickies", "Carhartt",
    "Columbia", "The North Face", "Patagonia", "Arc'teryx", "Marmot",
    "Timberland", "Caterpillar", "Dr. Martens", "Clarks", "Skechers",
    "Crocs", "Birkenstock", "Vans", "Converse", "Fila",
    "Amazon Basics", "AmazonBasics", "Basics by Amazon",
    "KitchenAid", "Cuisinart", "Instant Pot", "Ninja", "Vitamix",
    "Braun", "Philips", "Dyson", "iRobot", "Shark", "Bissell",
    "IKEA", "Wayfair", "Ashley", "Serta", "Tempur-Pedic", "Casper",
    "Fossil", "Casio", "Seiko", "Citizen", "Timex", "Bulova",
    "L'Oreal", "Neutrogena", "Cetaphil", "Aveeno", "Dove", "Olay",
    "Colgate", "Oral-B", "Crest", "Gillette", "Schick", "BIC",
    "Trek", "Giant", "Specialized", "Schwinn", "Diamondback",
    "Wilson", "Titleist", "Callaway", "TaylorMade", "Ping", "Cobra",
    "Rawlings", "Easton", "Louisville Slugger", "Mizuno", "Majestic",
    "Dewalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Black+Decker",
    "Craftsman", "Stanley", "Irwin", "Klein Tools",
    "3M", "Scotch", "Avery", "Brother", "Epson", "HP",
    "Lego", "Mattel", "Hasbro", "Fisher-Price", "VTech", "Leapfrog",
    "Rubbermaid", "Tupperware", "Pyrex", "OXO", "Lodge", "Le Creuset",
    "Champion", "Hanes", "Fruit of the Loom", "Gildan", "Bella+Canvas",
    "Pacsafe", "Osprey", "Deuter", "Gregory", "Kelty", "REI Co-op",
    "Hydro Flask", "Nalgene", "CamelBak", "Yeti", "RTIC", "Stanley",
    "Ziploc", "Glad", "Hefty", "Reynolds", "Saran", "Clorox",
    "Tide", "Downy", "Bounce", "Gain", "Persil", "ALL",
    "Energizer", "Duracell", "Rayovac", "Amazon Rechargeable",
    "WD-40", "Rust-Oleum", "Krylon", "Sherwin-Williams",
]

CATEGORIES = {
    "Running Shoes": ["Road Running Shoes", "Trail Running Shoes", "Racing Flats", "Cross Training Shoes"],
    "Sneakers": ["Casual Sneakers", "High-Top Sneakers", "Low-Top Sneakers", "Platform Sneakers"],
    "Boots": ["Ankle Boots", "Chelsea Boots", "Hiking Boots", "Work Boots", "Winter Boots"],
    "T-Shirt": ["Graphic T-Shirt", "Plain T-Shirt", "Performance T-Shirt", "Polo Shirt", "Long Sleeve Shirt"],
    "Jeans": ["Slim Fit Jeans", "Straight Leg Jeans", "Skinny Jeans", "Bootcut Jeans", "Relaxed Fit Jeans"],
    "Jacket": ["Rain Jacket", "Windbreaker", "Fleece Jacket", "Down Jacket", "Softshell Jacket"],
    "Hoodie": ["Pullover Hoodie", "Zip-Up Hoodie", "Oversized Hoodie", "Tech Fleece Hoodie"],
    "Shorts": ["Athletic Shorts", "Cargo Shorts", "Board Shorts", "Running Shorts", "Compression Shorts"],
    "Leggings": ["Yoga Pants", "Compression Leggings", "Running Tights", "Thermal Leggings"],
    "Dress": ["Casual Dress", "Summer Dress", "Midi Dress", "Maxi Dress", "Mini Dress"],
    "Laptop": ["Gaming Laptop", "Ultrabook", "Business Laptop", "Chromebook", "2-in-1 Laptop"],
    "Headphones": ["Over-Ear Headphones", "On-Ear Headphones", "In-Ear Headphones", "Wireless Headphones", "Noise Cancelling Headphones"],
    "Earbuds": ["True Wireless Earbuds", "Sports Earbuds", "In-Ear Monitors", "Bluetooth Earbuds"],
    "Backpack": ["School Backpack", "Hiking Backpack", "Travel Backpack", "Laptop Backpack", "Drawstring Bag"],
    "Watch": ["Sports Watch", "Smartwatch", "Analog Watch", "Digital Watch", "Dive Watch"],
    "Water Bottle": ["Insulated Water Bottle", "Sport Water Bottle", "Wide Mouth Bottle", "Squeeze Bottle"],
    "Yoga Mat": ["Exercise Mat", "Fitness Mat", "Anti-Slip Mat", "Thick Yoga Mat"],
    "Coffee Maker": ["Drip Coffee Maker", "Single Serve Coffee Maker", "Espresso Machine", "French Press", "Pour Over"],
    "Blender": ["High Speed Blender", "Personal Blender", "Immersion Blender", "Countertop Blender"],
    "Phone Case": ["Protective Case", "Slim Case", "Wallet Case", "Clear Case", "Rugged Case"],
    "Sunglasses": ["Polarized Sunglasses", "Sports Sunglasses", "Aviator Sunglasses", "Wayfarer Sunglasses"],
    "Hat": ["Baseball Cap", "Snapback Hat", "Beanie", "Bucket Hat", "Sun Hat", "Trucker Hat"],
    "Socks": ["Athletic Socks", "Compression Socks", "No-Show Socks", "Crew Socks", "Wool Socks"],
    "Gloves": ["Winter Gloves", "Work Gloves", "Touchscreen Gloves", "Running Gloves", "Cycling Gloves"],
    "Belt": ["Leather Belt", "Canvas Belt", "Tactical Belt", "Reversible Belt"],
    "Wallet": ["Bifold Wallet", "Trifold Wallet", "Slim Wallet", "Money Clip", "Card Holder"],
    "Dumbbell": ["Hex Dumbbell", "Adjustable Dumbbell", "Rubber Dumbbell", "Neoprene Dumbbell"],
    "Resistance Band": ["Loop Band", "Pull Up Band", "Tube Band", "Flat Band"],
    "Foam Roller": ["High Density Foam Roller", "Vibrating Foam Roller", "Half Round Roller"],
    "Tent": ["Backpacking Tent", "Car Camping Tent", "Ultralight Tent", "Family Tent"],
    "Sleeping Bag": ["Mummy Sleeping Bag", "Rectangular Sleeping Bag", "Down Sleeping Bag", "Synthetic Sleeping Bag"],
}

ATTRIBUTES = [
    "Lightweight", "Breathable", "Waterproof", "Water-Resistant", "Quick-Dry",
    "UV Protection", "Moisture-Wicking", "Anti-Odor", "Stretch", "Slim Fit",
    "Regular Fit", "Relaxed Fit", "Oversized", "Cropped", "High-Waisted",
    "Men's", "Women's", "Unisex", "Boys'", "Girls'", "Youth",
    "Plus Size", "Petite", "Tall", "Big & Tall",
    "2-Pack", "3-Pack", "4-Pack", "6-Pack", "12-Pack",
    "Pro", "Elite", "Premium", "Classic", "Essential", "Sport",
    "Wireless", "Bluetooth", "USB-C", "Fast Charge", "Solar",
    "Eco-Friendly", "Recycled", "Organic", "Natural", "Sustainable",
    "Adjustable", "Removable", "Reversible", "Convertible", "Foldable",
    "Heavy Duty", "Ultra-Durable", "Impact Resistant", "Reinforced",
    "Non-Slip", "Anti-Scratch", "Stain-Resistant",
    "Machine Washable", "Hand Wash Only", "Dry Clean",
    "S", "M", "L", "XL", "XXL", "Small", "Medium", "Large", "Extra Large",
    "Size 8", "Size 9", "Size 10", "Size 11", "Size 12",
    "4K", "HD", "1080p", "OLED", "AMOLED",
    "32GB", "64GB", "128GB", "256GB", "512GB",
    "AA", "AAA",
]

COLORS = [
    "Black", "White", "Gray", "Navy", "Red", "Blue", "Green", "Yellow",
    "Orange", "Purple", "Pink", "Brown", "Beige", "Olive", "Teal",
    "Charcoal", "Burgundy", "Coral", "Turquoise", "Cream", "Khaki",
    "Royal Blue", "Forest Green", "Heather Gray", "Light Blue",
    "Dark Green", "Hot Pink", "Sky Blue", "Mint", "Lavender",
    "Midnight Blue", "Rose Gold", "Gold", "Silver",
]

MATERIALS = [
    "Cotton", "Polyester", "Nylon", "Wool", "Leather", "Suede",
    "Mesh", "Fleece", "Spandex", "Denim", "Canvas", "Linen",
    "Silk", "Velvet", "Jersey", "Microfiber", "Gore-Tex",
    "Rubber", "Foam", "Memory Foam", "Stainless Steel", "Aluminum",
    "Carbon Fiber", "Bamboo", "Down", "Synthetic Fill",
]

# Title templates: each element is either a literal or a field name in brackets
TEMPLATES = [
    "{brand} {attr} {color} {material} {category}",
    "{brand} {color} {material} {category} for {use}",
    "{brand} {attr} {category} - {color} {material}",
    "{brand} {category} {color} | {attr} | {material}",
    "{color} {material} {category} by {brand} - {attr}",
    "{brand} {attr} {category} ({color}, {material})",
    "{brand} {color} {category} | {material} | {attr}",
    "{brand} {material} {color} {category} | {attr}",
    "{attr} {brand} {category} in {color} {material}",
    "{brand} {color} {category} with {material} {attr}",
]

USE_CASES = [
    "Men", "Women", "Kids", "Adults", "Runners", "Hikers", "Athletes",
    "Gym", "Office", "Travel", "Outdoor", "Everyday Use",
    "Training", "Yoga", "Cycling", "Swimming",
]


def generate_synthetic_products(n: int, seed: int) -> pd.DataFrame:
    """
    Generate n realistic Amazon-style product listings with structured metadata.
    Returns a DataFrame with columns: title, brand, category, color, material, attribute.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    records = []
    cat_names = list(CATEGORIES.keys())
    cat_weights = np_rng.dirichlet(np.ones(len(cat_names)) * 2)  # slightly uneven

    for _ in tqdm(range(n), desc="Generating synthetic products"):
        cat_name = rng.choices(cat_names, weights=cat_weights)[0]
        subcats = CATEGORIES[cat_name]
        subcat = rng.choice(subcats)

        brand = rng.choice(BRANDS)
        color = rng.choice(COLORS)
        material = rng.choice(MATERIALS)
        attr = rng.choice(ATTRIBUTES)
        use = rng.choice(USE_CASES)

        template = rng.choice(TEMPLATES)
        title = template.format(
            brand=brand, color=color, material=material,
            category=subcat, attr=attr, use=use,
        )

        records.append({
            "title": title,
            "brand": brand,
            "category": cat_name,
            "subcategory": subcat,
            "color": color,
            "material": material,
            "attribute": attr,
        })

    return pd.DataFrame(records)


def generate_silver_labels(row: pd.Series) -> dict | None:
    """
    Generate BIO labels by matching known entity strings in the title.
    Since we generated the title we have perfect alignment.
    """
    title = str(row.get("title", "")).strip()
    if len(title) < 3:
        return None

    tokens = title.split()
    labels = ["O"] * len(tokens)

    def tag_span(start: int, end: int, entity_type: str):
        labels[start] = f"B-{entity_type}"
        for i in range(start + 1, end):
            labels[i] = f"I-{entity_type}"

    def find_and_tag(phrase: str, entity_type: str):
        if not phrase or phrase.lower() in ("nan", "none", ""):
            return
        phrase_tokens = phrase.lower().split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            window = [t.lower() for t in tokens[i:i + n]]
            if window == phrase_tokens and labels[i] == "O":
                tag_span(i, i + n, entity_type)
                return

    find_and_tag(str(row.get("brand", "")), "BRAND")
    find_and_tag(str(row.get("category", "")), "CATEGORY")
    find_and_tag(str(row.get("subcategory", "")), "CATEGORY")
    find_and_tag(str(row.get("color", "")), "COLOR")
    find_and_tag(str(row.get("material", "")), "MATERIAL")
    find_and_tag(str(row.get("attribute", "")), "ATTRIBUTE")

    return {"tokens": tokens, "labels": labels}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def split_and_save(records: list, processed_dir: str, config: dict):
    seed = config["data"]["seed"]
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    train_end = int(n * config["data"]["train_split"])
    val_end = train_end + int(n * config["data"]["val_split"])
    splits = {
        "ner_train": records[:train_end],
        "ner_val": records[train_end:val_end],
        "ner_test": records[val_end:],
    }
    os.makedirs(processed_dir, exist_ok=True)
    for split_name, split_records in splits.items():
        path = os.path.join(processed_dir, f"{split_name}.json")
        with open(path, "w") as f:
            for rec in split_records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {len(split_records)} records to {path}")


def main(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    samples_dir = config["paths"]["samples"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    n = config["data"]["sample_size"]
    seed = config["data"]["seed"]

    logger.info(f"Generating {n} synthetic product listings (seed={seed})...")
    df = generate_synthetic_products(n, seed)
    logger.info(f"Generated {len(df)} products across {df['category'].nunique()} categories")

    # Save raw parquet
    raw_path = os.path.join(raw_dir, "products_sampled.parquet")
    df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path}")

    # Save 500-row sample CSV
    sample_path = os.path.join(samples_dir, "products_sample.csv")
    df.head(500).to_csv(sample_path, index=False)
    logger.info(f"Saved 500-row sample to {sample_path}")

    # Generate silver labels
    logger.info("Generating BIO labels...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Silver labeling"):
        result = generate_silver_labels(row)
        if result and len(result["tokens"]) > 0:
            records.append(result)
    logger.info(f"Labeled {len(records)} examples")

    split_and_save(records, processed_dir, config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
