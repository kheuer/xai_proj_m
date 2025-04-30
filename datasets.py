import pandas as pd
import os
from PIL import Image


# Build the Pacs Dataframe
builder = {"label": [], "image": [], "domain": []}

for domain in (
    "sketch",
    "photo",
    "cartoon",
    "art_painting",
):
    domain_path = os.path.join("pacs-dataset/pacs_data/pacs_data", domain)
    for label in os.listdir(domain_path):
        label_path = os.path.join(domain_path, label)
        for filename in os.listdir(label_path):
            builder["label"].append(label)
            builder["domain"].append(domain)
            builder["image"].append(Image.open(os.path.join(label_path, filename)))


pacs_df = pd.DataFrame(builder)
