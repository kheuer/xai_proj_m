import os
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
import requests

load_dotenv()

# Get webhook URL from env
ALERT_URL = os.getenv("ALERT_URL")
INFO_URL = os.getenv("INFO_URL")

def send_discord_notification(webhook_url, message, image_array=None, filename="image.png"):
    data = {"content": message}
    files = None

    if image_array is not None:
        # Convert NumPy array to image
        image = Image.fromarray(image_array.astype("uint8"))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        files = {"file": (filename, buffer, "image/png")}
        response = requests.post(webhook_url, data=data, files=files)
        buffer.close()
    else:
        response = requests.post(webhook_url, json=data)

    if response.status_code not in (200, 204):
        print(f"Failed: {response.status_code} - {response.text}")



def send_embed_progress(epoch, total_epochs, webhook_url):
    percent = int(100 * epoch / total_epochs)
    embed = {
        "title": "Training Progress",
        "description": f"Epoch: **{epoch}/{total_epochs}**\nProgress: **{percent}%**",
        "color": 0x3498db  # blue
    }
    requests.post(webhook_url, json={"embeds": [embed]})


def send_progress_bar(rel_model_idx, num_models, start_idx, webhook_url, duration, process, partition):
    percent = rel_model_idx / num_models
    filled = int(10 * percent)
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    bar = "█" * filled + "—" * (10 - filled)
    message = f"Progress: [{bar}] \n{start_idx + rel_model_idx}/{start_idx + num_models}, done models:{rel_model_idx}, ({percent:.0%}) ({hours:02}:{minutes:02}) on process {process} partition: {partition}"

    requests.post(webhook_url, json={"content": message})