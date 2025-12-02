import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

# === CONFIGURATION ===
EXCEL_FILE_PATH = ''            #File path fo images to download and compress
COLUMN_IMAGE_LINK = 'image_link'
COLUMN_SAMPLE_ID = 'sample_id'

SAVE_FOLDER = 'C:\\Users\\user\\Desktop\\amazon challenge\\images20-30k'

RESIZE_DIM = (224, 224)        # Resize dimensions
JPEG_QUALITY = 65              # Compression level (1–95)

START_INDEX = 20001             # Start from row
BATCH_SIZE = 10000              # Number of images to process

# === UTILITY FUNCTIONS ===

def get_existing_sample_ids(folder_path):
    """Get sample IDs for already downloaded & processed images."""
    existing_files = os.listdir(folder_path) if os.path.exists(folder_path) else []
    return {Path(f).stem for f in existing_files if f.lower().endswith(('.jpg'))}

def read_entries(file_path, column_image_link, column_sample_id, start=0, limit=None):
    df = pd.read_csv(file_path)
    if column_image_link not in df.columns or column_sample_id not in df.columns:
        raise ValueError(f"Missing column '{column_image_link}' or '{column_sample_id}' in CSV.")
    df = df.dropna(subset=[column_image_link, column_sample_id])
    df = df.iloc[start:start + limit if limit is not None else None]
    return list(zip(df[column_image_link], df[column_sample_id].astype(str)))

def download_resize_compress(image_url, sample_id, save_folder):
    """
    Downloads an image from URL, resizes and compresses it, then saves as JPEG.
    Filename will be <sample_id>.jpg
    """
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        # Load image from response into PIL
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Resize
        img = img.resize(RESIZE_DIM, Image.LANCZOS)

        # Output filename
        filename = f"{sample_id}.jpg"
        save_path = os.path.join(save_folder, filename)

        # Save with compression
        img.save(save_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

        return True
    except Exception as e:
        print(f"❌ Error processing {sample_id} from {image_url}\n{e}")
        return False

# === MAIN SCRIPT ===

if __name__ == "__main__":
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Read batch
    batch_entries = read_entries(EXCEL_FILE_PATH, COLUMN_IMAGE_LINK, COLUMN_SAMPLE_ID,
                                 start=START_INDEX, limit=BATCH_SIZE)

    print(f"📄 Loaded {len(batch_entries)} entries from row {START_INDEX} to {START_INDEX + BATCH_SIZE}")

    # Skip existing files
    processed_ids = get_existing_sample_ids(SAVE_FOLDER)
    entries_to_process = [(url, sid) for url, sid in batch_entries if sid not in processed_ids]

    print(f"📦 Images to download + compress: {len(entries_to_process)}")

    for url, sid in tqdm(entries_to_process, desc="Downloading + Processing"):
        success = download_resize_compress(url, sid, SAVE_FOLDER)
        if not success:
            print("🛑 Stopping due to download/processing failure.")
            break

    print("\n✅ All available images in this batch downloaded and compressed.")
