import os
import pandas as pd
from PIL import Image

# -------------------------
# CONFIG — CHANGE IF NEEDED
# -------------------------
CSV_PATH = "dataset/train.csv"     # or TrainClean_Verified.csv based on your structure
TEST_CSV_PATH = "dataset/test.csv"
IMG_DIR = "images"                 # or "65k-75k" for Kaggle
IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

# -------------------------


def check_csv_exists(path):
    print(f"\n🔍 Checking CSV exists: {path}")
    if os.path.exists(path):
        print("   ✔ Found")
        return True
    else:
        print("   ❌ NOT FOUND!")
        return False


def check_required_columns(df, required_cols):
    print("\n🔍 Checking required columns...")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("   ❌ Missing columns:", missing)
    else:
        print("   ✔ All required columns present")
    return missing


def check_missing_values(df, important_cols):
    print("\n🔍 Checking missing values...")
    for col in important_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"   ⚠ {col} has {nan_count} missing values")
        else:
            print(f"   ✔ {col}: No missing values")


def check_price_validity(df):
    print("\n🔍 Checking price sanity...")
    df2 = df[df["price"] > 0]

    if len(df2) != len(df):
        print(f"   ⚠ Found {len(df) - len(df2)} rows with invalid/non-positive price.")
    else:
        print("   ✔ All prices are > 0")

    print(f"\n   Price Stats:")
    print(f"   → Min: {df['price'].min():.3f}")
    print(f"   → Max: {df['price'].max():.3f}")
    print(f"   → Mean: {df['price'].mean():.3f}")
    print(f"   → Median: {df['price'].median():.3f}")


def check_duplicate_ids(df):
    print("\n🔍 Checking duplicate sample_id...")
    dup = df["sample_id"].duplicated().sum()
    if dup > 0:
        print(f"   ⚠ Duplicates found: {dup}")
    else:
        print("   ✔ No duplicate sample_ids")


def find_image_path(sid):
    for ext in IMG_EXTS:
        img_path = os.path.join(IMG_DIR, f"{sid}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None


def check_images(df):
    print("\n🔍 Checking image files...")

    missing = 0
    corrupt = 0

    for sid in df["sample_id"]:
        img_path = find_image_path(sid)

        if img_path is None:
            missing += 1
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()  # verify only (does not decode fully)
        except Exception:
            corrupt += 1

    print(f"\n   Image Summary:")
    print(f"   → Missing images: {missing}")
    print(f"   → Corrupt images: {corrupt}")
    print(f"   → Total samples: {len(df)}")


def check_distribution(df):
    print("\n📊 Dataset Distribution Summary:")
    print(df.describe(include="all"))
    print("\n📊 Price Histogram:")
    print(df["price"].describe())


# =============================
# MAIN SCRIPT
# =============================

if __name__ == "__main__":
    print("\n======================================")
    print("   DATASET CORRECTNESS CHECKER")
    print("======================================\n")

    # ---- Load train CSV ----
    if not check_csv_exists(CSV_PATH):
        exit()

    df = pd.read_csv(CSV_PATH)

    # ---- Basic checks ----
    required_columns = ["sample_id", "catalog_content", "price"]
    check_required_columns(df, required_columns)
    check_missing_values(df, ["sample_id", "catalog_content", "price"])
    check_duplicate_ids(df)
    check_price_validity(df)

    # ---- Image checks ----
    if os.path.exists(IMG_DIR):
        check_images(df)
    else:
        print(f"\n⚠ Image directory NOT FOUND: {IMG_DIR}")

    # ---- Distribution ----
    check_distribution(df)

    print("\n🎉 Dataset correctness check completed!\n")
