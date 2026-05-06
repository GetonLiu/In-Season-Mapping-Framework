import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
import time
from tqdm import tqdm
import json
import random

# ================= CONFIGURATION AREA =================
DATA_ROOT = r"D:\Inseason_mapping"
OUTPUT_DIR = r"D:\Inseason_mapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Spectral Indices
STANDARD_INDICES = ['EVI', 'NDVI', 'NDWI', 'NPCI', 'PPR', 'RVI', 'SAVI']
ALL_INDICES = STANDARD_INDICES

# 2. Crop Class Mapping
CROP_MAPPING = {
    'Cotton': 0, 'SCorn': 1, 'SWheat': 2,
    'DWheat': 3, 'Agroforestry': 4, 'Other': 5
}
YEARS = range(2018, 2025)
FILE_SUFFIX = "_Norm"


# ================= 🛠️ UTILITY FUNCTIONS =================
def get_points_from_shp(year, target_crs):
    shp_pattern = os.path.join(DATA_ROOT, f"Export_Train_*_{year}.shp")
    shp_files = glob.glob(shp_pattern)
    all_points, all_labels = [], []
    if not shp_files: return [], []

    for shp_path in shp_files:
        filename = os.path.basename(shp_path)
        try:
            crop_name = filename.split('Train_')[1].split(f'_{year}')[0]
        except:
            continue
        if crop_name not in CROP_MAPPING: continue

        gdf = gpd.read_file(shp_path)
        if gdf.crs != target_crs: gdf = gdf.to_crs(target_crs)
        coords = [(p.x, p.y) for p in gdf.geometry]
        all_points.extend(coords)
        all_labels.extend([CROP_MAPPING[crop_name]] * len(coords))
    return all_points, all_labels


def get_file_path(year, index_name):
    if index_name in STANDARD_INDICES:
        return os.path.join(DATA_ROOT, f"EMD_{year}_{index_name}.tif")
    return None


# ================= 🚀 CORE PIPELINE =================
def calculate_stats():
    """Step 1: Calculate global maximum for RVI"""
    print("🚀 [Step 1] Scanning feature distribution (RVI: Max)...")

    collected_rvi = []

    ref_path = get_file_path(2018, 'EVI')
    if not os.path.exists(ref_path): return None
    with rasterio.open(ref_path) as src:
        target_crs = src.crs

    MAX_SAMPLES = 50000

    for year in tqdm(YEARS, desc="Sampling Stats"):
        points, _ = get_points_from_shp(year, target_crs)
        if not points: continue

        if len(points) > MAX_SAMPLES:
            import random
            points = random.sample(points, MAX_SAMPLES)

        path_rvi = get_file_path(year, 'RVI')
        if os.path.exists(path_rvi):
            with rasterio.open(path_rvi) as src:
                val = np.array(list(src.sample(points)))
                if val.shape[1] > 18: val = val[:, :18]
                collected_rvi.append(val.flatten())

    stats = {}

    if collected_rvi:
        all_rvi = np.concatenate(collected_rvi)
        max_rvi = np.max(all_rvi)
        stats['RVI'] = float(max_rvi)
        print(f"  👉 RVI Max: {max_rvi:.4f}")
    else:
        print("⚠️ No RVI data collected. Cannot calculate maximum.")
        return None

    return stats


def process_norm():
    stats = calculate_stats()
    if not stats: return

    print("\n🚀 [Step 2] Executing normalization (EVI filtering only)...")

    ref_path = get_file_path(2018, 'EVI')
    with rasterio.open(ref_path) as src:
        target_crs = src.crs

    for year in tqdm(YEARS, desc="Processing"):
        points, labels = get_points_from_shp(year, target_crs)
        if not points: continue

        features_dict = {}
        valid_year = True

        for idx_name in ALL_INDICES:
            tif_path = get_file_path(year, idx_name)
            if not os.path.exists(tif_path): valid_year = False; break
            with rasterio.open(tif_path) as src:
                val = np.array(list(src.sample(points)))
                if val.shape[1] > 18: val = val[:, :18]
                features_dict[idx_name] = val

        if not valid_year: continue

        X_raw = np.stack([features_dict[idx] for idx in ALL_INDICES], axis=1)
        y_raw = np.array(labels)

        # ================= 🔥 DATA FILTERING 🔥 =================
        evi_idx = ALL_INDICES.index('EVI')

        mask_evi_abnormal = np.any((X_raw[:, evi_idx, :] > 1) | (X_raw[:, evi_idx, :] < -1), axis=1)
        mask_keep = ~mask_evi_abnormal

        X_clean = X_raw[mask_keep]
        y_clean = y_raw[mask_keep]

        if len(y_clean) == 0:
            print(f"⚠️ {year}: All data filtered out due to EVI out of range.")
            continue

        # ================= 🔥 NORMALIZATION & CLIPPING 🔥 =================
        X_norm = X_clean.copy()

        rvi_idx = ALL_INDICES.index('RVI')
        X_norm[:, rvi_idx, :] = np.clip(
            X_clean[:, rvi_idx, :] / (stats['RVI'] + 1e-6),
            0.0, 1.0
        )

        for name in ALL_INDICES:
            if name != 'RVI':
                idx = ALL_INDICES.index(name)
                X_norm[:, idx, :] = np.clip(X_clean[:, idx, :], -1.0, 1.0)

        out_x = f"X_{year}{FILE_SUFFIX}.npy"
        out_y = f"y_{year}{FILE_SUFFIX}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_x), X_norm.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, out_y), y_clean.astype(np.int64))

        drop_rate = 100 * (1 - len(y_clean) / len(y_raw))
        print(f"  ✅ {year}: Saved {len(y_clean)} samples. Dropped {drop_rate:.1f}% (EVI only).")

    # ================= 🔥 SAVE PARAMETERS JSON 🔥 =================
    norm_params = {}
    for name in ALL_INDICES:
        if name == 'RVI':
            norm_params[name] = {'divisor': stats['RVI'], 'action': 'div_max_clip_0_1'}
        else:
            norm_params[name] = {'divisor': 1.0, 'action': 'clip_minus1_1'}

    json_path = os.path.join(OUTPUT_DIR, "Sample_2018_2024.json")
    with open(json_path, 'w') as f:
        json.dump(norm_params, f, indent=4)
    print(f"\n📄 Normalization params saved to: {json_path}")


if __name__ == "__main__":
    start = time.time()
    process_norm()
    print(f"\n🎉 Done! Time: {(time.time() - start) / 60:.2f} min")