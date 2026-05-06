import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration & Paths
# ============================================================================
RESULT_ROOT = r"D:\Inseason_mapping"
SHP_ROOT = r"D:\Inseason_mapping"
SAVE_DIR = r"F:\Inseason_mapping"

os.makedirs(SAVE_DIR, exist_ok=True)

# Subdirectory for confusion matrices to keep the root directory clean
CM_DIR = os.path.join(SAVE_DIR, "Confusion_Matrices")
os.makedirs(CM_DIR, exist_ok=True)

RANDOM_SEED = 910
YEARS_TO_VALIDATE = [2023, 2024]
MODES = ['Prior']
POOL_SIZE_PER_CLASS = 3000
EARLY_WHEAT_SUB_SAMPLE = 1500

# Original SHP name mapping (Do not modify unless source filenames change)
SHP_MAPPING = {'Cotton': 0, 'SCorn': 1, 'SWheat': 2, 'DWheat': 3, 'Agroforestry': 4, 'Others': 5}

# Standardized English names for output CSVs
ID_TO_NAME = {
    0: 'Cotton',
    1: 'SCorn',
    2: 'SWheat',
    3: 'DWheat',
    4: 'Agroforestry',
    5: 'Others',
    99: 'Early_Wheat'
}

UNKNOWN_ID = 6
NODATA_ID = 255
TEST_STEPS = list(range(4, 19))
CLASS_START_STEPS = {0: 5, 4: 6, 99: 4, 1: 7, 5: 7, 3: 12, 2: 12}

DATES = {
    2023: ['Feb 17', 'Mar 04', 'Mar 19', 'Apr 03', 'Apr 18', 'May 03', 'May 18', 'Jun 02', 'Jun 17', 'Jul 02', 'Jul 17',
           'Aug 01', 'Aug 16', 'Aug 31', 'Sep 15', 'Sep 30', 'Oct 15', 'Oct 30'],
    2024: ['Feb 22', 'Mar 08', 'Mar 23', 'Apr 07', 'Apr 22', 'May 07', 'May 22', 'Jun 06', 'Jun 21', 'Jul 06', 'Jul 21',
           'Aug 05', 'Aug 20', 'Sep 04', 'Sep 19', 'Oct 04', 'Oct 19', 'Nov 01']
}


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def load_validation_pool(year):
    print(f"[*] Loading validation pool for {year}...")
    gdf_list = []
    for crop_name, class_id in SHP_MAPPING.items():
        shp_path = os.path.join(SHP_ROOT, f"Export_Train_{crop_name}_{year}.shp")
        if os.path.exists(shp_path):
            temp_gdf = gpd.read_file(shp_path)
            if len(temp_gdf) > POOL_SIZE_PER_CLASS:
                temp_gdf = temp_gdf.sample(n=POOL_SIZE_PER_CLASS, random_state=RANDOM_SEED)
            temp_gdf['True_Label'] = class_id
            gdf_list.append(temp_gdf)
    return pd.concat(gdf_list, ignore_index=True) if gdf_list else None


def extract_raster_values(gdf, raster_path):
    if not os.path.exists(raster_path):
        return None
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        coords = [(g.x, g.y) for g in gdf.geometry]
        return np.array([val[0] for val in src.sample(coords)])


def load_area_weights(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

    weights = {}
    for year in df['Year'].unique():
        weights[year] = dict(zip(df[df['Year'] == year]['Class_ID'], df[df['Year'] == year]['Proportion']))
    return weights


def prepare_step_metrics_effective(full_gdf, full_preds, step, current_weights):
    df_raw = full_gdf.copy()
    df_raw['Pred_Label'] = full_preds

    # Handle early-stage wheat classification logic
    if step <= 11:
        df_2 = df_raw[df_raw['True_Label'] == 2].sample(n=EARLY_WHEAT_SUB_SAMPLE, random_state=RANDOM_SEED)
        df_3 = df_raw[df_raw['True_Label'] == 3].sample(n=EARLY_WHEAT_SUB_SAMPLE, random_state=RANDOM_SEED)
        df_others = df_raw[~df_raw['True_Label'].isin([2, 3])]

        df_calc = pd.concat([df_others, df_2, df_3], ignore_index=True)
        df_calc.loc[df_calc['True_Label'].isin([2, 3]), 'True_Label'] = 99
        df_calc.loc[df_calc['Pred_Label'].isin([2, 3]), 'Pred_Label'] = 99
        active_ids = [cid for cid, s_step in CLASS_START_STEPS.items() if step >= s_step and cid not in [2, 3]]
    else:
        df_calc = df_raw.copy()
        active_ids = [cid for cid, s_step in CLASS_START_STEPS.items() if step >= s_step and cid != 99]

    # Calculate area-weighted proportions
    raw_weights = {cid: current_weights.get(cid, 1.0) for cid in active_ids}
    weight_sum = sum(raw_weights.values())
    W = {cid: w / weight_sum for cid, w in raw_weights.items()} if weight_sum > 0 else {}

    # Calculate Coverage
    per_class_cov = {}
    for cid in active_ids:
        sub_true = df_calc[df_calc['True_Label'] == cid]
        mapped_count = ((sub_true['Pred_Label'] != UNKNOWN_ID) & (sub_true['Pred_Label'] != NODATA_ID)).sum()
        per_class_cov[cid] = mapped_count / len(sub_true) if len(sub_true) > 0 else 0
    total_cov = sum(W.get(cid, 0) * per_class_cov.get(cid, 0) for cid in active_ids)

    # Calculate Confusion Rates and Recall
    recall_dict, confusion_rates = {}, {}
    for true_id in active_ids:
        sub_true = df_calc[df_calc['True_Label'] == true_id]
        total_true_c = len(sub_true)
        confusion_rates[true_id] = {}
        for pred_id in active_ids:
            rate = (sub_true['Pred_Label'] == pred_id).sum() / total_true_c if total_true_c > 0 else 0
            confusion_rates[true_id][pred_id] = rate
            if true_id == pred_id:
                recall_dict[true_id] = rate

    # Calculate Precision, F1, and Overall Accuracy (OA)
    f1_dict, precision_dict, oa_effective = {}, {}, 0
    for i in active_ids:
        R_i = recall_dict.get(i, 0)
        numerator = W.get(i, 0) * R_i
        denominator = sum(W.get(j, 0) * confusion_rates[j][i] for j in active_ids)
        P_i = numerator / denominator if denominator > 0 else 0

        precision_dict[i] = P_i
        f1_dict[i] = 2 * (P_i * R_i) / (P_i + R_i) if (P_i + R_i) > 0 else 0
        oa_effective += numerator

    return df_calc, total_cov, per_class_cov, oa_effective, precision_dict, recall_dict, f1_dict, active_ids


# ============================================================================
# Main Execution Pipeline
# ============================================================================
def main():
    weight_csv_path = os.path.join(SAVE_DIR, "Area_Proportions_2023_2024.csv")
    global_weights = load_area_weights(weight_csv_path) or {}

    metrics_records = []

    for year in YEARS_TO_VALIDATE:
        full_gdf = load_validation_pool(year)
        if full_gdf is None:
            print(f"[!] Warning: Validation vectors for {year} not found. Skipping.")
            continue

        current_year_weights = global_weights.get(year, {})

        for mode in MODES:
            for step in tqdm(TEST_STEPS, desc=f"Processing {year}-{mode}"):
                f_path = os.path.join(RESULT_ROOT, f"Steps_{year}_{mode}_TransformerDS",
                                      f"Result_{year}_Step_{step:02d}_{mode}.tif")
                if not os.path.exists(f_path):
                    f_path = os.path.join(RESULT_ROOT, f"Result_{year}_Step_{step:02d}_{mode}.tif")

                preds = extract_raster_values(full_gdf, f_path)
                if preds is None:
                    continue

                df_calc, t_cov, c_cov, oa_v, p_d, r_d, f1_d, active_ids = prepare_step_metrics_effective(
                    full_gdf, preds, step, current_year_weights
                )

                # 1. Record evaluation metrics
                record = {
                    'Year': year,
                    'Mode': mode,
                    'Step': step,
                    'Date': DATES[year][step - 1],
                    'OA': oa_v,
                    'Total_Coverage': t_cov
                }

                for class_id, class_name in ID_TO_NAME.items():
                    record[f'P_{class_name}'] = p_d.get(class_id, np.nan)
                    record[f'R_{class_name}'] = r_d.get(class_id, np.nan)
                    record[f'F1_{class_name}'] = f1_d.get(class_id, np.nan)
                    record[f'Cov_{class_name}'] = c_cov.get(class_id, np.nan)

                metrics_records.append(record)

                # 2. Export confusion matrix CSV (Prior mode only as an example)
                if mode == 'Prior':
                    if step <= 11:
                        cm_labels = [0, 4, 99, 1, 5]
                    else:
                        cm_labels = [0, 4, 3, 2, 1, 5]

                    cm_names = [ID_TO_NAME[i] for i in cm_labels]

                    cm = confusion_matrix(df_calc['True_Label'], df_calc['Pred_Label'], labels=cm_labels)
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"True_{n}" for n in cm_names],
                        columns=[f"Pred_{n}" for n in cm_names]
                    )

                    cm_save_path = os.path.join(CM_DIR, f"CM_Prior_{year}_Step_{step:02d}.csv")
                    cm_df.to_csv(cm_save_path)

    # 3. Save the main metrics table
    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        metrics_save_path = os.path.join(SAVE_DIR, "Validation_Metrics_All_Steps.csv")
        # Use utf-8-sig to ensure compatibility with external tools like Excel
        df_metrics.to_csv(metrics_save_path, index=False, encoding='utf-8-sig')
        print(f"\n[*] Metrics successfully exported to: {metrics_save_path}")
        print(f"[*] Confusion matrices exported to: {CM_DIR}")
    else:
        print("\n[!] Warning: No data was generated. Check input paths or TIFF files.")


if __name__ == "__main__":
    main()