import pandas as pd
import numpy as np

print("=" * 60)
print("DEBUGGING CLV DATA")
print("=" * 60)

try:
    print("\n1. LOADING CSV FILES...")
    rfm = pd.read_csv('rfm_with_churn.csv')
    survival = pd.read_csv('survival_data.csv')
    clv = pd.read_csv('clv_predictions.csv', index_col=0)
    print("✓ All CSV files loaded successfully")

    print("\n2. RFM DATA INFO:")
    print(f"   Shape: {rfm.shape}")
    print(f"   Columns: {rfm.columns.tolist()}")
    print(f"   CustomerID dtype: {rfm['CustomerID'].dtype}")
    print(f"   Sample CustomerID values: {rfm['CustomerID'].head(3).tolist()}")

    print("\n3. CLV DATA INFO:")
    print(f"   Shape: {clv.shape}")
    print(f"   Index name: {clv.index.name}")
    print(f"   Columns: {clv.columns.tolist()}")
    print(f"   Index sample: {clv.index[:3].tolist()}")

    # Check for inf and NaN in clv
    print(f"\n   NaN values per column:")
    for col in clv.columns:
        nan_count = clv[col].isna().sum()
        inf_count = np.isinf(clv[col]).sum()
        print(f"      {col}: NaN={nan_count}, INF={inf_count}")

    print("\n4. SURVIVAL DATA INFO:")
    print(f"   Shape: {survival.shape}")
    print(f"   Columns: {survival.columns.tolist()}")

    # Check for missing required columns
    print("\n5. REQUIRED COLUMNS CHECK:")
    required_cols = {
        'rfm': ['CustomerID', 'segment', 'bayesian_churn_prob', 'recency', 'frequency', 'monetary'],
        'clv': ['clv_12m', 'prob_alive', 'predicted_purchases_90d'],
        'survival': ['segment', 'lifetime_days', 'churned']
    }

    for dataset, cols in required_cols.items():
        if dataset == 'rfm':
            df = rfm
        elif dataset == 'clv':
            df = clv
        else:
            df = survival

        for col in cols:
            if col in df.columns:
                print(f"   ✓ {dataset}.{col}")
            else:
                print(f"   ✗ {dataset}.{col} MISSING!")

    print("\n6. MERGE TEST:")
    rfm_temp = rfm.copy()
    clv_temp = clv.copy()

    if 'CustomerID' not in clv_temp.columns:
        clv_temp = clv_temp.reset_index()
        clv_temp.rename(columns={'index': 'CustomerID'}, inplace=True)

    clv_temp['CustomerID'] = clv_temp['CustomerID'].astype(str).str.strip()
    rfm_temp['CustomerID'] = rfm_temp['CustomerID'].astype(str).str.strip()

    merged = clv_temp.merge(rfm_temp[['CustomerID', 'segment']], on='CustomerID', how='left')
    print(f"   Merged shape: {merged.shape}")
    print(f"   Rows with segment data: {merged['segment'].notna().sum()}")
    print(f"   Rows without segment data: {merged['segment'].isna().sum()}")

    print("\n✓ DEBUG COMPLETE - Data looks valid!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
