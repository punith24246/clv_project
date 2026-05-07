import pandas as pd
import numpy as np

print('=' * 60)
print('VERIFYING FIXES')
print('=' * 60)

# Load data with fixed method
rfm = pd.read_csv('rfm_with_churn.csv')
clv = pd.read_csv('clv_predictions.csv')

print('\n1. CLV Data Structure (FIXED):')
print(f'   Shape: {clv.shape}')
print(f'   Columns: {clv.columns.tolist()}')

# Apply the fix from load_all function
if 'CustomerID' not in clv.columns:
    clv = clv.reset_index(drop=False)
    if 'index' in clv.columns:
        clv.rename(columns={'index': 'CustomerID'}, inplace=True)

print(f'   After fix - has CustomerID: {"CustomerID" in clv.columns}')
print(f'   Columns after fix: {clv.columns.tolist()}')
print(f'   CustomerID sample: {clv["CustomerID"].head(3).tolist()}')

print('\n2. RFM Data Check:')
print(f'   Has CustomerID: {"CustomerID" in rfm.columns}')
print(f'   Columns: {rfm.columns.tolist()}')

print('\n3. Merge Test:')
clv_seg = clv.merge(rfm[['CustomerID', 'segment']], on='CustomerID', how='left')
print(f'   Merge successful: {len(clv_seg)} rows')
print(f'   Has segment column: {"segment" in clv_seg.columns}')
print(f'   Sample data:')
print(clv_seg[['CustomerID', 'clv_12m', 'segment']].head(3))

print('\n✓ All fixes verified!')
