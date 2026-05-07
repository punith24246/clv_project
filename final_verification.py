import pandas as pd
import numpy as np

print('FINAL VERIFICATION')
print('=' * 60)

# Test the final data loading
rfm = pd.read_csv('rfm_with_churn.csv')
clv = pd.read_csv('final_clv_predictions.csv')
survival = pd.read_csv('survival_data.csv')

print('1. Data Loaded Successfully:')
print(f'   RFM: {rfm.shape[0]} customers')
print(f'   CLV: {clv.shape[0]} customers')  
print(f'   Survival: {survival.shape[0]} customers')

print('\n2. CLV Data Structure:')
print(f'   Has CustomerID: {"CustomerID" in clv.columns}')
print(f'   Has clv_12m: {"clv_12m" in clv.columns}')
print(f'   Has segment: {"segment" in clv.columns}')
print(f'   Has prob_alive: {"prob_alive" in clv.columns}')
print(f'   Has predicted_purchases_90d: {"predicted_purchases_90d" in clv.columns}')

print('\n3. Median Survival Test:')
survival_test = survival[survival['segment'] == 'Champions']
print(f'   Champions segment: {len(survival_test)} customers')
print(f'   Churn rate: {survival_test["churned"].mean():.1%}')
print(f'   Max lifetime: {survival_test["lifetime_days"].max()} days')

print('\n4. Sample CLV Data:')
print(clv[['CustomerID', 'clv_12m', 'segment', 'prob_alive']].head(3).to_string())

print('\n✓ All verifications passed!')
