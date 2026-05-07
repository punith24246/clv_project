import pandas as pd

rfm = pd.read_csv('rfm_with_churn.csv')
clv = pd.read_csv('final_clv_predictions.csv')

print('ACTUAL PROJECT DATA')
print('=' * 60)
print('\n1. Segments:')
print(rfm['segment'].value_counts().sort_values(ascending=False))

print('\n2. Dataset Info:')
print(f'   Total Customers: {len(rfm)}')
print(f'   CLV Records: {len(clv)}')

print('\n3. CLV Statistics:')
print(f'   Total CLV: £{clv["clv_12m"].sum():,.2f}')
print(f'   Avg CLV: £{clv["clv_12m"].mean():,.2f}')
print(f'   Max CLV: £{clv["clv_12m"].max():,.2f}')

print('\n4. Churn Rate:')
print(f'   Est. Churn Rate: {(rfm["bayesian_churn_prob"] > 0.5).mean():.1%}')

print('\n5. Top 10% Customers:')
top_10 = clv.nlargest(int(len(clv) * 0.1), 'clv_12m')
pct = (top_10['clv_12m'].sum() / clv['clv_12m'].sum()) * 100
print(f'   Contribution: {pct:.0f}%')
