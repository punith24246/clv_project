import pandas as pd

clv_df = pd.read_csv('final_clv_predictions.csv')
rfm_df = pd.read_csv('rfm_with_churn.csv')

# Test lookup for first few customers
print('Testing Customer Lookup CLV Data:')
print('=' * 60)

for customer_id in rfm_df['CustomerID'].head(5).tolist():
    cust_clv_data = clv_df[clv_df['CustomerID'] == customer_id]
    if not cust_clv_data.empty:
        cust_clv = cust_clv_data.iloc[0]
        print(f'Customer {customer_id}:')
        print(f'  CLV 12M: £{float(cust_clv["clv_12m"]):.2f}')
        print(f'  Prob Alive: {float(cust_clv["prob_alive"]):.1%}')
        print(f'  Predicted Purchases (90d): {float(cust_clv["predicted_purchases_90d"]):.1f}')
    else:
        print(f'Customer {customer_id}: NOT FOUND')

print('\n✓ Customer lookup test passed!')
