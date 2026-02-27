import pandas as pd

def calculate_exposure():
    print("Loading engineered features...")
    df = pd.read_csv('data/engineered_features.csv')
    
    total_transactions = len(df)
    fraud_transactions = len(df[df['is_fraud'] == 1])
    
    # Abuse Rate Calculation
    abuse_rate = (fraud_transactions / total_transactions) * 100
    
    # Financial Exposure Quantification
    fraud_redemptions = df[df['is_fraud'] == 1]
    
    # Exposure by category (need original category names before one-hot encoding for display)
    # Reconstructing from dummy variables if necessary, but we can just use the original dataset instead
    raw_df = pd.read_csv('data/redemptions.csv')
    raw_fraud = raw_df[raw_df['is_fraud'] == 1]
    
    exposure_by_category = raw_fraud.groupby('category').agg({
        'amount_usd': 'sum',
        'points_redeemed': 'sum'
    })
    
    # Annual Exposure Projection
    # Dataset spans 2 years
    fraud_daily_avg = raw_fraud['amount_usd'].sum() / 730
    annual_exposure = fraud_daily_avg * 365
    
    print("=" * 40)
    print("FINANCIAL EXPOSURE & ABUSE RATE")
    print("=" * 40)
    print(f"Total Annual Exposure: ${annual_exposure:,.0f} (Target ~$67K)")
    print(f"Abuse Rate: {abuse_rate:.2f}% (Target 2.3%)")
    print("-" * 40)
    
    for category in exposure_by_category.index:
        cat_exposure = exposure_by_category.loc[category, 'amount_usd']
        cat_pct = (cat_exposure / raw_fraud['amount_usd'].sum()) * 100
        print(f"{category.title()} Exposure: ${cat_exposure:,.0f} ({cat_pct:.0f}%)")
    
    print("-" * 40)
    exposure_summary = raw_fraud.groupby('fraud_type')['amount_usd'].sum()
    for ftype, exp in exposure_summary.items():
        if ftype != 'none':
            print(f"{ftype.title()} Exposure: ${exp:,.0f}")
        
    print("=" * 40)
    
    # Write summary to a JSON file for the dashboard
    import json
    import os
    os.makedirs('data/metrics', exist_ok=True)
    
    metrics = {
        'total_annual_exposure': float(annual_exposure),
        'abuse_rate': float(abuse_rate),
        'category_exposure': exposure_by_category['amount_usd'].to_dict(),
        'type_exposure': exposure_summary.to_dict()
    }
    
    with open('data/metrics/exposure_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("Metrics saved to data/metrics/exposure_metrics.json")
    
if __name__ == "__main__":
    calculate_exposure()
