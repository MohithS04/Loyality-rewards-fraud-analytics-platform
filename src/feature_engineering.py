import pandas as pd
import numpy as np

def run_feature_engineering():
    print("Loading raw data...")
    members = pd.read_csv('data/members.csv')
    redemptions = pd.read_csv('data/redemptions.csv')

    # Convert dates
    members['join_date'] = pd.to_datetime(members['join_date'])
    redemptions['timestamp'] = pd.to_datetime(redemptions['timestamp'])

    print("Engineering member features...")
    # 1. Member attributes
    members['account_age_days'] = (pd.Timestamp('today') - members['join_date']).dt.days

    # Merge member features into redemptions for aggregate calculation
    df = redemptions.merge(members[['member_id', 'tier', 'account_age_days', 'city', 'state', 'email_domain', 'ip_address', 'device_id']], on='member_id', how='left')

    print("Engineering velocity & aggregation features...")
    df = df.sort_values(by=['member_id', 'timestamp'])
    
    # Time between redemptions
    df['time_since_last_redemption_h'] = df.groupby('member_id')['timestamp'].diff().dt.total_seconds() / 3600.0
    df['time_since_last_redemption_h'] = df['time_since_last_redemption_h'].fillna(-1) # First transaction
    
    # Rolling velocity points and counts (very memory intensive, simplifying for portfolio)
    # Using window approach or aggregate approach
    # Calculate global max/min/mean for each member
    member_aggregates = df.groupby('member_id').agg(
        total_redemptions=('transaction_id', 'count'),
        total_points_redeemed=('points_redeemed', 'sum'),
        avg_points_redeemed=('points_redeemed', 'mean'),
        std_points_redeemed=('points_redeemed', 'std'),
        max_points_redeemed=('points_redeemed', 'max'),
        total_value_usd=('amount_usd', 'sum'),
        avg_value_usd=('amount_usd', 'mean')
    ).reset_index()
    
    df = df.merge(member_aggregates, on='member_id', how='left')
    
    print("Engineering time-based features...")
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print("Engineering network & device sharing features...")
    # Count how many total members use the same ip or device
    ip_counts = df.groupby('ip_address')['member_id'].nunique().to_dict()
    device_counts = df.groupby('device_id')['member_id'].nunique().to_dict()
    
    df['shared_ip_count'] = df['ip_address'].map(ip_counts)
    df['shared_device_count'] = df['device_id'].map(device_counts)
    
    # Flag single IP used by multiple accounts
    df['is_shared_ip_high'] = (df['shared_ip_count'] > 2).astype(int)
    df['is_shared_device_high'] = (df['shared_device_count'] > 2).astype(int)
    
    # Categorical encodings (One Hot constraints, just encoding into categories)
    print("Encoding categorical variables for modeling...")
    feature_columns = [
        'points_redeemed', 'amount_usd', 
        'account_age_days', 'time_since_last_redemption_h',
        'total_redemptions', 'total_points_redeemed', 'avg_points_redeemed', 
        'std_points_redeemed', 'max_points_redeemed', 'total_value_usd', 'avg_value_usd',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'shared_ip_count', 'shared_device_count', 'is_shared_ip_high', 'is_shared_device_high'
    ]
    
    # Adding one-hot for tiers/categories
    cat_columns = ['tier', 'category', 'channel', 'email_domain']
    df_encoded = pd.get_dummies(df[cat_columns], drop_first=True)
    
    # Combine engineered features and targets
    final_features_df = pd.concat([df[['transaction_id', 'member_id', 'timestamp', 'is_fraud', 'fraud_type'] + feature_columns], df_encoded], axis=1)
    
    # Fill any remaining NaNs
    final_features_df = final_features_df.fillna(0)
    
    print(f"Generated data shape with {final_features_df.shape[1] - 5} features.")
    
    print("Saving feature engineered dataset...")
    final_features_df.to_csv('data/engineered_features.csv', index=False)
    print("Feature Engineering successful.")
    
if __name__ == "__main__":
    run_feature_engineering()
