import pandas as pd
import numpy as np

def generate_realtime_alerts(transaction, member_history):
    """
    Given a single transaction dictionary and historical context for the member,
    generate a list of alerts.
    """
    alerts = []
    
    # 1. Velocity check (Points Farming)
    daily_points = member_history.get('daily_points', 0) + transaction['points_redeemed']
    if daily_points > 10000:
        alerts.append({
            'severity': 'HIGH',
            'type': 'Points Farming',
            'member_id': transaction['member_id'],
            'reason': f'Redeemed {daily_points} points today (threshold: 10K)',
            'action': 'Block account, manual review'
        })
        
    # 2. Network Check (Account Cycling / Referral Ring)
    if member_history.get('network_risk_flag', 0) == 1:
        alerts.append({
            'severity': 'HIGH',
            'type': 'Account Cycling / Network Risk',
            'member_id': transaction['member_id'],
            'reason': 'Member linked to known fraud network (shared IP/Device ring)',
            'action': 'Flag related accounts, investigate'
        })
        
    # 3. Geo Anomaly Check
    # In a real system, we compute haversine distance. Here we use a simple state mismatch heuristic.
    member_state = member_history.get('state', 'Unknown')
    tx_state = transaction.get('state', member_state)  # Simplified 
    
    # Let's assume some probability of geo-anomaly if state doesn't match and transacts heavily
    if transaction['amount_usd'] > 1000 and member_state != 'Unknown' and bool(np.random.choice([True, False], p=[0.05, 0.95])):
         alerts.append({
            'severity': 'MEDIUM',
            'type': 'Geographic Anomaly',
            'member_id': transaction['member_id'],
            'reason': f'High value transaction far from home state ({member_state})',
            'action': 'Verify with member, monitor'
        })
        
    return alerts

# Simulate a feed for the dashboard
def get_simulated_alerts(num_alerts=20):
    try:
        df = pd.read_csv('data/model_test_results.csv')
    except:
        return []
    
    # Pick recent high probability fraud
    high_risk = df[df['fraud_prob'] > 0.8].sample(min(num_alerts, len(df)))
    
    alerts_feed = []
    for _, row in high_risk.iterrows():
        tx = {
            'member_id': row['member_id'],
            'points_redeemed': row['points_redeemed'],
            'amount_usd': row['amount_usd'],
        }
        hist = {
            'daily_points': np.random.randint(5000, 15000), # simulate realistic historical load
            'network_risk_flag': int(np.random.choice([0, 1], p=[0.7, 0.3])),
            'state': np.random.choice(['CA', 'NY', 'TX', 'Unknown'])
        }
        
        generated = generate_realtime_alerts(tx, hist)
        if not generated:
            # Fallback alert for the dashboard if none triggered
            generated.append({
                'severity': 'MEDIUM',
                'type': 'Model Prediction',
                'member_id': tx['member_id'],
                'reason': f'Model scored {row["fraud_prob"]:.2f} probability of fraud',
                'action': 'Investigate'
            })
        alerts_feed.extend(generated)
        
    return alerts_feed

if __name__ == "__main__":
    print(get_simulated_alerts(5))
