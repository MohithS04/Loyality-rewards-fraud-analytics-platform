import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import classification_report, confusion_matrix

def run_network_analysis():
    print("Loading data for network analysis...")
    members = pd.read_csv('data/members.csv')
    redemptions = pd.read_csv('data/engineered_features.csv')
    
    # We will build a graph based on shared IPs and Devices
    print("Building entity resolution graph for Account Cycling...")
    G = nx.Graph()
    
    # Adding nodes and edges
    # To avoid massive graphs, we only add edges for IPs and Devices that are shared
    ip_counts = members['ip_address'].value_counts()
    shared_ips = ip_counts[ip_counts > 1].index
    
    device_counts = members['device_id'].value_counts()
    shared_devices = device_counts[device_counts > 1].index
    
    # Filter members that share IPs or Devices
    shared_members = members[members['ip_address'].isin(shared_ips) | members['device_id'].isin(shared_devices)]
    
    print(f"Processing {len(shared_members)} members with shared attributes...")
    
    for _, row in shared_members.iterrows():
        member_node = f"M_{row['member_id']}"
        G.add_node(member_node, type='member')
        
        if row['ip_address'] in shared_ips:
            ip_node = f"IP_{row['ip_address']}"
            G.add_node(ip_node, type='ip')
            G.add_edge(member_node, ip_node)
            
        if row['device_id'] in shared_devices:
            dev_node = f"DEV_{row['device_id']}"
            G.add_node(dev_node, type='device')
            G.add_edge(member_node, dev_node)

    print("Identifying connected components (potential fraud rings)...")
    components = list(nx.connected_components(G))
    fraud_rings = [c for c in components if len([n for n in c if n.startswith('M_')]) >= 3]
    
    print(f"Found {len(fraud_rings)} potential fraud rings (3+ members).")
    
    # Tag members in fraud rings
    ring_members = set()
    for ring in fraud_rings:
        for node in ring:
            if node.startswith('M_'):
                ring_members.add(int(node.replace('M_', '')))
                
    # Evaluate Account Cycling detection
    cycling_actual = redemptions[redemptions['fraud_type'] == 'cycling']['member_id'].unique()
    
    true_positives = len(set(cycling_actual).intersection(ring_members))
    recall = true_positives / len(cycling_actual) if len(cycling_actual) > 0 else 0
    
    print(f"Account Cycling Detection Recall: {recall:.2%}")
    if recall >= 0.75:
        print("✅ Reached >75% target for Account Cycling.")
    else:
        print("❌ Did not reach 75% target for Account Cycling.")
        
    print("Saving network analysis results...")
    
    # Add network risk score to members in rings
    network_risk_df = pd.DataFrame({'member_id': list(ring_members), 'network_risk_flag': 1})
    
    redemptions = redemptions.merge(network_risk_df, on='member_id', how='left')
    redemptions['network_risk_flag'] = redemptions['network_risk_flag'].fillna(0)
    
    redemptions.to_csv('data/engineered_features.csv', index=False)
    print("Network risk flags added to features.")
    print("Account Cycling/Referral Network Analysis complete.")

if __name__ == "__main__":
    run_network_analysis()
