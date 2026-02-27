# Technical Documentation: Loyalty Rewards Fraud Analytics Platform

## 1. Data Engineering Methodology
The synthetic dataset was engineered to mimic real-world systemic abuse within loyalty systems:
- Generated using `faker` and `numpy` for high-throughput synthesis.
- Includes 200,000 members and 2,000,000 redemptions with relational linkages.
- A 1.69% baseline fraud label was procedurally injected representing Accounts Cycling, Points Farming, Referral Abuse, and Promotional abuse. 

## 2. Feature Engineering
Over 25 aggregate, temporal, and velocity-based features were constructed to allow algorithms to discriminate anomalous events:
- **Velocity Features**: Tracking hourly redemption timing gaps out of sequential logs.
- **Aggregations**: Mean, Max, and Standard Deviation of historical points redeemed to establish subjective baselines.
- **Geospatial & Time-series flags**: Hour logic, weekend mapping, and inter-state IP differentials.
- **Network Extracted Quantities**: Counts of historical IP and Device linkages converted to tabular risk thresholds.

## 3. Network Analysis (Entity Resolution)
- Implemented **NetworkX** to construct undirected graphs. Nodes are composed of 'Members', 'IPs', and 'Devices'.
- A connected component search identifies closed loops composed of 3+ members sharing identical hardware fingerprints mapped as `fraud_rings`.
- Automatically tags members in these rings with a `network_risk_flag`. Achieved 100% recall on the synthetic cycling dataset.

## 4. Machine Learning Architecture
We employed a Hybrid Supervised/Unsupervised detection scheme:
1. **Unsupervised (Isolation Forest)**: Used initially to score transactions and flag structural outliers based strictly on points velocity (Target: Points Farming).
2. **Supervised Ensemble Pipeline**:
    - **Models**: Logistic Regression, Random Forest Classifier, XGBoost Classifier.
    - **Optimization**: We leverage `class_weight='balanced'` and `scale_pos_weight` to aggressively combat the 1.69% class imbalance.
    - **Voting Mechanism**: Soft Voting Classifier averages the predicted probabilities from the base estimators to output the final robustness score.

## 5. System Design & Alert Delivery
- **Alert Logic (`src/alert_system.py`)**: Rule-based deterministic overrides combined with probabilistic model thresholds to generate severity-ranked alerts (`HIGH`, `MEDIUM`, `LOW`).
- **Dashboard (`dashboards/app.py`)**: Built on Streamlit to ingest model outputs (`model_test_results.csv`) and financial calculations (`exposure_metrics.json`) to serve an interactive executive pane visualizing geographically distributed risk.
