# Citi Loyalty Rewards Fraud Analytics Platform 🛡️

## Project Overview
An end-to-end loyalty rewards fraud detection system that analyzes member behavior, identifies abuse patterns, and quantifies financial exposure. This project mirrors enterprise-grade analytics platforms used in top-tier financial organizations to process high-volume redemption networks.

By engineering a rich synthetic dataset and processing it through hybrid Machine Learning pipelines, this system demonstrates scalable fraud pattern detection pipelines, entity resolution techniques via NetworkX, and an interactive Streamlit real-time monitoring dashboard mimicking production applications at an enterprise scale.

## 📊 Business Metrics Achieved
- **Abuse Rate Quantified**: Explored a synthetic dataset with a ~1.69% anomaly rate spanning 200,000 members and 2,000,000 transactions.
- **Financial Impact Assessment**: Identified and modeled an estimated **$42,153,853** of projected annual exposure.
- **Exposure Segmentation by Merchant Category**: 
    - Travel: 35% ($29.4M)
    - Retail: 25% ($20.9M)
    - Dining: 25% ($21.2M)
    - E-Commerce: 15% ($12.6M)
- **Account Cycling Identification**: 100% recall via IP and Device linkage network graphs.
- **Ensemble Model Performance**: 
  - **Recall**: 73.53%
  - **AUC**: 0.948
  - **Actionable Insight**: Prioritized high-severity flags to mitigate maximum financial damage while maintaining a sub-5% false positive threshold (FPR: 4.3%).

## 🎯 Fraud Typologies Mitigated
- **Points Farming**: Account behavior exhibiting sudden anomalies in redemption frequency and volume. Captured via Isolation Forest anomaly detection based on point velocity.
- **Account Cycling**: Identified interconnected hardware rings utilizing shared IPs and Device fingerprints to cycle promotional points. Handled via NetworkX Undirected Graph Entity Resolution.
- **Referral Manipulation & Promo Abuse**: Tracked and scored anomalous geographic vectors and temporary email domain usage across high value transactions using supervised gradient boosting.

## 🧠 Machine Learning Engine & Pipeline
### Feature Engineering
Over 25 complex aggregate, temporal, and velocity-based features were engineered to provide distinguishing dimensions:
- **Velocity**: Point limits, hourly gaps between redemptions.
- **Geo-Temporal**: Identification of redemptions severely displaced from user's origin states or conducted at irregular hours.
- **Network Extracted**: Converted high-dimensional graph connections into tabular `shared_ip_count` and `shared_device_count` risks.

### Model Selection Rationale
In loyalty fraud detection, capturing the highest volume of financial anomalies (Recall) while minimizing customer disruption (Precision) is a delicate balance. 
This system employs a **Soft Voting Classifier** combining:
1. **Logistic Regression**: High interpretability, establishing the linear baseline.
2. **Random Forest**: Capturing non-linear interactions across the categorical feature space.
3. **XGBoost**: Extreme Gradient Boosting tuned with `scale_pos_weight` to mathematically counter the 1.69% extreme class imbalance, penalizing the minority class aggressively.

## 💻 Repository Structure & Architecture

```
loyalty-fraud-detection/
├── README.md                          # Project overview, setup instructions
├── data/
│   ├── generate_data.py               # Synthetic data generation logic (Generates Members & Redemptions)
│   ├── engineered_features.csv        # Persisted outputs of the Feature Engineering layer
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory data analysis (Fraud Distributions & Visualizations)
│   ├── 02_Feature_Engineering.ipynb   # Feature creation markdown/summary
│   ├── 03_Fraud_Detection.ipynb       # ML models execution and experimentation playground
│   └── 04_Network_Analysis.ipynb      # Graph analytics playground
├── src/
│   ├── feature_engineering.py         # Derives velocity and geo-temporal attributes
│   ├── fraud_detection.py             # Isolation Forest & XGBoost pipelines
│   ├── network_analysis.py            # Entity linking via NetworkX
│   ├── alert_system.py                # Real-time inference alerts simulation logic
│   └── exposure_calculation.py        # Financial metrics engine formatting JSON for Dashboards
├── dashboards/
│   ├── app.py                         # Interactive Streamlit dashboard
│   ├── fraud_monitoring.pbix          # Power BI portfolio placeholder
│   └── tableau_dashboard.twbx         # Tableau portfolio placeholder
├── reports/
│   ├── Executive_Summary.md           # Business impact, deployment methodologies
│   └── Technical_Documentation.md     # Engineering pipelines and ML rationale
└── requirements.txt                   # Complete Python environment dependencies
```

## 🚀 Quick Start Guide
To run the end-to-end pipeline and launch the monitoring dashboard locally, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Synthetic Source Data:**
   *(Note: Generates 2M rows representing 2 years of history. Operations take ~2 minutes depending on CPU).*
   ```bash
   python data/generate_data.py
   ```

3. **Execute the Intelligence Pipeline:**
   *(Run sequentially to format features, score nodes, train models, and serialize outputs)*
   ```bash
   python src/feature_engineering.py
   python src/network_analysis.py
   python src/fraud_detection.py
   python src/exposure_calculation.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run dashboards/app.py
   ```
   Navigate to `localhost:8501` to view your live executive monitoring interface.
