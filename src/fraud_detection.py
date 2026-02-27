import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib

def run_fraud_detection_pipeline():
    print("Loading engineered features...")
    df = pd.read_csv('data/engineered_features.csv')
    
    # 1. Isolation Forest for Points Farming (Unsupervised Anomaly Detection)
    print("Running Isolation Forest for Points Farming detection...")
    iso_features = ['total_points_redeemed', 'avg_points_redeemed', 'max_points_redeemed', 'time_since_last_redemption_h']
    
    # We fit IF on a sample to find anomalies
    iso_forest = IsolationForest(contamination=0.023, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df[iso_features].fillna(0))
    
    # -1 means anomaly
    farming_predicted = df[df['anomaly_score'] == -1]['member_id']
    farming_actual = df[df['fraud_type'] == 'farming']['member_id']
    
    farming_recall = len(set(farming_predicted).intersection(farming_actual)) / len(farming_actual) if len(farming_actual) > 0 else 0
    print(f"Points Farming Detection Recall (Isolation Forest): {farming_recall:.2%}")
    if farming_recall >= 0.80:
        print("✅ Reached >80% target for Points Farming.")
    else:
        print("❌ Did not reach 80% target for Points Farming. (Target: 80%)")
        
    df['isolation_forest_flag'] = (df['anomaly_score'] == -1).astype(int)

    # 2. Supervised ML Pipeline
    print("\nPreparing Supervised ML Model...")
    
    # Target and Features
    y = df['is_fraud']
    
    # Exclude IDs, dates, and target leakage
    exclude_cols = ['transaction_id', 'member_id', 'timestamp', 'is_fraud', 'fraud_type', 'anomaly_score']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    
    # Train Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("Training constituent models...")
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # XGBoost
    # Convert feature names to avoid JSON errors in XGB
    X_train.columns = X_train.columns.str.replace('<', '')
    X_test.columns = X_test.columns.str.replace('<', '')
    
    scale_pos_weight = sum(y_train==0)/sum(y_train==1) if sum(y_train==1) > 0 else 1.0
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=-1)
    
    # Voting Classifier Assemble
    print("Training Ensemble Voting Classifier...")
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    
    print("\nEvaluating Ensemble Model...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    print("-" * 30)
    print(f"Recall:    {recall:.2%} (Target: >89%)")
    print(f"Precision: {precision:.2%} (Target: >75%)")
    print(f"AUC:       {auc:.2%} (Target: >92%)")
    print(f"FPR:       {fpr:.2%} (Target: <5%)")
    print("-" * 30)
    
    if recall >= 0.89 and precision >= 0.75 and auc >= 0.92 and fpr <= 0.05:
         print("✅ All Model Targets Met!")
    else:
         print("⚠️ Some Model Targets Missed (Check Output). It is acceptable for highly imbalanced synthetic data.")
         
    print("\nSaving final model...")
    joblib.dump(ensemble, 'src/ensemble_fraud_model.pkl')
    
    # Save test set for dashboard
    test_df = X_test.copy()
    test_df['is_fraud'] = y_test
    test_df['fraud_prob'] = y_proba
    test_df['prediction'] = y_pred
    # join back member ids and amounts for business logic 
    test_df = test_df.join(df[['member_id', 'fraud_type']])
    
    test_df.to_csv('data/model_test_results.csv', index=False)
    
    print("Fraud Detection Pipeline Complete!")

if __name__ == "__main__":
    run_fraud_detection_pipeline()
