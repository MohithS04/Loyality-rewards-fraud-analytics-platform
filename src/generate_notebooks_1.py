import nbformat as nbf

# Notebook 1: EDA
nb1 = nbf.v4.new_notebook()

nb1.cells = [
    nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\n\nThis notebook performs basic exploratory data analysis on the Loyalty Rewards Fraud dataset."),
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Load Data\nmembers = pd.read_csv('../data/members.csv')\nredemptions = pd.read_csv('../data/redemptions.csv')"),
    nbf.v4.new_markdown_cell("## Data Overview"),
    nbf.v4.new_code_cell("print(f\"Members: {members.shape}\")\nprint(f\"Redemptions: {redemptions.shape}\")\n\nprint(\"\\nFraud Distribution:\")\nprint(redemptions['is_fraud'].value_counts(normalize=True) * 100)\n\nprint(\"\\nFraud Types:\")\nprint(redemptions[redemptions['is_fraud']==1]['fraud_type'].value_counts())"),
    nbf.v4.new_markdown_cell("## Visualizations"),
    nbf.v4.new_code_cell("sns.countplot(data=redemptions, x='category', hue='is_fraud')\nplt.title('Transactions by Category and Fraud Status')\nplt.show()")
]

with open('notebooks/01_EDA.ipynb', 'w') as f:
    nbf.write(nb1, f)

# Notebook 2: Feature Engineering (Documentation of what feature_engineering.py did)
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# Feature Engineering\n\nThis notebook outlines the feature engineering process used to create the final analytical dataset. The actual implementation is in `src/feature_engineering.py`."),
    nbf.v4.new_code_cell("import pandas as pd\n\n# Loading the already processed data\nfeatures_df = pd.read_csv('../data/engineered_features.csv')\nfeatures_df.head()"),
    nbf.v4.new_markdown_cell("## Overview of Engineered Features\n\nWe added several types of features:\n- **Velocity features**: `time_since_last_redemption_h`, `total_redemptions`, `avg_points_redeemed`\n- **Time-based features**: `hour_of_day`, `day_of_week`, `is_weekend`\n- **Network features**: `shared_ip_count`, `shared_device_count`, `is_shared_ip_high`"),
    nbf.v4.new_code_cell("print(\"Engineered Columns:\")\nfor col in features_df.columns:\n    print(f\"- {col}\")")
]

with open('notebooks/02_Feature_Engineering.ipynb', 'w') as f:
    nbf.write(nb2, f)
    
print("Notebooks 1 and 2 generated.")
