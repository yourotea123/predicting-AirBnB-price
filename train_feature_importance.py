import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from Preprocessing import preprocess_data
import sys
import shap
import numpy as np

# --- Preprocess and Load Data ---
data = preprocess_data('train')

# --- Feature and Target Separation ---

loaded_model = xgb.XGBRegressor()
loaded_model.load_model('XGBoost.json')

# Get feature importance
feature_importances = loaded_model.feature_importances_
data = data.drop('price', axis=1)

# Extract feature importance values for each type
importance_types = ['weight', 'gain', 'cover']
importance_dataframes = {}

for importance_type in importance_types:
    # Get feature importance
    importance = loaded_model.get_booster().get_score(importance_type=importance_type)
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=False).head(20)
    importance_dataframes[importance_type] = importance_df

# Plot each importance type
for importance_type, df in importance_dataframes.items():
    plt.figure(figsize=(10, 6))
    plt.barh(df['Feature'], df['Importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top 20 Feature Importance ({importance_type.capitalize()})')
    plt.gca().invert_yaxis()  # Most important feature on top
    plt.show()
    
    
# SHAP--------------------------------------
# Check for missing or invalid values
data.fillna(0, inplace=True)  # Handle missing values
data.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values

# Ensure data matches model features
data = data[loaded_model.feature_names_in_]  # Use features in the model
data = data.astype(float)  # Ensure all columns are numeric

# Create SHAP explainer
explainer = shap.Explainer(loaded_model, data)
shap_values = explainer(data)

# Visualize SHAP results
shap.summary_plot(shap_values, data)