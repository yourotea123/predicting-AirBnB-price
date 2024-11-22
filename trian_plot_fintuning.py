from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from Preprocessing import preprocess_data
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap


# --- Preprocess Data ---
data=preprocess_data('train')
data_X = data.drop('price', axis=1)
data_Y = data['price']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

# --- Plot Parameter Tuning Graph for XGBoost ---
xgb_n_estimators_range = [50, 150, 250, 350, 450]
xgb_val_errors = []

for n in xgb_n_estimators_range:
    xgb_model = xgb.XGBRegressor(n_estimators=n, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_y_pred = xgb_model.predict(X_test)
    xgb_val_errors.append(mean_squared_error(y_test, xgb_y_pred))


# --- Plot Parameter Tuning Graph for RandomForest ---
rf_n_estimators_range = [50, 150, 250, 350, 450]
rf_val_errors = []

for n in rf_n_estimators_range:
    rf_model = RandomForestRegressor(n_estimators=n, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_val_errors.append(mean_squared_error(y_test, rf_y_pred))
    print(n)

# --- Plotting ---
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)  # Disable shared y-axis

# Plot for XGBoost
axes[0].plot(xgb_n_estimators_range, xgb_val_errors, marker='o', label='XGBoost', color='blue')
axes[0].set_xlabel("Number of Estimators")
axes[0].set_ylabel("Validation Mean Squared Error")
axes[0].set_title("XGBoost Validation MSE")
axes[0].grid()
axes[0].legend()
axes[0].set_ylim(min(xgb_val_errors) - 0.1, max(xgb_val_errors) + 0.1)  # Set y-axis range for XGBoost

# Plot for Random Forest
axes[1].plot(rf_n_estimators_range, rf_val_errors, marker='s', label='Random Forest', color='green')
axes[1].set_xlabel("Number of Estimators")
axes[1].set_title("Random Forest Validation MSE")
axes[1].grid()
axes[1].legend()
axes[1].set_ylim(min(rf_val_errors) - 0.1, max(rf_val_errors) + 0.1)  # Set y-axis range for Random Forest

# Adjust layout
plt.tight_layout()
plt.show()