from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from Preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt



def custom_mse_scorer(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_clipped = np.clip(y_pred_rounded, 0, 5)
    mse = mean_squared_error(y_true, y_pred)
    mse_rounded = mean_squared_error(y_true, y_pred_rounded)
    mse_clipped = mean_squared_error(y_true, y_pred_clipped)
    return mse, mse_rounded, mse_clipped

def nestedCV(random_search_model, model_name):
# Perform nested cross-validation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    mse_rounded_scores = []
    mse_clipped_scores = []

    for train_index, test_index in outer_cv.split(data_X):
        X_train_outer, X_test_outer = data_X.iloc[train_index], data_X.iloc[test_index]
        y_train_outer, y_test_outer = data_Y.iloc[train_index], data_Y.iloc[test_index]
        
        # Perform inner cross-validation for hyperparameter tuning
        random_search_model.fit(X_train_outer, y_train_outer)
        best_model = random_search_model.best_estimator_
        
        # Predict on the outer test set
        y_pred_outer = best_model.predict(X_test_outer)
        
        # Calculate metrics
        mse, mse_rounded, mse_clipped = custom_mse_scorer(y_test_outer, y_pred_outer)
        mse_scores.append(mse)
        mse_rounded_scores.append(mse_rounded)
        mse_clipped_scores.append(mse_clipped)

    # Average MSE scores across outer folds
    average_mse = np.mean(mse_scores)
    average_mse_rounded = np.mean(mse_rounded_scores)
    average_mse_clipped = np.mean(mse_clipped_scores)

    print(f"{model_name} Nested Cross-Validation MSE Scores for Each Fold:", mse_scores)
    print(f"{model_name} Nested Cross-Validation Rounded MSE Scores for Each Fold:", mse_rounded_scores)
    print(f"{model_name} Nested Cross-Validation Clipped MSE Scores for Each Fold:", mse_clipped_scores)
    print(f"{model_name} Average Nested Cross-Validation MSE:", average_mse)
    print(f"{model_name} Average Nested Cross-Validation Rounded MSE:", average_mse_rounded)
    print(f"{model_name} Average Nested Cross-Validation Clipped MSE:", average_mse_clipped)
    
    # Plot MSE scores across folds
    folds = range(1, 6)
    plt.figure(figsize=(10, 6))
    plt.plot(folds, mse_scores, marker='o', label='MSE')
    plt.plot(folds, mse_rounded_scores, marker='s', label='Rounded MSE')
    plt.plot(folds, mse_clipped_scores, marker='^', label='Clipped MSE')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Nested Cross-Validation MSE Scores Across Folds')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 3. Fit and evaluate the best model on the full train-test split
    random_search_model.fit(X_train, y_train)
    best_model = random_search_model.best_estimator_
    y_pred_xgb = best_model.predict(X_test)
    y_pred_xgb_rounded = np.round(y_pred_xgb).astype(int)
    y_pred_xgb_clipped = np.clip(y_pred_xgb_rounded, 0, 5)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mse_xgb_rounded = mean_squared_error(y_test, y_pred_xgb_rounded)
    mse_xgb_clipped = mean_squared_error(y_test, y_pred_xgb_clipped)

    print(f"{model_name} Best Parameters:", random_search_model.best_params_)
    print(f"{model_name} Mean Squared Error:", mse_xgb)
    print(f"{model_name} Mean Squared Error (rounded):", mse_xgb_rounded)
    print(f"{model_name} Mean Squared Error (clipped):", mse_xgb_clipped)
    if model_name=="XGBoost":
        best_model.save_model(f"{model_name}.json")
    else:       
        joblib.dump(best_model, f'{model_name}.pkl')
    

    # Calculate success rate per class
    classes = np.unique(y_test)  # Unique class labels
    success_rates = {}

    for cls in classes:
        total = np.sum(y_test == cls)  # Total instances of the class
        correct = np.sum((y_test == cls) & (y_pred_xgb_clipped == cls))  # Correct predictions for the class
        success_rates[cls] = (correct / total * 100) if total > 0 else 0  # Success rate in percentage

    # Display success rates
    success_rate_df = pd.DataFrame(list(success_rates.items()), columns=['Class', 'Success Rate (%)'])
    print(success_rate_df)

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_xgb_clipped, alpha=0.7, edgecolors='k', label='Predictions')

    # Add the perfect prediction line
    plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Perfect Prediction')

    # Format axes
    plt.xticks(ticks=range(0, 6), labels=range(0, 6))  # Use numerical class values
    plt.yticks(ticks=range(0, 6), labels=range(0, 6))  # Use numerical class values

    # Annotate success rate on the plot
    for i, rate in success_rates.items():
        plt.text(i, i + 0.3, f'{rate:.2f}%', fontsize=10, color='green', ha='center')

    plt.xlabel('Actual Values (Class)')
    plt.ylabel('Predicted Values (Class)')
    plt.title('Actual vs Predicted Values with Success Rate per Class')
    plt.legend()
    plt.grid(True)
    plt.show()

data=preprocess_data('train')
data_X = data.drop('price', axis=1)
data_Y = data['price']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)
# ---------------- 1.XGBoost---------------------------
# RandomizedSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.5, 1],
    'reg_alpha': [0, 0.1, 1, 10],      # L1 regularization
    'reg_lambda': [1, 10, 50, 100],    # L2 regularization
    'scale_pos_weight': [1, 2, 5, 10]  # Useful if target is imbalanced
}

xgboost_model = xgb.XGBRegressor(random_state=42)
random_search_XGBoost = RandomizedSearchCV(
    estimator=xgboost_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train

nestedCV(random_search_XGBoost, "XGBoost")


#-------
# RandomizedSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

random_forest_model = RandomForestRegressor(random_state=42)
random_search_RF = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train
nestedCV(random_search_RF, "RandomForest")