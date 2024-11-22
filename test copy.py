import pandas as pd
import xgboost as xgb
import numpy as np
import pandas as pd
from Preprocessing import preprocess_data
import os
import joblib
data=preprocess_data('test')

# --- Feature and Target Separation ---

loaded_model = xgb.XGBRegressor()
loaded_model.load_model('XGBoost.json')
# loaded_model = joblib.load('RandomForest.pkl')

y_pred_xgb=loaded_model.predict(data.drop(columns=["id"]))

y_pred_xgb_rounded = np.round(y_pred_xgb).astype(int)
y_pred_xgb_clipped = np.clip(y_pred_xgb_rounded, 0, 5)
results = pd.DataFrame({
    "id": data["id"],
    "price": y_pred_xgb_clipped
})

base_name = "predictions"
file_extension = ".csv"
counter = 1
while os.path.exists(f"./predictions/{base_name}{counter}{file_extension}"):
    counter += 1

results.to_csv(f"./predictions/{base_name}{counter}{file_extension}", index=False)
print(f"File saved as: {base_name}{counter}{file_extension}")