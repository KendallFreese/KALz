import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

# load model and metadata
artifact = joblib.load("MODELS/linear_model.joblib")
model = artifact["model"]
feature_cols = artifact["feature_cols"]
target = artifact["target"]
test_size = artifact["test_size"]
random_state = artifact["random_state"]

# read in clean data, remove whitespace and quotes from column names
df=pd.read_csv("DATA/airport_clean.csv")
df.columns = df.columns.str.replace('"','').str.strip()

# grab features and target from df
X=df[feature_cols]
y=df[target]

# split data into train and test sets, don't need train set
_, X_test, _, y_test = train_test_split(
    X,y, test_size=test_size,random_state=random_state
)

# predict on test set
preds = model.predict(X_test)
residuals = y_test - preds

# model evaluation - calculate RMSE, MAE, R^2
# RMSE measures average prediction error in same units as target variable (lower is better)
# R^2 measures proportion of variance in target explained by model (higher is better, max 1)
# MAE measures average absolute prediction error (lower is better)
rmse = np.sqrt(mean_squared_error(y_test,preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Model Evaluation Metrics:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R^2: {r2:.3f}")

# permutation importance
# - how much model performance decreases when we randomly shuffle each feature
perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
	scoring="r2"
)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": perm.importances_mean
    
}).sort_values("importance",ascending=False)

# save importance numbers
importance_df.to_csv("OUTPUT/lin_permutation_importance.csv", index=False)

# plot importance and save to file
# simple horizontal bar chart of 3 features
plt.figure(figsize=(8,4))
plt.barh(importance_df["feature"],importance_df["importance"],color="skyblue")
plt.xlabel("Importance (drop in model performance)")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("OUTPUT/lin_permutation_importance.png")
plt.close()

# plot residuals and save to file
# graph showing difference between actual and predicted values
plt.scatter(preds, residuals, alpha=0.4, color="#55a868")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Predicted Overall Rating")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.savefig("OUTPUT/lin_residuals_vs_predicted.png", dpi=300)
plt.close()

print(f"Saved tables and figures to: OUTPUT/")