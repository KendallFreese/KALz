import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# read in clean data, remove whitespace and quotes from column names
df=pd.read_csv("DATA/airport_clean.csv")
df.columns = df.columns.str.replace('"','').str.strip()

# define target and features (wanna predict overall rating based on sub-ratings)
target = "overall_rating"
feature_cols = [
    "queuing_rating",
    "terminal_cleanliness_rating",
    "airport_shopping_rating"
]

# grab features and target from df
X=df[feature_cols]
y=df[target]

# pipeline to scale features and fit linear regression model
# scaling makes sure all features are same scale and coefficients are comparable
preprocess = ColumnTransformer(
transformers=[
    ("num",StandardScaler(),feature_cols)
]
)

# define model - first preprocesses, then fits linear regression
model = Pipeline([
    ("preprocess",preprocess),
    ("reg",LinearRegression())
])

# split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2,random_state=42
)

# fit model on training data
model.fit(X_train,y_train)

# save model and metadata to use later for evaluation/analysis
artifact = {
	"model": model,
	"feature_cols": feature_cols,
	"target": target,
	"test_size": 0.2,
	"random_state": 42,
}
joblib.dump(artifact, "MODELS/linear_model.joblib")
print(f"Saved model artifact: MODELS/linear_model.joblib")