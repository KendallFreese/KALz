import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

"""
TF-IDF VECTORIZATION
"""
# load preprocessed data
df = pd.read_csv("data/preprocessed.csv")

# fill NaN with empty string just in case
df['processed_content'] = df['processed_content'].fillna('')

# configuring tf-idf vectorizer (converts text to a matrix of tf-idf features))
tfidf = TfidfVectorizer(
	min_df=5, # ignore terms in less than 5 docs
	max_df=0.9, # ignore terms in more than 90% docs
	ngram_range=(1, 1), # unigrams only
	max_features=None
)

print("Generating Feature Matrix...")

# create feature matrix (X) and target variable (y)
X = tfidf.fit_transform(df['processed_content'])
y = df['recommended']

"""
LOGISTIC REGRESSION CLASSIFIER
"""
# split data into 80% training 20% testing
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
	X, y, df.index, test_size=0.2, random_state=42
)

# initialize models (added random forest and gradient boosting for comparison)
# using class_weight='balanced' to handle any class imbalance (ex. more recomm3ended reviews than not recommended)
model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)

# cross-validation on training set for each model
# using f1_macro to balance precision/recall across classes
print("Cross-validating Models...")

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"Logistic Regression CV F1-Score: {np.mean(cv_scores):.3f}")

# training and testing for logistic reg
print("Training and Testing Logistic Regression Model...")

model.fit(X_train, y_train) # final fit

# save model
print("Saving Models and Results...")
joblib.dump(model, "MODELS/logistic_model.joblib")
joblib.dump(tfidf, "MODELS/tfidf_vectorizer.joblib")

# predict on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# save results and metrics to a csv
results_df = pd.DataFrame({
    'original_index': idx_test,
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_prob
})

metrics_df = pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1"],
    "value": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]
})

metrics_df.to_csv("OUTPUT/log_metrics.csv", index=False)
results_df.to_csv("OUTPUT/log_test_results.csv", index=False)
print("Saved to 'MODELS/' and 'OUTPUT/'.")