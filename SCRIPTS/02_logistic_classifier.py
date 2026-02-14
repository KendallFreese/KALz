import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

"""
STEP 3: TF-IDF VECTORIZATION
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
STEP 4: LOGISTIC REGRESSION CLASSIFIER
"""
# split data into 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# initialize models (added random forest and gradient boosting for comparison)
# using class_weight='balanced' to handle any class imbalance (ex. more recomm3ended reviews than not recommended)
lr_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# cross-validation on training set for each model
# using f1_macro to balance precision/recall across classes
print("Cross-validating Models...")

for name, model in [("Logistic Regression", lr_model),
					("Random Forest", rf_model),
					("Gradient Boosting", gb_model)]:
	
	cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
	print(f"{name} CV F1-Score: {np.mean(cv_scores):.3f}")

# training and testing for logistic reg
print("Training and Testing Logistic Regression Model...")

lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)

print("\n--- Final Logistic Regression Results ---")
print(f"Test AUC Score: {auc_score:.3f}")
print(report)

# extract coefficients (feature importance) from logistic reg
feature_names = tfidf.get_feature_names_out()
coefs = lr_model.coef_[0]
sorted_idx = np.argsort(coefs)
top_positive = [feature_names[i] for i in sorted_idx[-20:]][::-1]
top_negative = [feature_names[i] for i in sorted_idx[:20]]

print(f"Top Recommendation Drivers: {top_positive[:5]}")
print(f"Top Negative Drivers: {top_negative[:5]}")

# save model results to outputs/
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"logistic_results_{timestamp}.txt"

with output_file.open("w", encoding="utf-8") as f:
	f.write("--- Final Logistic Regression Results ---\n")
	f.write(f"Test AUC Score: {auc_score:.3f}\n\n")
	f.write(report + "\n")
	f.write(f"Top Recommendation Drivers: {top_positive[:5]}\n")
	f.write(f"Top Negative Drivers: {top_negative[:5]}\n")

print(f"Saved results to {output_file}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Assuming y_test and y_pred / y_prob are already defined from your C=0.1 model
print("--- Classification Report ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {auc_score:.4f}")

# Optional: Confusion Matrix nicely formatted
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")