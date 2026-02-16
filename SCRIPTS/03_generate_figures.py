import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

""" SETUP """
# load artifacts (model, vectorizer and results)
print("Loading artifacts...")
df_results = pd.read_csv("OUTPUT/log_test_results.csv")
model = joblib.load("MODELS/logistic_model.joblib")
vectorizer = joblib.load("MODELS/tfidf_vectorizer.joblib")

# aesthetics
sns.set_style("whitegrid")

"""
MODEL EVALUATION VISUALIZED
"""

# AUC-ROC curve visualization and save to OUTPUT/ folder
fpr, tpr, _ = roc_curve(df_results['actual'], df_results['probability'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("OUTPUT/log_roc_curve.png")
plt.close()

"""
COEFFICIENT INTERPRETATION
"""
# extract feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

# dataframe to sort easier
feature_df = pd.DataFrame({'word': feature_names, 'coef': coefs})

# top 10 positive and negative words
top_pos = feature_df.sort_values('coef', ascending=False).head(10)
top_neg = feature_df.sort_values('coef', ascending=True).head(10)

# positive driver words chart
plt.figure(figsize=(6, 6))
sns.barplot(x='coef', y='word', data=top_pos, hue='word', legend=False, palette='Greens_r')
plt.title("Top 10 Words Driving 'Recommendation' (Positive Coefficients)")
plt.xlabel("Coefficient Magnitude")
plt.tight_layout()
plt.savefig("OUTPUT/log_top_positive_features.png")
plt.close()

# negative driver words chart
# take absolute value for visualization length,
# but keep color to indicate negative
plt.figure(figsize=(6, 6))
sns.barplot(x=top_neg['coef'], y='word', data=top_neg, hue='word', legend=False, palette='Reds_r')
plt.title("Top 10 Words Driving 'Not Recommended' (Negative Coefficients)")
plt.xlabel("Coefficient Magnitude")
plt.tight_layout()
plt.savefig("OUTPUT/log_top_negative_features.png")
plt.close()

"""
VISUAL ERROR ANALYSIS
"""
pred_recommended = df_results[df_results['predicted'] == 1]
pred_not_recommended = df_results[df_results['predicted'] == 0]

# pie chart 1: model predicted "recommended"
# correct = true positive (actual 1), 
# wrong = false positive (actual 0)
tp = len(pred_recommended[pred_recommended['actual'] == 1])
fp = len(pred_recommended[pred_recommended['actual'] == 0])

plt.figure(figsize=(6, 6))
plt.pie([tp, fp], labels=['Correct (True Pos)', 'Incorrect (False Pos)'], 
        autopct='%1.1f%%', colors=["#34383c", '#ff9999'], startangle=90)
plt.title("Reliability of 'Recommended' Predictions\n(Precision)")
plt.tight_layout()
plt.savefig("OUTPUT/log_pie_recommended.png")
plt.close()

# pie chart 2: model predicted "not recommended"
# correct = true negative (actual 0), 
# wrong = false negative (actual 1)
tn = len(pred_not_recommended[pred_not_recommended['actual'] == 0])
fn = len(pred_not_recommended[pred_not_recommended['actual'] == 1])

plt.figure(figsize=(6, 6))
plt.pie([tn, fn], labels=['Correct (True Neg)', 'Incorrect (False Neg)'], 
        autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'], startangle=90)
plt.title("Reliability of 'Not Recommended' Predictions\n(Negative Predictive Value)")
plt.tight_layout()
plt.savefig("OUTPUT/log_pie_not_recommended.png")
plt.close()

print("All figures generated in 'OUTPUT/' folder.")