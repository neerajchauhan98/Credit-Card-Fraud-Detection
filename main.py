import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

df = pd.read_csv("creditcard.csv")
print("\n Dataset Loaded Successfully!")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n Dataset Info:")
print(df.info())
print("\n Checking Missing Values:")
print(df.isnull().sum())
print("\n Statistical Summary:")
print(df.describe())
print("\n Datatypes:")
print(df.dtypes.value_counts())
print("\n Columns in Dataset:")
print(df.columns.tolist())

print("\n Class Distribution:\n", df['Class'].value_counts())

plt.figure(figsize=(6,4))
ax = sns.countplot(x='Class', data=df, palette='cool')
plt.title('Fraud (1) vs Non-Fraud (0) Transactions')
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 50, int(p.get_height()), ha='center')
plt.savefig("output/class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='Class', y='Amount', data=df, palette='magma')
plt.title("Transaction Amount Distribution by Class")
plt.savefig("output/amount_vs_class.png", dpi=300, bbox_inches='tight')
plt.show()

if 'Time' in df.columns:
    plt.figure(figsize=(7,5))
    sns.histplot(df[df['Class']==0]['Time'], bins=50, color='blue', label='Non-Fraud', alpha=0.6)
    sns.histplot(df[df['Class']==1]['Time'], bins=50, color='red', label='Fraud', alpha=0.6)
    plt.legend()
    plt.title("Transaction Time Distribution (Fraud vs Non-Fraud)")
    plt.savefig("output/time_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

df = df.sample(frac=0.2, random_state=42)
print("\n Using 20% of data for faster model training:", df.shape)

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("\n Training data shape:", X_train.shape)
print(" Test data shape:", X_test.shape)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("\n After SMOTE:", X_train.shape)
print(" Fraud Class Count After SMOTE:\n", y_train.value_counts())

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='logloss')

models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
y_preds = {}

for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_preds[name] = model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"output/{name.lower().replace(' ','_')}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

for name in models.keys():
    evaluate_model(name, y_test, y_preds[name])

plt.figure(figsize=(8,6))
scores = {}
for name in models.keys():
    pred = y_preds[name]
    auc = roc_auc_score(y_test, pred)
    scores[name] = auc
    fpr, tpr, _ = roc_curve(y_test, pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison")
plt.legend()
plt.savefig("output/roc_auc_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

best_name = max(scores, key=scores.get)
best_model = models[best_name]
print(f"\n Best Model: {best_name} with AUC = {scores[best_name]:.3f}")

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='orange')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.title("Top 10 Important Features (Random Forest)")
    plt.xlabel("Feature Importance")
    plt.savefig("output/random_forest_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

sample = X_test.iloc[0:1]
pred = best_model.predict(sample)
print(f"\n Single Transaction Prediction ({best_name}):", "Fraud" if pred[0]==1 else "Not Fraud")
