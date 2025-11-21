import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"customer_churn_dataset-testing-master.csv")  

print("Dataset preview:")
print(df.head(), "\n")

categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df = df.drop(columns=['CustomerID', 'Last Interaction'])

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients):")
print(coefficients)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
plt.title('Feature Importance based on Logistic Regression Coefficients')
plt.show()

# Problem Description
# Build a logistic regression model to predict customer churn based on demographics, usage, and subscription details.

# Dataset Description
# Dataset includes features like Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Total Spend, and Churn (target: 1 = churned, 0 = not churned).

# Inference
# High churn linked to more support calls and payment delays.
# Longer tenure and higher spending reduce churn.
# Model achieved ~88% accuracy.
# Key features: Support Calls (+), Payment Delay (+), Tenure (−), Total Spend (−).

# Result
# Customers with frequent complaints or payment delays are more likely to churn, while loyal, high-spending users tend to stay.

# High number of support calls

# Frequent payment delays

# Low tenure or engagement

# Businesses can reduce churn by addressing customer issues proactively, offering loyalty rewards, or improving billing experiences.
