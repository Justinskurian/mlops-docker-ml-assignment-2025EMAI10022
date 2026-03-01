import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

# Load dataset
penguins = fetch_openml(name="penguins", version=1, as_frame=True)
df = penguins.frame.dropna()

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("species", axis=1)
y = df["species"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

precision = precision_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

print("Precision:", precision)
print("AUC Score:", auc)
