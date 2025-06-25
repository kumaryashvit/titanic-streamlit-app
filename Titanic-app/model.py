import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Simple preprocessing
df = df[["Survived", "Pclass", "Sex", "Age"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Features and target
X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'titanic-model.joblib')