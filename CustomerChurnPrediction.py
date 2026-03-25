
# STEP 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# STEP 2: Load Dataset

data = pd.read_csv("churn.csv")

# STEP 3: Data Preprocessing
data.drop("CustomerID", axis=1, inplace=True)

le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Churn"] = le.fit_transform(data["Churn"])

# STEP 4: EDA (IMPORTANT)

# Histogram
data.hist(figsize=(10,8))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# STEP 5: Features & Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# STEP 6: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 7: Models

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n====================")
    print(name)
    print("====================")
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    results[name] = accuracy_score(y_test, y_pred)

# STEP 8: Compare Models
plt.bar(results.keys(), results.values())
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()
print(data.head())
