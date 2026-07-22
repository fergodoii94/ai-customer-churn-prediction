from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Simulated Dataset: [Age, MonthlySpend, ContractYears] -> Churn (0 or 1)
data = {
    'age': [25, 45, 30, 50, 22, 40, 35, 60],
    'spend': [50, 100, 60, 120, 40, 90, 70, 150],
    'contract': [1, 2, 1, 3, 1, 2, 1, 3],
    'churn': [0, 1, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds)*100}%")
