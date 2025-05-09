import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
data = pd.read_csv('glass.csv')
#print(data.head())
#print(data['Type'].value_counts())
#print(data.isnull().sum())

X = data.drop(['Type'],axis=1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

rf_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    random_state=42
)
rf_tuned.fit(X_train_scaled, y_train)

y_pred_tuned = rf_tuned.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

ex_tree_clf = ExtraTreesClassifier(n_estimators=100, max_features=7, random_state=42)
ex_tree_clf.fit(X_train_scaled, y_train)

ex_pred_tuned = ex_tree_clf.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, ex_pred_tuned))
print(classification_report(y_test, ex_pred_tuned))
