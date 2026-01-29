import pandas as pd

# df = pd.read_csv('nba_shots_cleaned.csv')
df = pd.read_csv('nba_shots_zone_cleaned.csv')

# X = df[['SHOT_TYPE', 'BASIC_ZONE', 'SHOT_DISTANCE', 'TEAM_NAME']]
X = df[['SHOT_TYPE', 'BASIC_ZONE', 'ZONE_NAME', 'SHOT_DISTANCE', 'TEAM_NAME', 'SHOT_ZONE_CLASS']]

y = df['SHOT_MADE']

# X = pd.get_dummies(X, columns=['SHOT_TYPE', 'BASIC_ZONE', 'TEAM_NAME'], drop_first=True)
X = pd.get_dummies(X, columns=['SHOT_TYPE', 'BASIC_ZONE', 'ZONE_NAME', 'TEAM_NAME', 'SHOT_ZONE_CLASS'], drop_first=True)


# splits data into training and testing, 80/20
############################################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train a logistic regression model
############################################################################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# evaluate logistic regression model
############################################################################
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test) # test to make predictions with trained model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # prints out True Positive, True Negative, FP and FN
print("\nClassification Report:\n", classification_report(y_test, y_pred)) # prints precision and some other data


# trains a random forest model and checks feature importance
############################################################################
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# importances = rf.feature_importances_
# feature_names = X.columns

# sorted_features = sorted(zip(importances, feature_names), reverse=True)
# for importance, name in sorted_features:
#     print(f"{name}: {importance:.4f}") # prints importance of each feature in training model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = X.columns

sorted_features = sorted(zip(importances, feature_names), reverse=True)
for importance, name in sorted_features:
    print(f"{name}: {importance:.4f}")