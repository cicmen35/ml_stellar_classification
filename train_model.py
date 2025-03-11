import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('star_classification.csv')
print(df.head())
print(df.isnull().sum())

#feature selection
X = df.drop(columns=['obj_ID', 'class'])
y = df['class']

# encode categorical target labels (Galaxy, Star, Quasar -> 0, 1, 2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['class'])

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

# save model
joblib.dump(model, "stellar_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")     # for decoding predictions

print("Model and encoder saved successfully.")
