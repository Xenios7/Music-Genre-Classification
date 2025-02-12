from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
df = pd.read_csv("music_features.csv")

X = df.drop(columns=["genre"])  # Features (all columns except "genre")
y = df["genre"]  # Labels (Genres)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 80% Training Data (X_train, y_train) → Used for training the model.
# 20% Testing Data (X_test, y_test) → Used for evaluating performance.
# random_state=42 → Ensures that the data split is always the same every time you run

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#n_estimators specifies the number of decision trees in the random forest.
#The .fit() method is used to train a machine learning model on the given data.

# Predict genres for the test set
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


################
import joblib

# Save the trained model
joblib.dump(model, "music_genre_classifier.pkl")
print("✅ Model saved successfully!")






