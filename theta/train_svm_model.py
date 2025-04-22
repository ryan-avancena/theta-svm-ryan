import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load your CSV file
df = pd.read_csv("processed_traffic_data.csv")

# Step 2: Drop rows with missing or zero traffic data to avoid division errors
df = df[(df["BACK_AADT"] > 0) & (df["AHEAD_AADT"] > 0) & (df["PEAK_HOUR_VOLUME"] > 0)]

# Step 3: Compute average AADT and the traffic multiplier
average_aadt = (df["BACK_AADT"] + df["AHEAD_AADT"]) / 2
df["traffic_multiplier"] = 1 + (df["PEAK_HOUR_VOLUME"] / average_aadt)

# Step 4: Select relevant features
features = ["PM_HOUR", "BACK_PEAK_HOUR", "AHEAD_PEAK_HOUR", "BACK_AADT", "AHEAD_AADT"]
target = "traffic_multiplier"

X = df[features]
y = df[target]

# Step 5: Split the data (optional but helpful for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the SVM model with scaling
model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
model.fit(X_train, y_train)

# Step 7: Save the model
joblib.dump(model, "traffic_svm_model.pkl")

print("✅ SVM model trained and saved as traffic_svm_model.pkl")
