import pandas as pd
import numpy as np
from sklearn.svm import SVR
import joblib

# Load the data
peak_hours_path = "2022-peak-hours-ca - 2022 Peak Hour Report.csv"
traffic_volumes_path = "2022 AADT DATA - 2022 AADT DATA.csv"

peak_hours_df = pd.read_csv(peak_hours_path)
traffic_volumes_df = pd.read_csv(traffic_volumes_path)

# Clean and rename columns if necessary
peak_hours_df.rename(columns={"RTE": "ROUTE", "CO": "COUNTY"}, inplace=True)

# Merge the dataframes on ROUTE, COUNTY, PM
merged_df = pd.merge(
    traffic_volumes_df,
    peak_hours_df,
    on=["ROUTE", "COUNTY", "PM"],
    how="inner"
)

print(f"Merged {len(merged_df)} rows from Caltrans data.")

# Generate simulation examples
examples = []
for _, row in merged_df.iterrows():
    back_aadt = row.get("BACK_AADT", np.nan)
    ahead_aadt = row.get("AHEAD_AADT", np.nan)
    back_peak_hour = row.get("BACK_PEAK_HOUR", np.nan)
    ahead_peak_hour = row.get("AHEAD_PEAK_HOUR", np.nan)
    
    # Skip rows with missing critical data
    if np.isnan(back_aadt) or np.isnan(ahead_aadt) or np.isnan(back_peak_hour) or np.isnan(ahead_peak_hour):
        continue

    for hour in range(24):
        examples.append({
            "PM_HOUR": hour,
            "BACK_PEAK_HOUR": int(back_peak_hour) if not np.isnan(back_peak_hour) else 17,
            "AHEAD_PEAK_HOUR": int(ahead_peak_hour) if not np.isnan(ahead_peak_hour) else 17,
            "BACK_AADT": back_aadt,
            "AHEAD_AADT": ahead_aadt
        })

print(f"Generated {len(examples)} simulation examples.")

# Function to calculate realistic multiplier
def calculate_multiplier(pm_hour, back_peak_hour, ahead_peak_hour, back_aadt, ahead_aadt):
    base_multiplier = 1.0

    # Peak Hour Boost
    if abs(pm_hour - back_peak_hour) <= 1 or abs(pm_hour - ahead_peak_hour) <= 1:
        base_multiplier += 0.6

    # Nighttime reduction
    if (pm_hour >= 23 or pm_hour <= 5):
        base_multiplier -= 0.3

    # Midday moderate boost
    if 11 <= pm_hour <= 13:
        base_multiplier += 0.2

    # AADT scaling
    avg_aadt = (back_aadt + ahead_aadt) / 2
    base_multiplier += avg_aadt / 400000  # Scaling factor

    # Add small noise
    return max(base_multiplier + np.random.normal(0, 0.05), 1.0)

# Build feature and label arrays
X = []
y = []

for example in examples:
    features = [
        example["PM_HOUR"],
        example["BACK_PEAK_HOUR"],
        example["AHEAD_PEAK_HOUR"],
        example["BACK_AADT"],
        example["AHEAD_AADT"]
    ]
    multiplier = calculate_multiplier(*features)
    X.append(features)
    y.append(multiplier)

X = np.array(X)
y = np.array(y)

# Train SVM model
model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
model.fit(X, y)

# Save the model
joblib.dump(model, "traffic_svm_model.pkl")

print("✅ Model trained and saved as traffic_svm_model.pkl")
