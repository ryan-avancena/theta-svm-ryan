import numpy as np
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.model_selection import train_test_split
import joblib

# ✅ 1. Create better, more realistic fake data

# Features:
# PM_HOUR (0-23), BACK_PEAK_HOUR (0-23), AHEAD_PEAK_HOUR (0-23), BACK_AADT (10k-150k), AHEAD_AADT (10k-150k)
np.random.seed(42)
X = np.column_stack([
    np.random.randint(0, 24, 1000),        # PM_HOUR
    np.random.randint(0, 24, 1000),        # BACK_PEAK_HOUR
    np.random.randint(0, 24, 1000),        # AHEAD_PEAK_HOUR
    np.random.randint(10000, 150000, 1000),# BACK_AADT
    np.random.randint(10000, 150000, 1000) # AHEAD_AADT
])

# ✅ 2. Create corresponding "realistic" multipliers
# Assume:
# - Higher AADT = higher multiplier
# - PM rush hours (4-7pm) = higher multiplier

def calculate_multiplier(pm_hour, back_peak_hour, ahead_peak_hour, back_aadt, ahead_aadt):
    base_multiplier = 1.0

    # AADT scaling (always apply, even if small)
    avg_aadt = (back_aadt + ahead_aadt) / 2
    base_multiplier += avg_aadt / 400000

    # Time-based effects
    if 23 <= pm_hour <= 24 or 0 <= pm_hour <= 5:
        # Nighttime: reduce traffic by 30%
        base_multiplier -= 0.3
    elif 11 <= pm_hour <= 13:
        # Midday: slight boost
        base_multiplier += 0.2
    else:
        # Otherwise, check for peak hour boost
        if abs(pm_hour - back_peak_hour) <= 1 or abs(pm_hour - ahead_peak_hour) <= 1:
            base_multiplier += 0.6

    # Add small Gaussian noise
    noisy_multiplier = base_multiplier + np.random.normal(0, 0.05)

    # Make sure multiplier is at least 1.0
    return max(noisy_multiplier, 1.0)

y = np.apply_along_axis(calculate_multiplier, 1, X)

# ✅ 3. Split into train/test (optional for better quality)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 4. Train the SVM Regressor
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

# ✅ 5. Save the model
joblib.dump(model, "traffic_svm_model.pkl")

print("✅ New SVM model trained and saved as traffic_svm_model.pkl")