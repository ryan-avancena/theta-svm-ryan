import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load processed dataset (replace with actual CSV path if needed)
df_final = pd.read_csv("processed_traffic_data.csv")

print("🛠️ Checking DataFrame Columns:", df_final.columns)
print("🛠️ First 5 Rows:\n", df_final.head())

# Select features & target variable
feature_columns = ['PM_HOUR', 'BACK_PEAK_HOUR', 'AHEAD_PEAK_HOUR', 'BACK_AADT', 'AHEAD_AADT']
target_column = 'PEAK_HOUR_VOLUME'

# Ensure selected columns are numeric
for col in feature_columns:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

print(df_final[feature_columns].describe())  # Check if columns have values

# Drop rows with missing values in the selected feature columns
df_final_clean = df_final.dropna(subset=feature_columns)

# Check if we have enough data
print(f"🛠️ Number of valid rows after cleaning: {df_final_clean.shape[0]}")

if df_final_clean.shape[0] == 0:
    raise ValueError("❌ No valid data available after cleaning! Check your dataset.")

# Extract features and target variable
X = df_final_clean[feature_columns].values
y = df_final_clean['PEAK_HOUR_VOLUME'].values

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input for CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("traffic_cnn_model.h5")
print("✅ Model training complete! Saved as 'traffic_cnn_model.h5'")
