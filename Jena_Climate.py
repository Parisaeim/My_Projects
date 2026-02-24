import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('jena_climate_2009_2016.csv')
features_to_use = ['p (mbar)', 'T (degC)']
df_subset = df[['Date Time'] + features_to_use].copy()
df_subset['Date Time'] = pd.to_datetime(df_subset['Date Time'], format='%d.%m.%Y %H:%M:%S')
print(df_subset.head())
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_subset[features_to_use])
def create_sequences(data, window_size=24):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i : (i + window_size)])
    return np.array(X)

WINDOW_SIZE = 24
X_sequenced = create_sequences(scaled_features, WINDOW_SIZE)
split = int(len(X_sequenced) * 0.8)
train_x = X_sequenced[:split]
test_x = X_sequenced[split:]

print(f"Shape for LSTM (Samples, Steps, Features): {train_x.shape}")

#Building Model
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    # Encoder
    layers.Input(shape=(WINDOW_SIZE, len(features_to_use))),
    layers.LSTM(32, activation='relu', return_sequences=False),

    # Bridge
    layers.RepeatVector(WINDOW_SIZE),

    # Decoder
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.TimeDistributed(layers.Dense(len(features_to_use)))
])

model.compile(optimizer='adam', loss='mae')
model.summary()

# Training
history = model.fit(
    train_x, train_x,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)
#Calculating Reconstruction Error
train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

# Calculate MAE loss per sequence
# We take the mean across axis 1 (timesteps) and axis 2 (features)
train_mae_loss = np.mean(np.abs(train_pred - train_x), axis=(1, 2))
test_mae_loss = np.mean(np.abs(test_pred - test_x), axis=(1, 2))

print(f"Average Train MAE: {np.mean(train_mae_loss):.4f}")
print(f"Average Test MAE: {np.mean(test_mae_loss):.4f}")

#Anomaly Threshold
threshold = np.percentile(train_mae_loss, 99)

print(f"Reconstruction error threshold: {threshold:.4f}")

# Identify anomalies in the test set
is_anomaly = test_mae_loss > threshold
print(f"Number of anomalies detected in test set: {np.sum(is_anomaly)}")
import matplotlib.pyplot as plt
# Prepare a dataframe for the test results
# We need to align the test_mae_loss with the original timestamps
# Since we used a window of 24, the first test sequence corresponds to the 24th test index
test_dates = df_subset['Date Time'].iloc[split + WINDOW_SIZE:].reset_index(drop=True)
test_values = df_subset['T (degC)'].iloc[split + WINDOW_SIZE:].reset_index(drop=True)

# Create a results dataframe
results_df = pd.DataFrame({
    'Date Time': test_dates,
    'Loss': test_mae_loss,
    'Threshold': threshold,
    'Anomaly': is_anomaly,
    'Temp': test_values
})

# Plotting
plt.figure(figsize=(16, 8))

# Plot actual temperature
plt.plot(results_df['Date Time'], results_df['Temp'], label='Temperature', alpha=0.6, color='blue')

# Overlay anomalies
anomalies_only = results_df[results_df['Anomaly'] == True]
plt.scatter(anomalies_only['Date Time'], anomalies_only['Temp'],
            color='red', label='Detected Anomaly', s=10)

plt.title("Weather Anomaly Detection: Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (degC)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()