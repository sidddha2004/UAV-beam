import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Optional: disables GPU warnings if using CPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Load beamforming dataset
df = pd.read_csv('beamforming_prediction_dataset.csv')

# Select features for beamforming prediction
beamforming_features = ['distance', 'rx_x', 'rx_y', 'velocity_ms', 'hour', 'angle_to_rx', 'path_loss_db']

# Feature scaling (excluding target)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df[beamforming_features])
scaled_df = pd.DataFrame(features_scaled, columns=beamforming_features)
scaled_df['optimal_beam_index'] = df['optimal_beam_index']

# Create sequences for LSTM
sequence_length = 7
X_seq, y_seq = [], []

for i in range(len(scaled_df) - sequence_length):
    seq = scaled_df.iloc[i:i+sequence_length].drop(columns=['optimal_beam_index']).values
    target = int(scaled_df.iloc[i + sequence_length]['optimal_beam_index'])
    X_seq.append(seq)
    y_seq.append(target)

X_seq = np.array(X_seq)
y_seq = to_categorical(y_seq, num_classes=64)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Input(shape=(sequence_length, X_seq.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', TopKCategoricalAccuracy(k=5)]
)
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
