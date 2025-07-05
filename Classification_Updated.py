import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Load dataset
X = pd.read_csv('/content/synthetic_beamforming_dataset_doubled.csv')

# Preprocess: scale input features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X.drop(columns=['beam_index_target']))
scaled_df = pd.DataFrame(scaled_features, columns=X.columns[:-1])
scaled_df['beam_index_target'] = X['beam_index_target']  # Add target back

# Create sequences for LSTM
sequence_length = 7
X_sequences = []
y_labels = []

for i in range(len(scaled_df) - sequence_length):
    seq = scaled_df.iloc[i:i+sequence_length].drop(columns=['beam_index_target']).values
    target = int(scaled_df.iloc[i + sequence_length]['beam_index_target'])

    X_sequences.append(seq)
    y_labels.append(target)

# Convert to arrays
X_sequences = np.array(X_sequences)
y_labels = np.array(y_labels)

# One-hot encode the target labels
y_categorical = to_categorical(y_labels, num_classes=64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(LSTM(128,return_sequences=True, input_shape=(sequence_length, X_sequences.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',TopKCategoricalAccuracy(k=5)])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
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
