import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.title("LSTM Model for Oil Production Forecasting")

uploaded_file = st.file_uploader("Upload CSV file with oil production data", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col='DATE', parse_dates=True)

    st.write("### Data Preview", data.head())

    # Plotting original data
    st.subheader("Original Oil Production Data")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['OIL'], marker='o', linestyle='None', markersize=4, color='red')
    ax.set_title('Monthly Oil Production')
    ax.set_xlabel('Date')
    ax.set_ylabel('Oil Production')
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    st.pyplot(fig)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['OIL'].values.reshape(-1, 1))

    # Create sequences
    def create_sequences(data, sequence_length):
        sequences = []
        labels = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            labels.append(data[i + sequence_length])
        return np.array(sequences), np.array(labels)

    sequence_length = 3
    X, y = create_sequences(scaled_data, sequence_length)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/Test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write("Training model, please wait...")
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

    # Predict and inverse scale
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Plot predictions
    st.subheader("Predicted vs Actual Oil Production")
    fig2, ax2 = plt.subplots()
    ax2.plot(actual, label='Actual')
    ax2.plot(predictions, label='Predicted')
    ax2.legend()
    st.pyplot(fig2)
