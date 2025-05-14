import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit page setup
st.set_page_config(page_title="Oil Production LSTM Forecast", layout="centered")
st.title("🛢️ Oil Production Forecast using LSTM")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file, index_col='DATE', parse_dates=True)
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # Plotting
    st.subheader("Monthly Oil Production Plot")
    fig, ax = plt.subplots()
    ax.plot(data['OIL'], color='red', marker='o', linestyle='None', markersize=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Production")
    st.pyplot(fig)

    # Parameters
    st.subheader("LSTM Parameters")
    lag = st.slider("Lag Time (previous months)", 1, 12, 3)
    epochs = st.slider("Training Epochs", 10, 200, 50)
    train_ratio = st.slider("Training Data Ratio", 0.1, 0.9, 0.8)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['OIL'].values.reshape(-1, 1))

    # Create sequences
    def create_dataset(dataset, lag=1):
        X, y = [], []
        for i in range(len(dataset) - lag):
            X.append(dataset[i:i + lag, 0])
            y.append(dataset[i + lag, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, lag)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(lag, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    if st.button("Train Model"):
        with st.spinner("Training..."):
            model.fit(X_train, y_train, epochs=epochs, verbose=0)

        # Prediction
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot
        st.subheader("Prediction vs Actual")
        fig2, ax2 = plt.subplots()
        ax2.plot(actual, label="Actual")
        ax2.plot(predicted, label="Predicted")
        ax2.legend()
        st.pyplot(fig2)
