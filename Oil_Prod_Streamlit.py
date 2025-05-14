import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.dates as mdates

# Streamlit App Title
st.title('Oil Production Forecasting using LSTM')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col='DATE', parse_dates=True)

    st.subheader("Raw Data")
    st.write(data.head())

    # Plot original data
    st.subheader("Monthly Oil Production")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data['OIL'], marker='o', linestyle='None', markersize=4, color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Oil Production')
    ax1.tick_params(axis='x', labelrotation=90)
    st.pyplot(fig1)

    # Preprocessing
    sequence_length = st.number_input("Lag Time (Sequence Length)", min_value=1, max_value=100, value=3)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['OIL'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build and train model
    st.subheader("Training LSTM Model")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    # Prediction
    predicted = model.predict(X)
    predicted_rescaled = scaler.inverse_transform(predicted)
    y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

    # R² Score
    r_squared = r2_score(y_rescaled, predicted_rescaled)
    st.write(f"R-squared Score: {r_squared:.2f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Production")
    dates = data.index[-len(y_rescaled):]
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(dates, y_rescaled, marker='o', linestyle='None', markersize=5, color='red', label='Actual')
    ax2.plot(dates, predicted_rescaled, label='Predicted', color='blue')
    ax2.legend()
    ax2.tick_params(axis='x', labelrotation=90)
    st.pyplot(fig2)

    # Forecast future
    def predict_future_values(model, initial_data, num_predictions):
        predictions = []
        current_sequence = initial_data.copy()
        for _ in range(num_predictions):
            current_sequence_reshaped = current_sequence.reshape((1, sequence_length, 1))
            next_value = model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_value[0, 0])
            current_sequence = np.append(current_sequence[1:], next_value)
        return np.array(predictions)

    num_future_predictions = st.slider("Months to Predict Into Future", 1, 60, 24)
    last_sequence = scaled_data[-sequence_length:]
    future_predictions_scaled = predict_future_values(model, last_sequence, num_future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1], periods=num_future_predictions + 1, freq='M')[1:]

    # Plot future predictions
    st.subheader("Forecast for Next Years")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(dates, y_rescaled, marker='o', linestyle='None', markersize=5, color='red', label='Actual')
    ax3.plot(dates, predicted_rescaled, label='Predicted', color='blue')
    ax3.plot(future_dates, future_predictions_rescaled, label='Future Prediction', color='green', marker='o')
    ax3.set_title('Actual, Predicted & Future Oil Production')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Oil Production')
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax3.tick_params(axis='x', labelrotation=45)
    ax3.legend()
    st.pyplot(fig3)

else:
    st.info("Please upload a CSV file with 'DATE' and 'OIL' columns.")
