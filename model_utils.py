# File: model_utils.py
# ---------------------------
import numpy as np
# from sklearn.preprocessing import MinMaxScaler # Scaler will be loaded
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import os
from dotenv import load_dotenv
import joblib # For loading the scaler

load_dotenv()

MODELS_DIR = os.getenv('MODELS_DIR', './trained_models/')
SEQUENCE_LENGTH = 30 # Number of past time steps to use for prediction
N_FEATURES = 1 # Univariate model (only using sales quantity)

def create_lstm_model(sequence_length=SEQUENCE_LENGTH, n_features=N_FEATURES):
    """
    Defines a simple LSTM model architecture.
    This function is primarily used by train.py.
    """
    model = Sequential()
    # Add Input layer to specify input_shape for the first LSTM layer
    model.add(Input(shape=(sequence_length, n_features))) 
    model.add(LSTM(50, activation='relu')) # 50 LSTM units
    model.add(Dense(1)) # Output layer to predict 1 step ahead
    model.compile(optimizer='adam', loss='mse') # 'mse' (mean squared error) is a common loss for regression
    return model

def preprocess_data_for_prediction(sales_data_df, scaler):
    """
    Prepares historical sales data for LSTM prediction using a pre-fitted scaler.
    - sales_data_df: Pandas DataFrame with a 'total_sold_on_day' column, already resampled to daily.
    - scaler: The scikit-learn MinMaxScaler object that was fitted on the training data.
    Returns the last sequence suitable for model input, or None if data is insufficient.
    """
    if sales_data_df.empty or len(sales_data_df) < SEQUENCE_LENGTH:
        print(f"Preprocessing error: Not enough historical data points. Need at least {SEQUENCE_LENGTH}, got {len(sales_data_df)}.")
        return None
    if scaler is None:
        print("Preprocessing error: Scaler object not provided.")
        return None

    # Extract the 'total_sold_on_day' column and reshape for the scaler
    sales_values = sales_data_df['total_sold_on_day'].values.reshape(-1, 1)
    
    try:
        # Transform the data using the provided (already fitted) scaler
        scaled_data = scaler.transform(sales_values)
    except Exception as e:
        print(f"Error during scaling/transform: {e}")
        return None

    # We only need the last `SEQUENCE_LENGTH` data points to form the input sequence
    if len(scaled_data) >= SEQUENCE_LENGTH:
        last_sequence = scaled_data[-SEQUENCE_LENGTH:]
        # Reshape the sequence to be [1, SEQUENCE_LENGTH, N_FEATURES] for the LSTM model
        last_sequence_reshaped = last_sequence.reshape((1, SEQUENCE_LENGTH, N_FEATURES))
        return last_sequence_reshaped
    else:
        print(f"Not enough data to form a full sequence after scaling. Need {SEQUENCE_LENGTH}, got {len(scaled_data)}.")
        return None

def load_lstm_model_and_scaler(produk_jadi_id):
    """
    Loads a pre-trained LSTM model (from .keras file) and its corresponding scaler 
    (from .joblib file) for a specific produk_jadi_id.
    Returns (model, scaler) or (None, None) if loading fails.
    """
    model_filename = f"produk_jadi_{produk_jadi_id}_model.keras"
    scaler_filename = f"produk_jadi_{produk_jadi_id}_scaler.joblib"
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    scaler_path = os.path.join(MODELS_DIR, scaler_filename)
    
    model = None
    scaler = None

    if os.path.exists(model_path):
        try:
            model = load_model(model_path) 
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None, None # Return None for both if model loading fails
    else:
        print(f"Model file not found: {model_path}")
        return None, None 

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler {scaler_path}: {e}")
            # Even if model loads, if scaler fails, prediction won't be correct/possible
            return model, None # Model might be loaded, but scaler is missing or failed to load
    else:
        print(f"Scaler file not found: {scaler_path}")
        return model, None 

    return model, scaler


def predict_sales_for_product(produk_jadi_id, historical_sales_df, forecast_horizon_days=7):
    """
    Predicts sales for a single product for a given number of future days (forecast_horizon_days).
    - produk_jadi_id: The ID of the product to forecast.
    - historical_sales_df: Pandas DataFrame of historical sales for THIS SPECIFIC PRODUCT. 
                           It must contain a 'total_sold_on_day' column and be already
                           resampled to daily frequency with missing values filled (e.g., with 0).
    - forecast_horizon_days: Number of future days to predict.
    Returns a list of predicted sales quantities (integers).
    """
    model, scaler = load_lstm_model_and_scaler(produk_jadi_id)

    if model is None or scaler is None:
        warning_msg = "Cannot make predictions for this product: "
        if model is None: warning_msg += "model not found or failed to load. "
        if scaler is None: warning_msg += "scaler not found or failed to load."
        print(f"For produk_jadi_id {produk_jadi_id}: {warning_msg}")
        return [0] * forecast_horizon_days # Return list of zeros (as integers)

    # Prepare the last known sequence from historical_sales_df using the loaded scaler
    last_known_sequence = preprocess_data_for_prediction(historical_sales_df, scaler=scaler)

    if last_known_sequence is None:
        # This implies preprocess_data_for_prediction found issues (e.g. not enough data points)
        print(f"Could not prepare input sequence from historical data for produk_jadi_id {produk_jadi_id}.")
        return [0] * forecast_horizon_days

    predictions_scaled = []
    current_batch_for_prediction = last_known_sequence.copy() # Start with the last known sequence

    # Iteratively predict for the forecast_horizon_days
    for _ in range(forecast_horizon_days):
        # Predict the next step (model.predict expects a batch)
        next_step_pred_scaled = model.predict(current_batch_for_prediction, verbose=0)[0] # verbose=0 to suppress Keras logs
        predictions_scaled.append(next_step_pred_scaled[0]) # Append the scalar prediction

        # Update the batch for the next prediction:
        # Reshape the prediction to be [1, 1, N_FEATURES]
        new_step_reshaped = next_step_pred_scaled.reshape((1, 1, N_FEATURES))
        # Append the new prediction and remove the oldest step from the batch
        current_batch_for_prediction = np.append(current_batch_for_prediction[:, 1:, :], new_step_reshaped, axis=1)

    # Inverse transform the scaled predictions to their original scale
    if predictions_scaled: # Check if any predictions were made
        predictions_original_scale = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
        # Ensure predictions are non-negative integers (as sales are counts)
        final_predictions = [max(0, int(round(p[0]))) for p in predictions_original_scale]
    else:
        final_predictions = [0] * forecast_horizon_days
    
    return final_predictions
