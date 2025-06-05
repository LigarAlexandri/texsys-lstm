import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # Changed from keras to tensorflow.keras
from tensorflow.keras.layers import LSTM, Dense, Input # Added Input
import os
from dotenv import load_dotenv
import joblib # For saving the scaler

# Assuming database.py and model_utils.py are in the same directory or accessible
import database # To fetch historical data
import model_utils # For create_lstm_model and constants

load_dotenv()

MODELS_DIR = os.getenv('MODELS_DIR', './trained_models/')
SEQUENCE_LENGTH = model_utils.SEQUENCE_LENGTH # e.g., 30
N_FEATURES = model_utils.N_FEATURES         # e.g., 1 for univariate
EPOCHS = 50 # Example
BATCH_SIZE = 16 # Example

def create_sequences(data, sequence_length):
    """Creates sequences for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model_for_product(produk_jadi_id, sales_df):
    """Trains and saves an LSTM model for a single produk_jadi_id."""
    print(f"\n--- Training model for Produk Jadi ID: {produk_jadi_id} ---")

    if sales_df.empty or len(sales_df) < SEQUENCE_LENGTH + 10: # Need enough data for sequences and test
        print(f"Not enough data to train model for produk_jadi_id {produk_jadi_id}. Skipping.")
        return

    # 1. Prepare Data
    sales_values = sales_df['total_sold_on_day'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_values) # Fit scaler ON TRAINING DATA

    # 2. Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    if X.shape[0] == 0:
        print(f"Could not create sequences for produk_jadi_id {produk_jadi_id}. Skipping.")
        return

    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))

    # 3. Split data (simple split, consider time-series cross-validation for robust evaluation)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Not enough data after splitting for produk_jadi_id {produk_jadi_id}. Skipping.")
        return

    # 4. Create and Compile Model
    model = model_utils.create_lstm_model(SEQUENCE_LENGTH, N_FEATURES)
    # model.summary() # Optional: print model summary

    # 5. Train Model
    print(f"Starting training for produk_jadi_id {produk_jadi_id}...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1 
    )
    print("Training complete.")

    # 6. Save Model and Scaler
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Save model using the .keras format
    model_path = os.path.join(MODELS_DIR, f"produk_jadi_{produk_jadi_id}_model.keras") # CHANGED
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save the scaler
    scaler_path = os.path.join(MODELS_DIR, f"produk_jadi_{produk_jadi_id}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")


def main():
    print("Starting LSTM model training process...")
    
    all_produk_ids = database.get_all_produk_jadi_ids()
    if not all_produk_ids:
        print("No produk_jadi found to train models for.")
        return

    historical_sales_all_df = database.get_historical_sales()

    if historical_sales_all_df.empty:
        print("No historical sales data found in the database.")
        return

    for produk_id in all_produk_ids:
        product_sales_df = historical_sales_all_df[historical_sales_all_df['produk_jadi_id'] == produk_id].copy()
        
        if not product_sales_df.empty:
            product_sales_df['sale_date'] = pd.to_datetime(product_sales_df['sale_date'])
            product_sales_df = product_sales_df.sort_values('sale_date')
            
            # IMPORTANT: Resample to daily frequency and fill missing dates (e.g., with 0)
            # This ensures a continuous time series if there are days with no sales.
            # And is crucial for the LSTM to understand "days with no sales".
            product_sales_df = product_sales_df.set_index('sale_date') \
                .resample('D')['total_sold_on_day'] \
                .sum().fillna(0).reset_index() 

            train_model_for_product(produk_id, product_sales_df)
        else:
            print(f"No sales data found for produk_jadi_id {produk_id} to start training.")

    print("\nAll training processes finished.")

if __name__ == '__main__':
    main()