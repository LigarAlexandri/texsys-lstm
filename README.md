# Texsys-lstm (CURRENTLY WORK IN PROGRESS)
This is the LSTM model for forecasting sales, how much bahan baku to buy and barang jadi needed to make for a TRIJAYA textile convection shop in Banyuwangi Indonesia
Made as a part of Final Submission for SCM (Supply Chain Management) class

# Where is the System? (CURRENTLY WORK IN PROGRESS)
Here it is 
https://github.com/Muzaky/texsys-scm 

# LSTM Forecasting System Documentation - Trijaya Textile Store

## 1. Overview

This system is designed to forecast sales demand for finished goods (`produk_jadi`) and then calculate the required raw materials (`bahan_baku`) needed for production and purchasing. It utilizes LSTM (Long Short-Term Memory) neural networks for sales forecasting.

The system consists of three main parts:
1.  **Database**: Stores product information, recipes, historical sales, stock transactions, and initial stock levels (e.g., MySQL).
2.  **Python Backend (Flask API)**:
    * Handles data fetching from the database.
    * Contains scripts for training LSTM models for each finished product.
    * Serves a Flask API that loads trained models to provide sales forecasts and calculate derived material requirements.
3.  **Frontend Web Application (e.g., Laravel)**:
    * Provides a user interface to trigger forecasts.
    * Calls the Flask API to get prediction results.
    * Displays the forecasts and material requirements to the user.

## 2. Python Flask API Project Structure

Your Python backend (referred to as `textile_api` in examples) should have the following structure:

textile_api/
├── venv/                     # Python virtual environment (ignored by Git)
├── trained_models/           # Directory to store trained .keras models and .joblib scalers (ignored by Git if large)
├── .env                      # Environment variables (DB credentials, paths - IGNORED BY GIT)
├── .env.example              # Example environment variables
├── .gitignore                # Specifies intentionally untracked files by Git
├── app.py                    # Main Flask application, defines API endpoints
├── database.py               # Functions for database connections and queries
├── model_utils.py            # LSTM model definition, loading, preprocessing, prediction logic
├── train.py                  # Script to train LSTM models for each product
├── requirements.txt          # Python package dependencies
└── generate_large_data.py    # (Optional) Script to generate sample SQL data for testing

## 3. Setup and Installation

### 3.1. Python Environment
1.  **Install Python**: Ensure you have Python 3.8+ installed.
2.  **Create Project Directory**: Create a folder for your Flask API (e.g., `textile_api`).
3.  **Navigate to Directory**: Open your terminal/command prompt and `cd` into this directory.
4.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    ```
5.  **Activate Virtual Environment**:
    * Windows: `venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
    (Your terminal prompt should now show `(venv)`)
6.  **Install Dependencies**: Make sure `requirements.txt` is in your project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` should include:
    ```
    flask
    flask_cors
    tensorflow
    pandas
    numpy
    scikit-learn
    mysql-connector-python
    python-dotenv
    joblib
    ```

### 3.2. Database Setup
1.  **Install MySQL Server**: If not already installed.
2.  **Create Database**: Create a database for the project (e.g., `scm`).
3.  **Create Tables**: Use the `CREATE TABLE IF NOT EXISTS` statements generated by `generate_large_data.py` (or your Laravel migrations if they define the same structure) to create the necessary tables:
    * `bahan_baku`
    * `produk_jadi`
    * `resep_produk`
    * `penjualan`
    * `log_transaksi`
4.  **Populate Initial Data**:
    * For testing, you can use the `sample_data_large.sql` file (generated by `generate_large_data.py`) to populate your database with a substantial amount of sample sales and transaction history.
    * Import this `.sql` file into your database (e.g., using MySQL Workbench, phpMyAdmin, or the `mysql` command line: `mysql -u your_user -p scm < sample_data_large.sql`).
    * Ensure your actual `bahan_baku`, `produk_jadi`, and `resep_produk` tables contain accurate definitions for your store's items.

### 3.3. Environment Variables (`.env` file for Flask API)
1.  In your `textile_api` directory, create a file named `.env`.
2.  Copy the contents from `.env.example` (if you have one) or add the following, **replacing with your actual details**:
    ```env
    DB_HOST=localhost
    DB_USER=your_db_username
    DB_PASSWORD=your_db_password
    DB_NAME=scm
    DB_PORT=3306

    # Directory where trained models and scalers are stored
    MODELS_DIR=./trained_models/
    ```
    **Important**: Add `.env` to your `.gitignore` file to prevent committing sensitive credentials.

## 4. Data Generation (Optional - For Initial Testing)

If you need sample data to test the system or train initial models:
1.  Ensure `generate_large_data.py` is in your `textile_api` directory.
2.  Activate your virtual environment.
3.  Run the script:
    ```bash
    python generate_large_data.py
    ```
4.  This will create `sample_data_large.sql`. Import this SQL file into your database as described in section 3.2.
    * **Note**: This script also generates the `CREATE TABLE` statements. If your tables already exist and have data, you might want to drop/truncate them before importing, or only import the `INSERT` statements.

## 5. Training the LSTM Models

This step uses historical sales data from the `penjualan` table to train an LSTM model for each `produk_jadi`. These models are then saved for later use by the API.

1.  **Prerequisites**:
    * Ensure your database is populated with sufficient historical sales data in the `penjualan` table for each product you want to forecast. The more data (e.g., 1-2 years of daily sales data per product), the better the potential model quality.
    * The `database.py` script must be able to connect to your database (check `.env`).
    * The `MODELS_DIR` specified in `.env` must exist or be creatable by the script.
2.  **Activate Virtual Environment**: If not already active.
3.  **Run the Training Script**:
    ```bash
    python train.py
    ```
4.  **Process**:
    * `train.py` iterates through each `produk_jadi_id`.
    * It fetches historical daily sales for that product from the `penjualan` table.
    * It preprocesses the data (scaling, creating sequences).
    * It trains an LSTM model (defined in `model_utils.py`).
    * It saves the trained model as a `.keras` file (e.g., `trained_models/produk_jadi_1_model.keras`).
    * It saves the scaler used for that model as a `.joblib` file (e.g., `trained_models/produk_jadi_1_scaler.joblib`).
5.  **Output**:
    * Monitor the console for training progress and any errors (e.g., "Not enough data to train...").
    * Check your `MODELS_DIR` for the saved `.keras` model files and `.joblib` scaler files.
6.  **Frequency**: Training should be done initially and then re-run periodically (e.g., weekly, monthly) as new sales data becomes available to keep the models up-to-date.

## 6. Running the Flask Forecasting API

This API serves the predictions using the pre-trained models.

1.  **Prerequisites**:
    * LSTM models and scalers must have been trained and saved in the `MODELS_DIR`.
    * The `database.py` script must be able to connect to your database (for fetching current stock, recipes, and recent history for prediction context).
2.  **Activate Virtual Environment**: If not already active.
3.  **Run the Flask App**:
    ```bash
    python app.py
    ```
4.  **Output**:
    * The Flask development server will start, typically on `http://localhost:5001` (or the port specified at the end of `app.py`).
    * The console will indicate that the server is running and listening for requests.
    * Keep this terminal window open; closing it will stop the API.
5.  **API Endpoints**:
    * **`GET /forecast/produk_jadi/<produk_id>`**: Forecasts sales for a single product.
        * Query Parameters: `forecast_days` (int, default 7), `history_days` (int, default 90).
    * **`GET /forecast/full_analysis`**: Provides a comprehensive forecast including:
        * Sales forecast for all `produk_jadi`.
        * Calculated `produk_jadi` to make.
        * Calculated `bahan_baku` total needed for production.
        * Calculated `bahan_baku` to purchase.
        * Query Parameters: `forecast_days` (int, default 7), `history_days` (int, default 90), `safety_stock_pj_days` (int, default 3), `safety_stock_bb_days` (int, default 7).
6.  **Testing with Postman/cURL**:
    * Use tools like Postman or cURL to send GET requests to these endpoints to test if the API is working correctly and returning JSON responses.
    * Example: `http://localhost:5001/forecast/full_analysis?forecast_days=14&history_days=180`

## 7. Integration with Web Application (e.g., Laravel)

Your main web application (e.g., built with Laravel) will interact with the Flask API to get forecasts.

1.  **API URL Configuration**:
    * In your Laravel application's `.env` file, define the base URL for your Flask API:
        ```env
        FLASK_API_URL=http://localhost:5001
        ```
    * Ensure Laravel can access this URL (they might be running on the same machine or different machines on the same network, or the Flask API might be publicly exposed if deployed).
2.  **Frontend JavaScript**:
    * The Laravel Blade view (e.g., `forecast/index.blade.php`) will use JavaScript's `fetch` API to make GET requests to the Flask API's `/forecast/full_analysis` endpoint.
    * It will pass parameters like `forecast_days` as query string parameters.
    * The JavaScript will then parse the JSON response from the Flask API and display the data (e.g., in tables, charts).
    * Refer to the `textile_laravel_forecast_blade` immersive for an example of how the JavaScript in the Blade file is structured.
