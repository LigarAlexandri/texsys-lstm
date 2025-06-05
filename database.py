import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv() # Load environment variables from .env file

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=os.getenv('DB_PORT', 3306) # Default port if not specified
        )
        # print("Database connection successful") # For debugging
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def fetch_query_as_df(query, params=None):
    """Fetches data from the database using a query and returns a Pandas DataFrame."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame() # Return empty DataFrame on connection error
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except mysql.connector.Error as err:
        print(f"Error executing query: {err}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during query execution: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Data Fetching Functions ---

def get_historical_sales(produk_jadi_id=None, start_date=None, end_date=None):
    """Fetches aggregated daily sales for a specific produk_jadi or all."""
    query = """
        SELECT
            DATE(tanggal_penjualan) AS sale_date,
            produk_jadi_id,
            SUM(jumlah_terjual) AS total_sold_on_day
        FROM penjualan
    """
    conditions = []
    params = []
    if produk_jadi_id:
        conditions.append("produk_jadi_id = %s")
        params.append(produk_jadi_id)
    if start_date:
        conditions.append("DATE(tanggal_penjualan) >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(tanggal_penjualan) <= %s")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += """
        GROUP BY DATE(tanggal_penjualan), produk_jadi_id
        ORDER BY produk_jadi_id, sale_date;
    """
    return fetch_query_as_df(query, tuple(params) if params else None)


def get_current_stock(item_id, item_type):
    """
    Gets the current stock level for a given item_id and item_type ('produk_jadi' or 'bahan_baku')
    by summing transactions in log_transaksi.
    """
    query = """
        SELECT SUM(jumlah) AS current_stock
        FROM log_transaksi
        WHERE item_id = %s AND tipe_item = %s;
    """
    df = fetch_query_as_df(query, (item_id, item_type))
    if not df.empty and pd.notna(df['current_stock'].iloc[0]):
        return float(df['current_stock'].iloc[0])
    return 0.0 # Default to 0 if no stock or error

def get_all_produk_jadi_ids():
    """Fetches all unique produk_jadi_id from the produk_jadi table."""
    query = "SELECT id FROM produk_jadi ORDER BY id;"
    df = fetch_query_as_df(query)
    if not df.empty:
        return df['id'].tolist()
    return []

def get_recipes():
    """Fetches all product recipes."""
    query = "SELECT produk_jadi_id, bahan_baku_id, jumlah_dibutuhkan FROM resep_produk;"
    return fetch_query_as_df(query)

