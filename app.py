from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

import database
import model_utils

load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, restrict in production

# Ensure the models directory exists if we were to save models here (usually done in training)
# if not os.path.exists(model_utils.MODELS_DIR):
# os.makedirs(model_utils.MODELS_DIR)


@app.route('/')
def home():
    return "Textile Forecasting API is running!"

@app.route('/forecast/produk_jadi/<int:produk_id>', methods=['GET'])
def forecast_single_produk_jadi(produk_id):
    """
    Forecasts sales for a single produk_jadi.
    Query params:
        - forecast_days (int, default 7): Number of days to forecast.
        - history_days (int, default 90): Number of past days of sales to use for prediction.
    """
    try:
        forecast_days = int(request.args.get('forecast_days', 7))
        history_days = int(request.args.get('history_days', 90)) # How much history to fetch for context
    except ValueError:
        return jsonify({"error": "Invalid query parameter format for forecast_days or history_days"}), 400

    end_date_history = datetime.now().date()
    start_date_history = end_date_history - timedelta(days=history_days)

    historical_sales_df = database.get_historical_sales(
        produk_jadi_id=produk_id,
        start_date=start_date_history,
        end_date=end_date_history
    )

    if historical_sales_df.empty or len(historical_sales_df) < model_utils.SEQUENCE_LENGTH:
        return jsonify({
            "produk_jadi_id": produk_id,
            "warning": "Not enough historical data for robust prediction.",
            "forecasted_sales": [0.0] * forecast_days, # Fallback
            "message": "Consider providing more sales history or reducing history_days if this is initial data."
        }), 200 # 200 with warning, or 404 if product itself doesn't exist

    # Filter for the specific product, as get_historical_sales might return all if id is None
    product_specific_sales_df = historical_sales_df[historical_sales_df['produk_jadi_id'] == produk_id]

    if product_specific_sales_df.empty or len(product_specific_sales_df) < model_utils.SEQUENCE_LENGTH:
         return jsonify({
            "produk_jadi_id": produk_id,
            "warning": "Not enough historical data for this specific product.",
            "forecasted_sales": [0.0] * forecast_days,
        }), 200


    predictions = model_utils.predict_sales_for_product(
        produk_id,
        product_specific_sales_df, # Pass only the relevant product's sales
        forecast_horizon_days=forecast_days
    )

    # Generate future dates for the forecast
    last_historical_date_str = product_specific_sales_df['sale_date'].max()
    if isinstance(last_historical_date_str, str): # if sale_date is already string
        last_historical_date = datetime.strptime(last_historical_date_str, '%Y-%m-%d').date()
    elif isinstance(last_historical_date_str, pd.Timestamp): # if sale_date is Timestamp
        last_historical_date = last_historical_date_str.date()
    else: # assuming it's already a date object
        last_historical_date = last_historical_date_str
        
    forecast_dates = [(last_historical_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]


    return jsonify({
        "produk_jadi_id": produk_id,
        "forecast_dates": forecast_dates,
        "forecasted_sales": predictions
    })

@app.route('/forecast/full_analysis', methods=['GET'])
def full_analysis_forecast():
    """
    Provides a full forecast:
    1. Sales forecast for all produk_jadi.
    2. Calculated produk_jadi to make.
    3. Calculated bahan_baku needed.
    4. Calculated bahan_baku to purchase.
    Query params:
        - forecast_days (int, default 7): Number of days to forecast.
        - history_days (int, default 90): Past sales history to use.
        - safety_stock_pj_days (int, default 3): Safety stock for produk jadi in days of avg future sales.
        - safety_stock_bb_days (int, default 7): Safety stock for bahan baku in days of avg future usage.
    """
    try:
        forecast_days = int(request.args.get('forecast_days', 7))
        history_days = int(request.args.get('history_days', 90))
        safety_stock_pj_days = int(request.args.get('safety_stock_pj_days', 3))
        safety_stock_bb_days = int(request.args.get('safety_stock_bb_days', 7))
    except ValueError:
        return jsonify({"error": "Invalid query parameter format."}), 400

    all_produk_jadi_ids = database.get_all_produk_jadi_ids()
    if not all_produk_jadi_ids:
        return jsonify({"error": "No finished products (produk_jadi) found in the database."}), 404

    end_date_history = datetime.now().date()
    start_date_history = end_date_history - timedelta(days=history_days)
    
    # Fetch all historical sales once
    all_historical_sales_df = database.get_historical_sales(
        start_date=start_date_history,
        end_date=end_date_history
    )

    full_forecast_results = {
        "produk_jadi_forecasts": [],
        "produk_jadi_to_make": [],
        "bahan_baku_total_needed": {},
        "bahan_baku_to_purchase": []
    }
    
    recipes_df = database.get_recipes()
    if recipes_df.empty:
        return jsonify({"error": "No product recipes (resep_produk) found. Cannot calculate material needs."}), 404

    # 1. Sales forecast for all produk_jadi
    for pj_id in all_produk_jadi_ids:
        product_specific_sales_df = all_historical_sales_df[all_historical_sales_df['produk_jadi_id'] == pj_id]
        
        if product_specific_sales_df.empty or len(product_specific_sales_df) < model_utils.SEQUENCE_LENGTH:
            predictions = [0.0] * forecast_days # Fallback
            forecast_dates = [(end_date_history + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
            warning_msg = "Not enough historical data for robust prediction."
        else:
            predictions = model_utils.predict_sales_for_product(
                pj_id,
                product_specific_sales_df,
                forecast_horizon_days=forecast_days
            )
            last_historical_date_str = product_specific_sales_df['sale_date'].max()
            if isinstance(last_historical_date_str, str):
                last_historical_date = datetime.strptime(last_historical_date_str, '%Y-%m-%d').date()
            elif isinstance(last_historical_date_str, pd.Timestamp):
                last_historical_date = last_historical_date_str.date()
            else:
                last_historical_date = last_historical_date_str

            forecast_dates = [(last_historical_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
            warning_msg = None

        total_forecasted_sales = sum(predictions)
        avg_daily_forecasted_sales = total_forecasted_sales / forecast_days if forecast_days > 0 else 0

        full_forecast_results["produk_jadi_forecasts"].append({
            "produk_jadi_id": pj_id,
            "forecast_dates": forecast_dates,
            "forecasted_sales_per_day": predictions,
            "total_forecasted_sales_period": total_forecasted_sales,
            "warning": warning_msg
        })

        # 2. Calculate produk_jadi to make
        current_stock_pj = database.get_current_stock(pj_id, 'produk_jadi')
        safety_stock_pj = avg_daily_forecasted_sales * safety_stock_pj_days
        qty_to_make = max(0, round(total_forecasted_sales - current_stock_pj + safety_stock_pj))
        
        full_forecast_results["produk_jadi_to_make"].append({
            "produk_jadi_id": pj_id,
            "current_stock": current_stock_pj,
            "total_forecasted_sales": total_forecasted_sales,
            "calculated_safety_stock": round(safety_stock_pj),
            "quantity_to_make": qty_to_make
        })

        # Aggregate bahan_baku needed for this pj_id to be made
        if qty_to_make > 0:
            product_recipe = recipes_df[recipes_df['produk_jadi_id'] == pj_id]
            for _, recipe_item in product_recipe.iterrows():
                bb_id = int(recipe_item['bahan_baku_id'])
                needed_for_this_bb = recipe_item['jumlah_dibutuhkan'] * qty_to_make
                full_forecast_results["bahan_baku_total_needed"][bb_id] = \
                    full_forecast_results["bahan_baku_total_needed"].get(bb_id, 0.0) + needed_for_this_bb
    
    # 3. Calculate bahan_baku to purchase
    # To calculate safety stock for bahan baku, we need average daily usage.
    # This is a bit more complex as usage depends on multi-product recipes.
    # For simplicity, we'll base it on the total needed over the forecast period.
    
    for bb_id, total_needed_for_period in full_forecast_results["bahan_baku_total_needed"].items():
        current_stock_bb = database.get_current_stock(bb_id, 'bahan_baku')
        avg_daily_usage_bb = total_needed_for_period / forecast_days if forecast_days > 0 else 0
        safety_stock_bb = avg_daily_usage_bb * safety_stock_bb_days
        qty_to_purchase = max(0, round(total_needed_for_period - current_stock_bb + safety_stock_bb))

        full_forecast_results["bahan_baku_to_purchase"].append({
            "bahan_baku_id": bb_id,
            "current_stock": current_stock_bb,
            "total_calculated_need_for_period": round(total_needed_for_period, 2),
            "calculated_safety_stock": round(safety_stock_bb, 2),
            "quantity_to_purchase": qty_to_purchase
        })

    return jsonify(full_forecast_results)


if __name__ == '__main__':
    app.run(debug=True, port=5001) # Run on a different port than Laravel's default