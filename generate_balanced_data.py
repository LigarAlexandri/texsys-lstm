import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- Configuration ---
OUTPUT_SQL_FILE = "sample_data_large_balanced.sql" # New output file name

# Product and Raw Material Definitions with INCREASED INITIAL STOCK
BAHAN_BAKU_DEFS = [
    {'id': 1, 'nama': 'Kain Katun Polos', 'satuan': 'meter', 'initial_stok': 3000.0, 'harga': 35000.0}, # Increased from 300
    {'id': 2, 'nama': 'Kancing Plastik Putih', 'satuan': 'lusin', 'initial_stok': 2000.0, 'harga': 18000.0}, # Increased from 200
    {'id': 3, 'nama': 'Badge Sekolah Bordir', 'satuan': 'pcs', 'initial_stok': 5000.0, 'harga': 5000.0}   # Increased from 400
]
PRODUK_JADI_DEFS = [
    {'id': 1, 'kategori': 'Baju Olahraga', 'initial_stok': 500, 'harga': 240000.0},     # Increased from 70
    {'id': 2, 'kategori': 'Baju Putih Abu Abu', 'initial_stok': 450, 'harga': 270000.0}, # Increased from 60
    {'id': 3, 'kategori': 'Baju Batik', 'initial_stok': 300, 'harga': 290000.0},         # Increased from 50
    {'id': 4, 'kategori': 'Baju Pramuka', 'initial_stok': 400, 'harga': 200000.0}      # Increased from 65
]
RESEP_PRODUK_DEFS = [
    {'produk_jadi_id': 1, 'bahan_baku_id': 2, 'jumlah_dibutuhkan': 7.0},
    {'produk_jadi_id': 1, 'bahan_baku_id': 3, 'jumlah_dibutuhkan': 3.0},
    {'produk_jadi_id': 1, 'bahan_baku_id': 1, 'jumlah_dibutuhkan': 1.5},

    {'produk_jadi_id': 2, 'bahan_baku_id': 1, 'jumlah_dibutuhkan': 2.75},
    {'produk_jadi_id': 2, 'bahan_baku_id': 2, 'jumlah_dibutuhkan': 8.0},
    {'produk_jadi_id': 2, 'bahan_baku_id': 3, 'jumlah_dibutuhkan': 2.0},

    {'produk_jadi_id': 3, 'bahan_baku_id': 1, 'jumlah_dibutuhkan': 2.0},
    {'produk_jadi_id': 3, 'bahan_baku_id': 2, 'jumlah_dibutuhkan': 6.0},

    {'produk_jadi_id': 4, 'bahan_baku_id': 1, 'jumlah_dibutuhkan': 2.25},
    {'produk_jadi_id': 4, 'bahan_baku_id': 2, 'jumlah_dibutuhkan': 7.0},
    {'produk_jadi_id': 4, 'bahan_baku_id': 3, 'jumlah_dibutuhkan': 4.0}
]

# Simulation Parameters
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31) # 2 years of data
DAYS_IN_PERIOD = (END_DATE - START_DATE).days + 1

# --- Helper function to generate SQL INSERT statements ---
def data_to_sql_inserts(data_list, table_name, columns_list):
    sql_inserts = []
    sql_inserts.append(f"-- Data for table: {table_name}")
    for row_dict in data_list:
        values_list = []
        for col in columns_list:
            val = row_dict.get(col)
            if val is None:
                values_list.append("NULL")
            elif isinstance(val, str):
                escaped_val = str(val).replace("'", "''")
                values_list.append(f"'{escaped_val}'")
            elif isinstance(val, (datetime, pd.Timestamp)):
                values_list.append(f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'")
            elif isinstance(val, (np.integer, int)):
                 values_list.append(str(val))
            elif isinstance(val, (np.floating, float)):
                 values_list.append(f"{val:.2f}")
            else:
                values_list.append(str(val))
        columns_str = ', '.join(f"`{col}`" for col in columns_list)
        values_str = ', '.join(values_list)
        sql_inserts.append(f"INSERT INTO `{table_name}` ({columns_str}) VALUES ({values_str});")
    sql_inserts.append("\n")
    return "\n".join(sql_inserts)

# --- SQL CREATE TABLE IF NOT EXISTS Statements ---
def get_create_table_statements():
    statements = ["-- Schema Definitions --\n"]
    statements.append("""
CREATE TABLE IF NOT EXISTS `bahan_baku` (
  `id` INT NOT NULL PRIMARY KEY,
  `nama` VARCHAR(255) NOT NULL,
  `satuan` VARCHAR(50),
  `stok_level` DECIMAL(10,2) DEFAULT 0.00,
  `harga` DECIMAL(12,2) DEFAULT 0.00,
  `created_at` DATETIME,
  `updated_at` DATETIME
);
""")
    statements.append("""
CREATE TABLE IF NOT EXISTS `produk_jadi` (
  `id` INT NOT NULL PRIMARY KEY,
  `kategori` VARCHAR(100) NOT NULL,
  `stok_level` INT DEFAULT 0,
  `harga` DECIMAL(12,2) DEFAULT 0.00,
  `created_at` DATETIME,
  `updated_at` DATETIME
);
""")
    statements.append("""
CREATE TABLE IF NOT EXISTS `resep_produk` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `produk_jadi_id` INT NOT NULL,
  `bahan_baku_id` INT NOT NULL,
  `jumlah_dibutuhkan` DECIMAL(10,2) NOT NULL,
  `created_at` DATETIME,
  `updated_at` DATETIME,
  FOREIGN KEY (`produk_jadi_id`) REFERENCES `produk_jadi`(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`bahan_baku_id`) REFERENCES `bahan_baku`(`id`) ON DELETE CASCADE
);
""")
    statements.append("""
CREATE TABLE IF NOT EXISTS `penjualan` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `tanggal_penjualan` DATETIME NOT NULL,
  `produk_jadi_id` INT NOT NULL,
  `jumlah_terjual` INT NOT NULL,
  `total_harga` DECIMAL(15,2) NOT NULL,
  `created_at` DATETIME,
  `updated_at` DATETIME,
  FOREIGN KEY (`produk_jadi_id`) REFERENCES `produk_jadi`(`id`) ON DELETE RESTRICT
);
""")
    statements.append("""
CREATE TABLE IF NOT EXISTS `log_transaksi` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `tanggal` DATETIME NOT NULL,
  `tipe_item` VARCHAR(50) NOT NULL COMMENT 'e.g., bahan_baku, produk_jadi',
  `item_id` INT NOT NULL COMMENT 'References id in bahan_baku or produk_jadi',
  `tipe_transaksi` VARCHAR(100) NOT NULL COMMENT 'e.g., INITIAL_STOCK, PENJUALAN, PENGGUNAAN_PRODUKSI, PEMBELIAN_BAHAN_BAKU, PRODUKSI_SELESAI',
  `jumlah` DECIMAL(10,2) NOT NULL COMMENT 'Positive for stock in, negative for stock out',
  `catatan` TEXT,
  `created_at` DATETIME,
  `updated_at` DATETIME
);
""")
    statements.append("\n")
    return "".join(statements)

# --- Data Generation Logic ---
# 1. Bahan Baku Data
bahan_baku_data_list = [{'id': bb['id'], 'nama': bb['nama'], 'satuan': bb['satuan'], 'stok_level': bb['initial_stok'], 'harga': bb['harga'], 'created_at': START_DATE - timedelta(days=1), 'updated_at': START_DATE - timedelta(days=1)} for bb in BAHAN_BAKU_DEFS]
bahan_baku_columns = ['id', 'nama', 'satuan', 'stok_level', 'harga', 'created_at', 'updated_at']

# 2. Produk Jadi Data
produk_jadi_data_list = [{'id': pj['id'], 'kategori': pj['kategori'], 'stok_level': pj['initial_stok'], 'harga': pj['harga'], 'created_at': START_DATE - timedelta(days=1), 'updated_at': START_DATE - timedelta(days=1)} for pj in PRODUK_JADI_DEFS]
produk_jadi_columns = ['id', 'kategori', 'stok_level', 'harga', 'created_at', 'updated_at']

# 3. Resep Produk Data
resep_produk_data_list = [{'id': i + 1, 'produk_jadi_id': resep['produk_jadi_id'], 'bahan_baku_id': resep['bahan_baku_id'], 'jumlah_dibutuhkan': resep['jumlah_dibutuhkan'], 'created_at': START_DATE - timedelta(days=1), 'updated_at': START_DATE - timedelta(days=1)} for i, resep in enumerate(RESEP_PRODUK_DEFS)]
resep_produk_columns = ['id', 'produk_jadi_id', 'bahan_baku_id', 'jumlah_dibutuhkan', 'created_at', 'updated_at']

# --- Simulate Sales and Transactions ---
penjualan_data_list = []
log_transaksi_data_list = []
penjualan_id_counter = 1
log_id_counter = 1

# Initial stock logs
for bb in BAHAN_BAKU_DEFS:
    log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': START_DATE, 'tipe_item': 'bahan_baku', 'item_id': bb['id'], 'tipe_transaksi': 'INITIAL_STOCK', 'jumlah': bb['initial_stok'], 'catatan': f"Stok awal {bb['nama']}", 'created_at': START_DATE, 'updated_at': START_DATE}); log_id_counter += 1
for pj in PRODUK_JADI_DEFS:
    log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': START_DATE, 'tipe_item': 'produk_jadi', 'item_id': pj['id'], 'tipe_transaksi': 'INITIAL_STOCK', 'jumlah': pj['initial_stok'], 'catatan': f"Stok awal {pj['kategori']}", 'created_at': START_DATE, 'updated_at': START_DATE}); log_id_counter += 1

# Recipe and Product Maps for easy lookup
resep_map = {}
for r_item in RESEP_PRODUK_DEFS:
    if r_item['produk_jadi_id'] not in resep_map: resep_map[r_item['produk_jadi_id']] = []
    resep_map[r_item['produk_jadi_id']].append({'bahan_baku_id': r_item['bahan_baku_id'], 'jumlah': r_item['jumlah_dibutuhkan']})

for day_offset in range(DAYS_IN_PERIOD):
    current_date = START_DATE + timedelta(days=day_offset)
    current_dt_for_log = datetime(current_date.year, current_date.month, current_date.day, random.randint(9,17), random.randint(0,59), random.randint(0,59))

    # Simulate Sales
    if random.random() < 0.8: # 80% chance of sales
        num_sales_today = random.randint(1, 5)
        for _ in range(num_sales_today):
            sold_produk = random.choice(PRODUK_JADI_DEFS)
            jumlah_terjual = random.randint(1, 5) # Increased max quantity
            total_harga = jumlah_terjual * sold_produk['harga']
            sale_dt = current_dt_for_log + timedelta(minutes=random.randint(0,60))
            penjualan_data_list.append({'id': penjualan_id_counter, 'tanggal_penjualan': sale_dt, 'produk_jadi_id': sold_produk['id'], 'jumlah_terjual': jumlah_terjual, 'total_harga': total_harga, 'created_at': sale_dt, 'updated_at': sale_dt})
            log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': sale_dt, 'tipe_item': 'produk_jadi', 'item_id': sold_produk['id'], 'tipe_transaksi': 'PENJUALAN', 'jumlah': -jumlah_terjual, 'catatan': f"Penjualan INV{penjualan_id_counter}", 'created_at': sale_dt, 'updated_at': sale_dt}); log_id_counter += 1
            if sold_produk['id'] in resep_map:
                for recipe_item in resep_map[sold_produk['id']]:
                    bb_used_qty = recipe_item['jumlah'] * jumlah_terjual
                    log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': sale_dt, 'tipe_item': 'bahan_baku', 'item_id': recipe_item['bahan_baku_id'], 'tipe_transaksi': 'PENGGUNAAN_PRODUKSI', 'jumlah': -bb_used_qty, 'catatan': f"Untuk INV{penjualan_id_counter}", 'created_at': sale_dt, 'updated_at': sale_dt}); log_id_counter += 1
            penjualan_id_counter += 1

    # Simulate Bahan Baku Purchase (more frequent, larger quantity)
    if day_offset > 0 and day_offset % 7 == 0: # CHANGED to every 7 days
        purchased_bb = random.choice(BAHAN_BAKU_DEFS)
        purchase_qty = random.randint(200, 500) # CHANGED to larger quantity
        purchase_dt = current_dt_for_log - timedelta(hours=random.randint(1,3))
        log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': purchase_dt, 'tipe_item': 'bahan_baku', 'item_id': purchased_bb['id'], 'tipe_transaksi': 'PEMBELIAN_BAHAN_BAKU', 'jumlah': purchase_qty, 'catatan': f"Pembelian {purchased_bb['nama']}", 'created_at': purchase_dt, 'updated_at': purchase_dt}); log_id_counter += 1
        
    # Simulate Produk Jadi Production (more frequent, larger quantity)
    if day_offset > 0 and day_offset % 10 == 0: # CHANGED to every 10 days
        produced_pj = random.choice(PRODUK_JADI_DEFS)
        production_qty = random.randint(50, 150) # CHANGED to larger quantity
        production_dt = current_dt_for_log - timedelta(hours=random.randint(2,5))
        log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': production_dt, 'tipe_item': 'produk_jadi', 'item_id': produced_pj['id'], 'tipe_transaksi': 'PRODUKSI_SELESAI', 'jumlah': production_qty, 'catatan': f"Produksi Selesai Batch {produced_pj['kategori']}-{day_offset}", 'created_at': production_dt, 'updated_at': production_dt}); log_id_counter += 1
        if produced_pj['id'] in resep_map:
            for recipe_item in resep_map[produced_pj['id']]:
                bb_consumed_qty = recipe_item['jumlah'] * production_qty
                log_transaksi_data_list.append({'id': log_id_counter, 'tanggal': production_dt, 'tipe_item': 'bahan_baku', 'item_id': recipe_item['bahan_baku_id'], 'tipe_transaksi': 'PENGGUNAAN_PRODUKSI', 'jumlah': -bb_consumed_qty, 'catatan': f"Untuk Produksi Batch {produced_pj['kategori']}-{day_offset}", 'created_at': production_dt, 'updated_at': production_dt}); log_id_counter += 1

penjualan_columns = ['id', 'tanggal_penjualan', 'produk_jadi_id', 'jumlah_terjual', 'total_harga', 'created_at', 'updated_at']
log_transaksi_columns = ['id', 'tanggal', 'tipe_item', 'item_id', 'tipe_transaksi', 'jumlah', 'catatan', 'created_at', 'updated_at']

# --- Collect all SQL statements ---
all_sql_statements = ["-- Generated Large and Balanced SQL Dataset --\n"]
all_sql_statements.append(f"-- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} --\n\n")
all_sql_statements.append(get_create_table_statements())
all_sql_statements.append(data_to_sql_inserts(bahan_baku_data_list, 'bahan_baku', bahan_baku_columns))
all_sql_statements.append(data_to_sql_inserts(produk_jadi_data_list, 'produk_jadi', produk_jadi_columns))
all_sql_statements.append(data_to_sql_inserts(resep_produk_data_list, 'resep_produk', resep_produk_columns))
all_sql_statements.append(data_to_sql_inserts(penjualan_data_list, 'penjualan', penjualan_columns))
all_sql_statements.append(data_to_sql_inserts(log_transaksi_data_list, 'log_transaksi', log_transaksi_columns))

# --- Write to SQL file ---
try:
    with open(OUTPUT_SQL_FILE, 'w', encoding='utf-8') as f:
        for statement_group in all_sql_statements:
            f.write(statement_group)
    print(f"\nSuccessfully generated large and balanced SQL dataset and saved to '{OUTPUT_SQL_FILE}'")
    print(f"Total Penjualan records generated: {len(penjualan_data_list)}")
    print(f"Total Log Transaksi records generated: {len(log_transaksi_data_list)}")
except IOError as e:
    print(f"\nError writing to file '{OUTPUT_SQL_FILE}': {e}")

print("\n\nNext Steps:")
print("1. Clear your database tables (`TRUNCATE TABLE table_name;` for each table is a good way to start clean).")
print(f"2. Import the newly generated '{OUTPUT_SQL_FILE}' into your database.")
print("3. Re-run `python train.py` to train the models on this new, more balanced data.")
print("4. Restart your Flask API with `python app.py`.")
print("5. Test the API again with Postman. You should now see positive `current_stock` values.")

