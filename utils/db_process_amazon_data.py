"""
Simple Amazon Data Loader
========================
Load Amazon .gz files into PostgreSQL
"""

import os
import json
import gzip
import psycopg2


def create_database_if_not_exists(db_name):
    """Create PostgreSQL database if it doesn't exist"""
    print(f"Checking/creating database: {db_name}", flush=True)
    
    # Connect to postgres database first
    conn_admin = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database='airflow',
        user=os.getenv('POSTGRES_USER', 'airflow'),
        password=os.getenv('POSTGRES_PASSWORD', 'airflow')
    )
    conn_admin.autocommit = True
    cursor_admin = conn_admin.cursor()
    
    try:
        cursor_admin.execute(f'CREATE DATABASE "{db_name}"')
        print(f"Created database: {db_name}", flush=True)
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database {db_name} already exists", flush=True)
    finally:
        cursor_admin.close()
        conn_admin.close()


def load_gz_file(file_path, table_name, conn, max_records=None):
    """Load .gz JSON file into SQLite table"""
    print(f"Loading {file_path} into {table_name}...", flush=True)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Discover columns from first few records
    columns = set()
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Sample first 10 records
                break
            try:
                data = json.loads(line.strip())
                columns.update(data.keys())
            except:
                continue
    
    columns = sorted(list(columns))
    print(f"Found columns: {columns}", flush=True)
    
    # Create table (PostgreSQL syntax)
    cursor = conn.cursor()
    if table_name == 'reviews':
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} (id SERIAL PRIMARY KEY, {', '.join([f'{col} TEXT' for col in columns])})"
    else:
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} TEXT' for col in columns])})"
    
    cursor.execute(sql)
    conn.commit()
    
    # Load data
    records_loaded = 0
    batch = []
    
    # PostgreSQL uses %s placeholders
    placeholders = ', '.join(['%s' for _ in columns])
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if max_records and records_loaded >= max_records:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Extract values in column order
                values = []
                for col in columns:
                    value = data.get(col)
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value) if value else None
                    elif value is not None:
                        value = str(value)
                    
                   
                    if value:
                        value = value.replace('\x00', '')
                    
                    values.append(value)
                
                batch.append(values)
                records_loaded += 1
                
                # Insert in batches of 1000
                if len(batch) >= 1000:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                    batch = []
                    
                    if records_loaded % 100000 == 0:
                        print(f"  Loaded {records_loaded:,} records...", flush=True)
                        
            except Exception as e:
                print(f"Error processing record: {e}", flush=True)
                continue
    
    # Insert remaining records
    if batch:
        cursor.executemany(insert_sql, batch)
        conn.commit()
    
    print(f"Saved {records_loaded:,} records into {table_name}", flush=True)
    return records_loaded


def main():
    print("Amazon Data Loader", flush=True)
    print("=" * 30, flush=True)
    
    # Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLE_SIZE = 4000000  # Number of reviews to load
    
    # Create database if needed
    db_name = 'amazon_recommendation'
    create_database_if_not_exists(db_name)
    
    # Connect to target database
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=db_name,
        user=os.getenv('POSTGRES_USER', 'airflow'),
        password=os.getenv('POSTGRES_PASSWORD', 'airflow')
    )
    
    try:
        # Load all products
        products_file = os.path.join(DATA_DIR, "meta_Electronics.json.gz")
        load_gz_file(products_file, 'products', conn)

        # Load reviews
        reviews_file = os.path.join(DATA_DIR, "Electronics_5.json.gz")
        load_gz_file(reviews_file, 'reviews', conn, max_records=SAMPLE_SIZE)
        

        
        # Create relevant_products table
        print("Creating relevant_products table...")
        
        # Use asin as the link column (we know this from the Amazon data)
        cursor = conn.cursor()
        link_col = 'asin'
        print(f"Using '{link_col}' to link tables")
        
        # Create relevant_products
        cursor.execute(f"""
            CREATE TABLE relevant_products AS 
            SELECT * FROM products 
            WHERE {link_col} IN (SELECT DISTINCT {link_col} FROM reviews WHERE {link_col} IS NOT NULL)
        """)
        
        # Drop original products table
        cursor.execute("DROP TABLE products")
        conn.commit()
        
        # Get final counts
        cursor.execute("SELECT COUNT(*) FROM reviews")
        review_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM relevant_products")
        product_count = cursor.fetchone()[0]
        
        print(f"\nDatabase: PostgreSQL")
        print(f"Reviews: {review_count:,}")
        print(f"Relevant Products: {product_count:,}")
            
    finally:
        conn.close()


if __name__ == "__main__":
    main()
