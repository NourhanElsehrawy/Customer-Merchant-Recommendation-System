"""
Database Data Loader
===================
Load Amazon data from PostgreSQL database
"""

import os
import pandas as pd
import psycopg2


def get_db_connection():
    """Get raw psycopg2 connection for pandas compatibility"""
    host = os.getenv('POSTGRES_HOST', 'postgres')
    port = os.getenv('POSTGRES_PORT', '5432')
    user = os.getenv('POSTGRES_USER', 'airflow')
    password = os.getenv('POSTGRES_PASSWORD', 'airflow')
    database = os.getenv('POSTGRES_DATABASE', 'amazon_recommendation')
    
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )


def load_amazon_data_from_db():
    """
    Load Amazon data from PostgreSQL database
        
    Returns:
    --------
    tuple: (reviews_df, products_df)
    """
    conn = get_db_connection()
    
    try:
        # Load reviews data
        reviews_query = "SELECT * FROM reviews"
        reviews_df = pd.read_sql_query(reviews_query, conn)
        
        # Load products data
        products_query = "SELECT * FROM relevant_products"
        products_df = pd.read_sql_query(products_query, conn)
        
        return reviews_df, products_df
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        raise
        
    finally:
        conn.close()


def check_database_tables():
    """Check what tables exist in the database"""
    try:
        conn = get_db_connection()
        
        # Get list of tables
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        
        tables_df = pd.read_sql_query(tables_query, conn)
        print("Available tables:")
        
        table_list = []
        for table in tables_df['table_name']:
            print(f"  - {table}")
            table_list.append(table)
            
            # Get row count
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                count_df = pd.read_sql_query(count_query, conn)
                print(f"    {count_df.iloc[0]['count']:,} rows")
            except Exception as count_error:
                print(f"    Error getting count: {count_error}")
        
        conn.close()
        return table_list
        
    except Exception as e:
        print(f"Error checking database: {e}")
        return []


def apply_k_core_filter(reviews_df, k=5):
    """
    Apply k-core filtering to ensure both users and items have at least k reviews
    
    Parameters:
    -----------
    reviews_df : pd.DataFrame
        Reviews dataframe with 'reviewerid' and 'asin' columns
    k : int, default=5
        Minimum number of reviews required for both users and items
        
    Returns:
    --------
    pd.DataFrame: Filtered reviews dataframe where every user has ≥k reviews 
                  and every item has ≥k reviews
    """
    print(f"Applying {k}-core filtering...")
    print(f"Original dataset: {len(reviews_df):,} reviews")
    print(f"Original users: {reviews_df['reviewerid'].nunique():,}")
    print(f"Original items: {reviews_df['asin'].nunique():,}")
    
    df = reviews_df.copy()
    
    # Iteratively filter until convergence
    iteration = 0
    prev_size = 0
    
    while len(df) != prev_size:
        iteration += 1
        prev_size = len(df)
        
        # Count reviews per user and item
        user_counts = df['reviewerid'].value_counts()
        item_counts = df['asin'].value_counts()
        
        # Filter users with at least k reviews
        valid_users = user_counts[user_counts >= k].index
        df = df[df['reviewerid'].isin(valid_users)]
        
        # Filter items with at least k reviews  
        valid_items = item_counts[item_counts >= k].index
        df = df[df['asin'].isin(valid_items)]
        
        print(f"Iteration {iteration}: {len(df):,} reviews, "
              f"{df['reviewerid'].nunique():,} users, "
              f"{df['asin'].nunique():,} items")
    
    print(f"\nFinal {k}-core dataset:")
    print(f"Reviews: {len(df):,} ({len(df)/len(reviews_df)*100:.1f}% of original)")
    print(f"Users: {df['reviewerid'].nunique():,} ({df['reviewerid'].nunique()/reviews_df['reviewerid'].nunique()*100:.1f}% of original)")
    print(f"Items: {df['asin'].nunique():,} ({df['asin'].nunique()/reviews_df['asin'].nunique()*100:.1f}% of original)")
    
    # Verify k-core property
    final_user_counts = df['reviewerid'].value_counts()
    final_item_counts = df['asin'].value_counts()
    
    min_user_reviews = final_user_counts.min()
    min_item_reviews = final_item_counts.min()
    
    print(f"\nVerification:")
    print(f"Minimum reviews per user: {min_user_reviews}")
    print(f"Minimum reviews per item: {min_item_reviews}")
    print(f"{k}-core property satisfied: {min_user_reviews >= k and min_item_reviews >= k}")
    
    return df


def load_amazon_data_k_core(k=5):
    """
    Load Amazon data from PostgreSQL database and apply k-core filtering

    Returns:
    --------
    tuple: (reviews_k_core_df, products_df)
        - reviews_k_core_df: k-core filtered reviews (users and items with ≥k reviews)
        - products_df: All products (unchanged)
    """
    print("Loading Amazon data with 5-core filtering...")
    
    # Load original data
    reviews_df, products_df = load_amazon_data_from_db()
    
    # Apply 5-core filtering to reviews
    reviews_5core = apply_k_core_filter(reviews_df, k=5)
    
    # Filter products to only those that appear in 5-core reviews
    valid_asins = set(reviews_5core['asin'].unique())
    products_filtered = products_df[products_df['asin'].isin(valid_asins)].copy()
    
    print(f"\nProducts filtered: {len(products_df)} → {len(products_filtered)} "
          f"({len(products_filtered)/len(products_df)*100:.1f}% retained)")
    
    return reviews_5core, products_filtered