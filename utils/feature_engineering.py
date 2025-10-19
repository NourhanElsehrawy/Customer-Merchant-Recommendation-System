"""
Optimized Feature Engineering for Recommendation System
Lightweight, memory-safe version (<200 lines)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def optimize_dtypes(df):
    """Reduce memory footprint by downcasting numeric columns"""
    for c in df.select_dtypes(include='int').columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    for c in df.select_dtypes(include='float').columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    return df


def apply_temporal_cutoff(df, cutoff_date=None):
    """Prevent leakage by training only on data before cutoff_date"""
    if cutoff_date and 'unixreviewtime' in df.columns:
        df['review_date'] = pd.to_datetime(df['unixreviewtime'], unit='s')
        df = df[df['review_date'] <= pd.to_datetime(cutoff_date)]
        log.info(f"Applied temporal cutoff: {len(df):,} reviews remain")
    return df


def create_user_features(df):
    """Aggregate essential user-level signals"""
    log.info("Building user features...")
    if 'verified' in df.columns and df['verified'].dtype == 'object':
        df['verified'] = df['verified'].isin(['True', 'true']).astype(int)

    g = df.groupby('reviewerid')
    user_feats = g['overall'].agg(['count', 'mean', 'std']).rename(
        columns={'count':'user_review_count','mean':'user_avg_rating','std':'user_rating_std'}
    )
    user_feats['user_positive_ratio'] = g['overall'].apply(lambda x: (x >= 4).mean())
    user_feats['user_rating_range'] = g['overall'].apply(lambda x: x.max() - x.min())
    if 'verified' in df.columns:
        user_feats['user_verified_ratio'] = g['verified'].mean()
    return user_feats.fillna(0)


def create_item_features(df, products_df=None):
    """Aggregate essential item-level signals"""
    log.info("Building item features...")
    if 'verified' in df.columns and df['verified'].dtype == 'object':
        df['verified'] = df['verified'].isin(['True', 'true']).astype(int)

    g = df.groupby('asin')
    item_feats = g['overall'].agg(['count', 'mean', 'std']).rename(
        columns={'count':'item_review_count','mean':'item_avg_rating','std':'item_rating_std'}
    )
    item_feats['item_unique_reviewers'] = g['reviewerid'].nunique()
    item_feats['item_positive_ratio'] = g['overall'].apply(lambda x: (x >= 4).mean())
    item_feats['item_rating_skew'] = g['overall'].apply(lambda x: x.skew() if len(x) > 2 else 0)
    if 'verified' in df.columns:
        item_feats['item_verified_ratio'] = g['verified'].mean()

    if products_df is not None and 'main_cat' in products_df.columns:
        category_sizes = products_df['main_cat'].value_counts()
        item_feats['item_category_size'] = products_df.set_index('asin')['main_cat'].map(category_sizes)
    return item_feats.fillna(0)


def create_interaction_features(df, user_feats=None, item_feats=None):
    """Simple user–item interaction-level stats"""
    log.info("Building interaction features...")
    inter = df[['reviewerid', 'asin', 'overall']].rename(columns={'reviewerid':'user','asin':'item','overall':'rating'})

    if user_feats is not None:
        inter = inter.merge(user_feats[['user_avg_rating']], left_on='user', right_index=True, how='left')
    else:
        inter['user_avg_rating'] = df.groupby('reviewerid')['overall'].transform('mean')

    if item_feats is not None:
        inter = inter.merge(item_feats[['item_avg_rating']], left_on='item', right_index=True, how='left')
    else:
        inter['item_avg_rating'] = df.groupby('asin')['overall'].transform('mean')

    inter['rating_deviation'] = inter['rating'] - (inter['user_avg_rating'] + inter['item_avg_rating']) / 2
    return inter.fillna(0)


def create_sparse_interaction_matrix(df, products_df, min_interactions=5):
    """Efficient sparse customer-merchant matrix"""
    log.info("Creating sparse customer-merchant interaction matrix...")
    
    # Validate input
    if df.empty or 'reviewerid' not in df.columns or 'asin' not in df.columns:
        log.warning("Invalid input data for interaction matrix")
        return csr_matrix((0, 0)), np.array([]), np.array([])
    
    if products_df is None or 'brand' not in products_df.columns:
        log.warning("Products data with brand info required for merchant matrix")
        return csr_matrix((0, 0)), np.array([]), np.array([])
    
    # Merge with products to get merchant info
    df_with_merchants = df.merge(products_df[['asin', 'brand']], on='asin', how='left')
    df_with_merchants = df_with_merchants.dropna(subset=['brand'])
    
    log.info(f"Original data: {len(df):,} interactions")
    log.info(f"With merchant info: {len(df_with_merchants):,} interactions")
    
    # Filter by minimum interactions
    customer_counts = df_with_merchants['reviewerid'].value_counts()
    merchant_counts = df_with_merchants['brand'].value_counts()
    log.info(f"Customers with >={min_interactions} interactions: {sum(customer_counts >= min_interactions):,}")
    log.info(f"Merchants with >={min_interactions} interactions: {sum(merchant_counts >= min_interactions):,}")
    
    active_customers = customer_counts[customer_counts >= min_interactions].index
    active_merchants = merchant_counts[merchant_counts >= min_interactions].index
    
    filtered_df = df_with_merchants[
        df_with_merchants['reviewerid'].isin(active_customers) & 
        df_with_merchants['brand'].isin(active_merchants)
    ].copy()
    
    log.info(f"After filtering: {len(filtered_df):,} interactions")
    
    if filtered_df.empty:
        log.warning("No interactions remain after filtering")
        return csr_matrix((0, 0)), np.array([]), np.array([])
    
    # Aggregate customer-merchant interactions (mean rating + interaction count bonus)
    customer_merchant_agg = filtered_df.groupby(['reviewerid', 'brand']).agg({
        'overall': ['mean', 'count']
    }).reset_index()
    
    customer_merchant_agg.columns = ['reviewerid', 'brand', 'mean_rating', 'interaction_count']
    customer_merchant_agg['interaction_score'] = (
        customer_merchant_agg['mean_rating'] * 0.7 + 
        customer_merchant_agg['interaction_count'] * 0.3
    )
    
    log.info(f"Aggregated to {len(customer_merchant_agg):,} customer-merchant pairs")
    
    # Create categorical mappings
    customers = pd.Categorical(customer_merchant_agg['reviewerid'])
    merchants = pd.Categorical(customer_merchant_agg['brand'])
    values = customer_merchant_agg['interaction_score'].astype('float32')
    
    # Create sparse matrix
    n_customers = len(customers.categories)
    n_merchants = len(merchants.categories)
    
    mat = csr_matrix((values, (customers.codes, merchants.codes)), shape=(n_customers, n_merchants))
    
    density = mat.nnz / (n_customers * n_merchants) if n_customers > 0 and n_merchants > 0 else 0
    log.info(f"Customer-Merchant matrix: {n_customers:,} customers × {n_merchants:,} merchants ({density*100:.3f}% density)")
    log.info(f"Non-zero entries: {mat.nnz:,}")
    
    return mat, customers.categories, merchants.categories



def create_all_features(reviews_df, products_df=None, cutoff_date=None, save_path=None):
    """Main feature pipeline - creates customer-merchant features"""
    log.info("Starting customer-merchant feature creation...")
    df = optimize_dtypes(apply_temporal_cutoff(reviews_df, cutoff_date))

    user_feats = create_user_features(df)
    item_feats = create_item_features(df, products_df)
    inter_feats = create_interaction_features(df, user_feats, item_feats)
    
    # Create customer-merchant interaction matrix
    merchant_matrix, customers, merchants = create_sparse_interaction_matrix(df, products_df)
    
    # Create merchant features
    if products_df is not None and 'brand' in products_df.columns:
        df_with_merchants = df.merge(products_df[['asin', 'brand']], on='asin', how='left')
        df_with_merchants = df_with_merchants.dropna(subset=['brand'])
        
        merchant_feats = df_with_merchants.groupby('brand').agg({
            'overall': ['count', 'mean', 'std'],
            'reviewerid': 'nunique'
        }).round(2)
        merchant_feats.columns = ['review_count', 'avg_rating', 'rating_std', 'customer_count']
        merchant_feats['rating_std'] = merchant_feats['rating_std'].fillna(0)
    else:
        merchant_feats = pd.DataFrame()

    features = {
        'user_features': user_feats,
        'item_features': item_feats,
        'interaction_features': inter_feats,
        'interaction_matrix': merchant_matrix,  # Now customer-merchant matrix
        'merchant_features': merchant_feats,
        'customers': customers,
        'merchants': merchants
    }

    if save_path:
        user_feats.to_parquet(f"{save_path}/user_features.parquet")
        item_feats.to_parquet(f"{save_path}/item_features.parquet")
        inter_feats.to_parquet(f"{save_path}/interaction_features.parquet")
        if not merchant_feats.empty:
            merchant_feats.to_parquet(f"{save_path}/merchant_features.parquet")
        log.info(f"Features saved to {save_path}")

    log.info(f"Customer-merchant feature creation completed using {len(df):,} reviews")
    return features



def load_and_create_features(cutoff_date=None):
    from db_loader import load_amazon_data_k_core
    from preprocessing import clean_reviews_data, clean_products_data

    log.info("Loading data...")
    reviews_raw, products_raw = load_amazon_data_k_core()
    reviews_clean = clean_reviews_data(reviews_raw)
    products_clean = clean_products_data(products_raw)
    return reviews_raw, products_raw, create_all_features(reviews_clean, products_clean, cutoff_date)
