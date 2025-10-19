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


def create_sparse_interaction_matrix(df, min_interactions=5):
    """Efficient sparse user-item matrix with improvements"""
    log.info("Creating sparse interaction matrix...")
    
    # Validate input
    if df.empty or 'reviewerid' not in df.columns or 'asin' not in df.columns:
        log.warning("Invalid input data for interaction matrix")
        return csr_matrix((0, 0)), np.array([]), np.array([])
    
    # Filter by minimum interactions
    log.info(f"Original data: {len(df):,} interactions")
    user_counts = df['reviewerid'].value_counts()
    item_counts = df['asin'].value_counts()
    log.info(f"Users with >={min_interactions} interactions: {sum(user_counts >= min_interactions):,}")
    log.info(f"Items with >={min_interactions} interactions: {sum(item_counts >= min_interactions):,}")
    
    active_users = user_counts[user_counts >= min_interactions].index
    active_items = item_counts[item_counts >= min_interactions].index
    
    filtered_df = df[
        df['reviewerid'].isin(active_users) & 
        df['asin'].isin(active_items)
    ].copy()
    
    log.info(f"After filtering: {len(filtered_df):,} interactions")
    
    if filtered_df.empty:
        log.warning("No interactions remain after filtering")
        return csr_matrix((0, 0)), np.array([]), np.array([])
    
    # Handle duplicates by taking mean rating
    before_dedup = len(filtered_df)
    filtered_df = filtered_df.groupby(['reviewerid', 'asin'])['overall'].mean().reset_index()
    log.info(f"After deduplication: {len(filtered_df):,} interactions (removed {before_dedup - len(filtered_df):,} duplicates)")
    
    # Create categorical mappings
    users = pd.Categorical(filtered_df['reviewerid'])
    items = pd.Categorical(filtered_df['asin'])
    values = filtered_df['overall'].astype('float32')
    
    # Debug: Check data before matrix creation
    log.info(f"Building matrix from {len(filtered_df):,} interactions")
    log.info(f"User codes range: {users.codes.min()} to {users.codes.max()}")
    log.info(f"Item codes range: {items.codes.min()} to {items.codes.max()}")
    log.info(f"Rating values range: {values.min():.1f} to {values.max():.1f}")
    
    # Create sparse matrix - fix the shape issue
    n_users = len(users.categories)
    n_items = len(items.categories)
    
    mat = csr_matrix((values, (users.codes, items.codes)), shape=(n_users, n_items))
    
    # Verify matrix has data
    if mat.nnz == 0:
        log.error("Matrix has no non-zero entries! Check input data.")
        log.error(f"Input values: min={values.min()}, max={values.max()}, count={len(values)}")
    
    density = mat.nnz / (n_users * n_items) if n_users > 0 and n_items > 0 else 0
    log.info(f"Sparse matrix: {n_users:,} users × {n_items:,} items ({density*100:.3f}% density)")
    log.info(f"Non-zero entries: {mat.nnz:,}")
    
    return mat, users.categories, items.categories



def create_all_features(reviews_df, products_df=None, cutoff_date=None, save_path=None):
    """Main feature pipeline"""
    log.info("Starting feature creation...")
    df = optimize_dtypes(apply_temporal_cutoff(reviews_df, cutoff_date))

    user_feats = create_user_features(df)
    item_feats = create_item_features(df, products_df)
    inter_feats = create_interaction_features(df, user_feats, item_feats)
    sparse_mat, users, items = create_sparse_interaction_matrix(df)

    features = {
        'user_features': user_feats,
        'item_features': item_feats,
        'interaction_features': inter_feats,
        'interaction_matrix': sparse_mat
    }

    if save_path:
        user_feats.to_parquet(f"{save_path}/user_features.parquet")
        item_feats.to_parquet(f"{save_path}/item_features.parquet")
        inter_feats.to_parquet(f"{save_path}/interaction_features.parquet")
        log.info(f"Features saved to {save_path}")

    log.info(f"Feature creation completed using {len(df):,} reviews")
    return features



def load_and_create_features(cutoff_date=None):
    from db_loader import load_amazon_data_k_core
    from preprocessing import clean_reviews_data, clean_products_data

    log.info("Loading data...")
    reviews_raw, products_raw = load_amazon_data_k_core()
    reviews_clean = clean_reviews_data(reviews_raw)
    products_clean = clean_products_data(products_raw)
    return create_all_features(reviews_clean, products_clean, cutoff_date)
