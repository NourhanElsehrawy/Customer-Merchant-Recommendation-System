"""
Phase 1: Data Preprocessing Utilities
Clean and prepare data for feature engineering.
"""

import pandas as pd
import numpy as np


def clean_reviews_data(df):
    """Clean reviews dataset for Phase 1"""
    print(f"Cleaning reviews data: {df.shape[0]:,} records")
    
    # Keep essential columns and clean them (using actual column names from DB)
    essential_cols = ['reviewerid', 'asin', 'overall', 'reviewtext', 'summary', 'unixreviewtime', 'verified', 'vote']
    df_clean = df[[col for col in essential_cols if col in df.columns]].copy()
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['reviewerid', 'asin', 'overall'])
    
    # Convert rating to numeric (handling any string/mixed types)
    df_clean['overall'] = pd.to_numeric(df_clean['overall'], errors='coerce')
    df_clean = df_clean.dropna(subset=['overall'])  # Remove invalid ratings
    
    # Filter valid ratings (1-5)
    df_clean = df_clean[(df_clean['overall'] >= 1) & (df_clean['overall'] <= 5)]
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Add text length features for analysis
    if 'reviewtext' in df_clean.columns:
        df_clean['reviewtext_length'] = df_clean['reviewtext'].fillna('').str.len()
    
    if 'summary' in df_clean.columns:
        df_clean['summary_length'] = df_clean['summary'].fillna('').str.len()
    
    print(f"Cleaned reviews data: {df_clean.shape[0]:,} records ({df.shape[0] - df_clean.shape[0]:,} removed)")
    return df_clean


def clean_products_data(df):
    """Clean products dataset for Phase 1"""
    print(f"Cleaning products data: {df.shape[0]:,} records")
    
    # Keep essential columns (using actual column names from DB)
    essential_cols = ['asin', 'title', 'main_cat', 'brand', 'price', 'category']
    df_clean = df[[col for col in essential_cols if col in df.columns]].copy()
    
    # Use 'category' if 'main_cat' is not available
    if 'main_cat' not in df_clean.columns and 'category' in df_clean.columns:
        df_clean['main_cat'] = df_clean['category']
    
    # Remove rows without asin
    df_clean = df_clean.dropna(subset=['asin'])
    
    # Remove duplicates based on asin
    df_clean = df_clean.drop_duplicates(subset=['asin'])
    
    # Fill missing values with proper defaults
    df_clean['title'] = df_clean['title'].fillna('Unknown Product')
    df_clean['main_cat'] = df_clean['main_cat'].fillna('Other')
    df_clean['brand'] = df_clean['brand'].fillna('Unknown')
    
    # Add title length feature for analysis
    df_clean['title_length'] = df_clean['title'].str.len()
    
    print(f"Cleaned products data: {df_clean.shape[0]:,} records ({df.shape[0] - df_clean.shape[0]:,} removed)")
    return df_clean


def create_interaction_matrix(reviews_df, min_interactions=5):
    """
    Create user-item interaction matrix
    
    Args:
        reviews_df: Cleaned reviews data
        min_interactions: Minimum interactions to keep users/items
    Returns:
        pd.DataFrame: Interaction matrix
    """
    # Filter users and items with minimum interactions
    user_counts = reviews_df['reviewerid'].value_counts()
    item_counts = reviews_df['asin'].value_counts()
    
    active_users = user_counts[user_counts >= min_interactions].index
    active_items = item_counts[item_counts >= min_interactions].index
    
    filtered_df = reviews_df[
        (reviews_df['reviewerid'].isin(active_users)) &
        (reviews_df['asin'].isin(active_items))
    ]
    
    # Create interaction matrix
    matrix = pd.pivot_table(
        filtered_df,
        values='overall',
        index='reviewerid', 
        columns='asin',
        aggfunc='mean',
        fill_value=0
    )
    
    return matrix


def get_data_quality_report(reviews_df, products_df):
    """
    Generate comprehensive data quality report for Amazon datasets
    
    Args:
        reviews_df: Raw reviews DataFrame
        products_df: Raw products DataFrame
    
    Returns:
        dict: Quality metrics for reviews, products, and integration
    """
    report = {
        'reviews': {},
        'products': {},
        'integration': {}
    }
    
    # Reviews data quality
    reviews_total = len(reviews_df)
    report['reviews']['total_records'] = reviews_total
    report['reviews']['missing_reviewer_id'] = reviews_df['reviewerid'].isnull().sum()
    report['reviews']['missing_asin'] = reviews_df['asin'].isnull().sum()
    report['reviews']['missing_rating'] = reviews_df['overall'].isnull().sum()
    report['reviews']['duplicate_records'] = reviews_df.duplicated().sum()
    report['reviews']['unique_customers'] = reviews_df['reviewerid'].nunique()
    report['reviews']['unique_products'] = reviews_df['asin'].nunique()
    
    # Rating validation
    if 'overall' in reviews_df.columns:
        # Convert to numeric for validation
        ratings_numeric = pd.to_numeric(reviews_df['overall'], errors='coerce')
        report['reviews']['invalid_ratings'] = ratings_numeric.isnull().sum()
        valid_ratings = ratings_numeric.dropna()
        report['reviews']['ratings_out_of_range'] = ((valid_ratings < 1) | (valid_ratings > 5)).sum()
    
    # Text content quality
    if 'reviewtext' in reviews_df.columns:
        report['reviews']['missing_review_text'] = reviews_df['reviewtext'].isnull().sum()
        report['reviews']['empty_review_text'] = (reviews_df['reviewtext'].fillna('').str.strip() == '').sum()
    
    if 'summary' in reviews_df.columns:
        report['reviews']['missing_summary'] = reviews_df['summary'].isnull().sum()
    
    # Products data quality
    products_total = len(products_df)
    report['products']['total_records'] = products_total
    report['products']['missing_asin'] = products_df['asin'].isnull().sum()
    report['products']['duplicate_asins'] = products_df['asin'].duplicated().sum()
    
    if 'title' in products_df.columns:
        report['products']['missing_title'] = products_df['title'].isnull().sum()
    
    if 'main_cat' in products_df.columns:
        report['products']['missing_category'] = products_df['main_cat'].isnull().sum()
        report['products']['unique_categories'] = products_df['main_cat'].nunique()
    
    if 'brand' in products_df.columns:
        report['products']['missing_brand'] = products_df['brand'].isnull().sum()
        report['products']['unique_brands'] = products_df['brand'].nunique()
    
    # Integration quality
    reviewed_products = set(reviews_df['asin'].dropna().unique())
    product_catalog = set(products_df['asin'].dropna().unique())
    
    report['integration']['products_with_reviews'] = len(reviewed_products)
    report['integration']['products_in_catalog'] = len(product_catalog)
    report['integration']['products_with_metadata'] = len(reviewed_products.intersection(product_catalog))
    
    # Coverage percentages
    if len(reviewed_products) > 0:
        report['integration']['metadata_coverage'] = (len(reviewed_products.intersection(product_catalog)) / len(reviewed_products)) * 100
    else:
        report['integration']['metadata_coverage'] = 0.0
    
    report['integration']['orphaned_products'] = len(product_catalog - reviewed_products)
    report['integration']['unmatched_reviews'] = len(reviewed_products - product_catalog)
    
    return report