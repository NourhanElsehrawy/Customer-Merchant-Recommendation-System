# Customer-Merchant Recommendation System

A machine learning recommendation system built with Amazon Electronics dataset that suggests relevant products to customers using multiple algorithms.

## What it does

- **Recommends products** to customers based on their purchase history
- **Uses multiple algorithms**: Popularity, Collaborative Filtering, Content-Based, and Hybrid models
- **Handles new users** with fallback mechanisms
- **Shows product names** and recommendation scores

## Project Structure

```
├── notebooks/                 
│   ├── 01_Data_Exploration_and_Processing.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Recommendation_System.ipynb  
├── utils/                     # Data processing utilities
├── airflow/                  
└── data/                      # Amazon dataset
```

## Quick Start

1. **Setup environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the system**:
   - Open `notebooks/03_Recommendation_System.ipynb`
   - Run all cells to see recommendations in action

## Key Features

- **Multiple Models**: Popularity, Collaborative Filtering, Content-Based, Hybrid
- **Advanced Features**: Cold user filtering, hyperparameter tuning, text embeddings
- **Memory Efficient**: Handles large datasets with sparse matrices
- **Production Ready**: Includes deployment guidance and monitoring
- **Comprehensive Evaluation**: Precision, Recall, Coverage metrics

## Models Included

1. **Popularity**: Recommends trending products
2. **Collaborative Filtering**: "Users like you also liked..."
3. **Content-Based**: Products similar to your interests  
4. **Hybrid**: Combines multiple approaches intelligently
5. **Advanced Models**: NMF, ALS algorithms

## Technology Stack

- **Python**: pandas, numpy, scikit-learn
- **Machine Learning**: SVD, NMF, TF-IDF, cosine similarity  
- **Data Processing**: Apache Airflow 
- **Visualization**: matplotlib

## Results

The system provides personalized recommendations with explanations:
```
Recommendations for User 123:
1. Wireless Headphones (Score: 0.95) - Similar users liked
2. Gaming Mouse (Score: 0.87) - Based on your interests  
3. USB Cable (Score: 0.72) - Popular item
```