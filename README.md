# README: Recipe Recommendation System

This project implements a recipe recommendation system based on user ratings, protein content normalization, and calorie constraints. The system compares two algorithms: K-Nearest Neighbors (KNNBasic) and Matrix Factorization (SVD), evaluating their performance and predicting user preferences.

## Overview

The project includes the following key steps:
1. Data Preprocessing
2. Protein Content Normalization
3. Recommendation System with KNNBasic (Item-Based Collaborative Filtering)
4. Recommendation System with Matrix Factorization (SVD)
5. Model Evaluation and Comparison
6. Binary Classification for Threshold-Based Recommendations

---

## Step-by-Step Implementation

### 1. Data Preprocessing

```python
# Find the minimum and maximum values in the 'protein (PDV)' column
min_protein = df['protein (PDV)'].min()
max_protein = df['protein (PDV)'].max()

print("Minimum protein:", min_protein)
print("Maximum protein:", max_protein)
```

### 2. Protein Content Normalization

Normalize the `protein (PDV)` column to a range of 1 to 5.

```python
# Original range of protein values
original_min = 2
original_max = 184.5

# New range for normalization
new_min = 1
new_max = 5

# Normalize the column
df['protein_normalized'] = new_min + ((df['protein (PDV)'] - original_min) * (new_max - new_min) / (original_max - original_min))
```

### 3. Recommendation System with KNNBasic

#### 3.1 Data Loading
Define the dataset reader and load the data for collaborative filtering.

```python
from surprise import Dataset, Reader

reader = Reader(line_format='user item rating', rating_scale=(min_protein, max_protein))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'protein_normalized']], reader)
```

#### 3.2 Splitting Data

```python
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```

#### 3.3 Model Training and Prediction

```python
from surprise import KNNBasic

# Define item-based similarity options
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based filtering
}

knn_model = KNNBasic(sim_options=sim_options)
knn_model.fit(trainset)
```

#### 3.4 Generate Recommendations

```python
def get_recommendations(user_id, calorie_limit):
    all_recipe_ids = df['recipe_id'].unique()
    rated_recipe_ids = df.loc[df['user_id'] == user_id, 'recipe_id'].tolist()

    unrated_recipe_ids = [recipe_id for recipe_id in all_recipe_ids if recipe_id not in rated_recipe_ids]
    recommendations = []

    for recipe_id in unrated_recipe_ids:
        prediction = knn_model.predict(user_id, recipe_id)
        recommendations.append((recipe_id, prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:10]

    filtered_recommendations = []
    for recipe_id, estimated_rating in top_recommendations:
        calories = df.loc[df['recipe_id'] == recipe_id, 'calories'].iloc[0]
        if calories <= calorie_limit:
            recipe_info = df.loc[df['recipe_id'] == recipe_id, ['name', 'description']].iloc[0]
            filtered_recommendations.append((recipe_id, recipe_info['name'], recipe_info['description']))

    return pd.DataFrame(filtered_recommendations, columns=['recipe_id', 'name', 'description'])
```

Example:

```python
user_id = 29956
calorie_limit = 2000
recommendations = get_recommendations(user_id, calorie_limit)
print(recommendations)
```

---

### 4. Recommendation System with Matrix Factorization (SVD)

#### 4.1 Model Training

```python
from surprise import SVD

mf_model = SVD()
mf_model.fit(trainset)
```

#### 4.2 Generate Recommendations

The process is identical to the KNNBasic model but uses the `mf_model` for predictions.

```python
recommendations = get_recommendations(user_id, calorie_limit)
print(recommendations)
```

---

### 5. Model Evaluation

#### Cross-Validation for KNNBasic and SVD

```python
from surprise.model_selection import cross_validate

# KNNBasic Model
knn_results = cross_validate(knn_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# SVD Model
mf_results = cross_validate(mf_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print Results
print("KNNBasic Model - Avg RMSE: {:.4f}, Avg MAE: {:.4f}".format(knn_results['test_rmse'].mean(), knn_results['test_mae'].mean()))
print("SVD Model - Avg RMSE: {:.4f}, Avg MAE: {:.4f}".format(mf_results['test_rmse'].mean(), mf_results['test_mae'].mean()))
```

---

### 6. Binary Classification for Threshold-Based Recommendations

Convert predictions into binary classifications and calculate precision, recall, and F1-score.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Predict Test Set
predictions = mf_model.test(testset)

# Convert Predicted Ratings to Binary
threshold = 3.5
binary_predictions = [1 if pred.est >= threshold else 0 for pred in predictions]

# Convert Actual Ratings to Binary
actual_ratings = [1 if actual_rating >= threshold else 0 for _, _, actual_rating in testset]

# Calculate Metrics
precision = precision_score(actual_ratings, binary_predictions)
recall = recall_score(actual_ratings, binary_predictions)
f1 = f1_score(actual_ratings, binary_predictions)

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))
```

---

## Results

- **KNNBasic Model**
  - Avg RMSE: 0.9482
  - Avg MAE: 0.7694

- **SVD Model**
  - Avg RMSE: 0.7358
  - Avg MAE: 0.4562

- **Binary Classification Metrics**
  - Precision: 0.9415
  - Recall: 0.9995
  - F1-Score: 0.9696
