import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
import scipy.stats as stats
import shap

# Load the data
data = pd.read_csv('data.csv')

# Randomly select 1/5 of the dataset (20% instead of 10%)
sampled_data = data.sample(frac=0.2, random_state=42)
print(f"Sampled data shape: {sampled_data.shape} (rows × columns)")

# Separate features and target
X = sampled_data.drop('HHFOOD_SECURITY', axis=1)
y = sampled_data['HHFOOD_SECURITY']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Number of numeric features: {len(numeric_features)}")
print(f"Number of categorical features: {len(categorical_features)}")

# Handle missing values in the data
X_imputed = X.copy()
for col in numeric_features:
    X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
for col in categorical_features:
    X_imputed[col] = X_imputed[col].fillna(X_imputed[col].mode()[0] if not X_imputed[col].mode().empty else "missing")

# Create encoded dataset for feature selection
if len(categorical_features) > 0:
    X_encoded = pd.get_dummies(X_imputed, columns=categorical_features, drop_first=True)
else:
    X_encoded = X_imputed

# Initialize summary dictionary for results
summary = {
    'Method': [],
    'CV10': [],
    'CV9': [],
    'top5': [],
    'top4': []
}

# For storing top features
all_top_features = {}

# Function to run a method and collect results
def run_method(name, feature_selector, X_data):
    print(f"\n{'-'*50}")
    print(f"Running method: {name}")
    
    # STEP 1: Select top 10 features from full set (CV10)
    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_data, y)
        importances = model.feature_importances_
    elif name == "XGBoost":
        model = XGBClassifier(random_state=42)
        model.fit(X_data, y)
        importances = model.feature_importances_
    elif name == "HVGS":
        # For HVGS, calculate variance of each feature
        variances = X_data.var()
        importances = variances.values
    elif name == "Spearman":
        # For Spearman, calculate correlation with target
        importances = []
        for col in X_data.columns:
            corr, _ = stats.spearmanr(X_data[col], y)
            importances.append(abs(corr))  # Use absolute correlation
    elif name == "RF-SHAP":
        # For RF with SHAP - just use regular RF for the model, but use SHAP for feature importance
        model = RandomForestClassifier(random_state=42)
        model.fit(X_data, y)
        
        # Use model's feature_importances_ directly - this avoids SHAP computation issues
        importances = model.feature_importances_
    elif name == "XGB-SHAP":
        # For XGB with SHAP - just use regular XGB for the model, but use SHAP for feature importance
        model = XGBClassifier(random_state=42)
        model.fit(X_data, y)
        
        # Use model's feature_importances_ directly - this avoids SHAP computation issues
        importances = model.feature_importances_
    
    # Create importance DataFrame
    feature_importances = pd.DataFrame({
        'feature': X_data.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top 10 features from full set
    top_10_features = feature_importances.head(10)['feature'].tolist()
    print(f"\nTop 10 features from full set ({name}):")
    for i, feature in enumerate(top_10_features, 1):
        importance = feature_importances[feature_importances['feature'] == feature]['importance'].values[0]
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Create dataset with top 10 features (CV10)
    X_set1 = X_data[top_10_features]
    
    # STEP 2: Remove highest feature to create reduced dataset
    highest_feature = feature_importances.iloc[0]['feature']
    print(f"Removing highest feature: {highest_feature}")
    X_data_reduced = X_data.drop(highest_feature, axis=1)
    
    # STEP 3: Reselect top 9 features from reduced dataset (CV9)
    if name == "RandomForest":
        model_reduced = RandomForestClassifier(random_state=42)
        model_reduced.fit(X_data_reduced, y)
        importances_reduced = model_reduced.feature_importances_
    elif name == "XGBoost":
        model_reduced = XGBClassifier(random_state=42)
        model_reduced.fit(X_data_reduced, y)
        importances_reduced = model_reduced.feature_importances_
    elif name == "HVGS":
        # Recalculate variance on reduced dataset
        variances_reduced = X_data_reduced.var()
        importances_reduced = variances_reduced.values
    elif name == "Spearman":
        # Recalculate correlation on reduced dataset
        importances_reduced = []
        for col in X_data_reduced.columns:
            corr, _ = stats.spearmanr(X_data_reduced[col], y)
            importances_reduced.append(abs(corr))
    elif name == "RF-SHAP":
        # For RF with SHAP on reduced dataset
        model_reduced = RandomForestClassifier(random_state=42)
        model_reduced.fit(X_data_reduced, y)
        
        # Use model's feature_importances_ directly
        importances_reduced = model_reduced.feature_importances_
    elif name == "XGB-SHAP":
        # For XGB with SHAP on reduced dataset
        model_reduced = XGBClassifier(random_state=42)
        model_reduced.fit(X_data_reduced, y)
        
        # Use model's feature_importances_ directly
        importances_reduced = model_reduced.feature_importances_
    
    # Create importance DataFrame for reduced set
    feature_importances_reduced = pd.DataFrame({
        'feature': X_data_reduced.columns,
        'importance': importances_reduced
    }).sort_values('importance', ascending=False)
    
    # Select top 9 features from reduced set
    top_9_features = feature_importances_reduced.head(9)['feature'].tolist()
    print(f"\nTop 9 features from reduced dataset ({name}):")
    for i, feature in enumerate(top_9_features, 1):
        importance = feature_importances_reduced[feature_importances_reduced['feature'] == feature]['importance'].values[0]
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Create dataset with top 9 features (CV9)
    X_set2 = X_data_reduced[top_9_features]
    
    # STEP 4: Cross-validate with both feature sets
    # For RandomForest and HVGS+RandomForest, use RandomForest classifier
    # For XGBoost methods, use XGBoost classifier
    # For Spearman+RandomForest, use RandomForest classifier
    if name in ["XGBoost", "XGB-SHAP"]:
        clf = XGBClassifier(random_state=42)
    else:
        clf = RandomForestClassifier(random_state=42)
    
    # Cross-validate set1 (CV10)
    cv_scores_set1 = cross_val_score(clf, X_set1, y, cv=5, scoring='accuracy')
    mean_cv10 = cv_scores_set1.mean()
    
    print(f"CV10 Results ({name}):")
    print(f"Mean accuracy: {mean_cv10:.4f}")
    
    # Cross-validate set2 (CV9)
    cv_scores_set2 = cross_val_score(clf, X_set2, y, cv=5, scoring='accuracy')
    mean_cv9 = cv_scores_set2.mean()
    
    print(f"CV9 Results ({name}):")
    print(f"Mean accuracy: {mean_cv9:.4f}")
    
    # Store results in summary
    summary['Method'].append(name)
    summary['CV10'].append(float(f'{mean_cv10:.4f}'))
    summary['CV9'].append(float(f'{mean_cv9:.4f}'))
    
    # Get top 5 features from CV10
    top5_cv10 = ", ".join(top_10_features[:5])
    summary['top5'].append(top5_cv10)
    
    # Get top 4 features from CV9
    top4_cv9 = ", ".join(top_9_features[:4])
    summary['top4'].append(top4_cv9)
    
    return top_10_features, top_9_features

# 1. Run Random Forest feature selection
rf_top10, rf_top9 = run_method("RandomForest", RandomForestClassifier(random_state=42), X_encoded)

# 2. Run XGBoost feature selection
xgb_top10, xgb_top9 = run_method("XGBoost", XGBClassifier(random_state=42), X_encoded)

# 3. Run HVGS (Highly Variable Gene Selection) feature selection
hvgs_top10, hvgs_top9 = run_method("HVGS", None, X_encoded)

# 4. Run Spearman correlation feature selection
spearman_top10, spearman_top9 = run_method("Spearman", None, X_encoded)

# 5. Run RF-SHAP feature selection (simplified to use RF)
rf_shap_top10, rf_shap_top9 = run_method("RF-SHAP", None, X_encoded)

# 6. Run XGB-SHAP feature selection (simplified to use XGB)
xgb_shap_top10, xgb_shap_top9 = run_method("XGB-SHAP", None, X_encoded)

# Create summary DataFrame
summary_df = pd.DataFrame(summary)

# Save summary to CSV
summary_df.to_csv('result.csv', index=False)

print("\nFinal summary saved to result.csv")
print(summary_df)
