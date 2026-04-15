import pandas as pd

import pandas as pd
from scipy import stats

# 1. Create the structured data
data = {
    "Model": ["Base model", "Base model", "Clini feature model", "Clini feature model", 
              "Dosi feature model", "Dosi feature model", "Radio feature model", "Radio feature model"],
    "Metric": ["ROC AUC", "F1-score", "ROC AUC", "F1-score", "ROC AUC", "F1-score", "ROC AUC", "F1-score"],
    "Fold 1": [0.8250, 0.6667, 0.7250, 0.4000, 0.7875, 0.6667, 0.8500, 0.0000],
    "Fold 2": [0.6875, 0.4000, 0.4625, 0.2500, 0.7750, 0.4348, 0.6375, 0.4706],
    "Fold 3": [0.9191, 0.5714, 0.7500, 0.3200, 0.8571, 0.5000, 0.8235, 0.6667],
    "Fold 4": [0.7941, 0.4615, 0.6250, 0.3200, 0.8250, 0.5514, 0.9118, 0.5333],
    "Fold 5": [0.7794, 0.3333, 0.8235, 0.4286, 0.7353, 0.2857, 0.5588, 0.3636]
}

df = pd.DataFrame(data)

# 2. Extract fold scores for comparison
# Comparison: Each model vs. Base Model
metrics = df["Metric"].unique()
other_models = [m for m in df["Model"].unique() if m != "Base model"]
test_results = []

for metric in metrics:
    # Get scores for the base model
    base_scores = df[(df["Model"] == "Base model") & (df["Metric"] == metric)].iloc[0, 2:].values.astype(float)
    
    for model in other_models:
        # Get scores for the comparison model
        comp_scores = df[(df["Model"] == model) & (df["Metric"] == metric)].iloc[0, 2:].values.astype(float)
        
        # Perform Paired T-test (assumes normality)
        t_stat, p_val_t = stats.ttest_rel(base_scores, comp_scores)
        
        # Perform Wilcoxon Signed-Rank test (non-parametric, better for small N=5)
        w_stat, p_val_w = stats.wilcoxon(base_scores, comp_scores)
        
        test_results.append({
            "Metric": metric,
            "Comparison": f"Base vs {model}",
            "Base Mean": base_scores.mean(),
            "Model Mean": comp_scores.mean(),
            "T-test p-value": p_val_t,
            "Wilcoxon p-value": p_val_w
        })
        print(f"Comparison: Base vs {model} for {metric}")
        print(f"Base scores: {base_scores.mean():.4f}, Model scores: {comp_scores.mean():.4f}")
        print(f"T-test p-value: {p_val_t:.4f}, Wilcoxon p-value: {p_val_w:.4f}\n")


data = {
    "Model": ["Base model", "Base model", "Clini feature model", "Clini feature model", "Dosi feature model", "Dosi feature model", "Radio feature model", "Radio feature model"],
    "Metric": ["ROC AUC", "F1-score", "ROC AUC", "F1-score", "ROC AUC", "F1-score", "ROC AUC", "F1-score"],
    "Fold 1": [ 0.800000, 0.571429, 0.742857, 0.615385,  # Clini
        0.371429, 0.588235,  # Dosi
        0.828571, 0.588235   # Radio
    ],
    "Fold 2": [
        0.942857, 0.625000,  # Base model
        0.514286, 0.500000,  # Clini
        0.514286, 0.500000,  # Dosi
        0.800000, 0.769231   # Radio
    ],
    "Fold 3": [
        0.685714, 0.545455,  # Base model
        0.657143, 0.588235,  # Clini
        0.571429, 0.588235,  # Dosi
        0.857143, 0.769231   # Radio
    ],
    "Fold 4": [
        0.766667, 0.615385,  # Base model
        0.633333, 0.500000,  # Clini
        0.266667, 0.625000,  # Dosi
        0.633333, 0.625000   # Radio
    ],
    "Fold 5": [
        0.766667, 0.625000,  # Base model
        0.700000, 0.769231,  # Clini
        0.333333, 0.571429,  # Dosi
        0.800000, 0.625000   # Radio
    ]
}

df = pd.DataFrame(data)

# 2. Extract fold scores for comparison
# Comparison: Each model vs. Base Model
metrics = df["Metric"].unique()
other_models = [m for m in df["Model"].unique() if m != "Base model"]
test_results = []

for metric in metrics:
    # Get scores for the base model
    base_scores = df[(df["Model"] == "Base model") & (df["Metric"] == metric)].iloc[0, 2:].values.astype(float)
    
    for model in other_models:
        # Get scores for the comparison model
        comp_scores = df[(df["Model"] == model) & (df["Metric"] == metric)].iloc[0, 2:].values.astype(float)
        
        # Perform Paired T-test (assumes normality)
        t_stat, p_val_t = stats.ttest_rel(base_scores, comp_scores)
        
        # Perform Wilcoxon Signed-Rank test (non-parametric, better for small N=5)
        w_stat, p_val_w = stats.wilcoxon(base_scores, comp_scores)
        
        test_results.append({
            "Metric": metric,
            "Comparison": f"Base vs {model}",
            "Base Mean": base_scores.mean(),
            "Model Mean": comp_scores.mean(),
            "T-test p-value": p_val_t,
            "Wilcoxon p-value": p_val_w
        })
        print(f"Comparison: Base vs {model} for {metric}")
        print(f"Base scores: {base_scores.mean():.4f}, Model scores: {comp_scores.mean():.4f}")
        print(f"T-test p-value: {p_val_t:.4f}, Wilcoxon p-value: {p_val_w:.4f}\n")
