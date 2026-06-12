"""
NFL Draft Prediction Pipeline
==============================
End-to-end ML pipeline for predicting NFL draft selections.
Maximizes ROC-AUC through advanced feature engineering, multi-model
training (LightGBM, XGBoost, CatBoost), Optuna HPO, and ensemble strategies.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_FOLDS = 5
OPTUNA_TRIALS = 50
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

# ============================================================
# STEP 1: EXPLORATORY DATA ANALYSIS
# ============================================================
print("=" * 70)
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

train = pd.read_csv(INPUT_DIR / "train.csv")
test = pd.read_csv(INPUT_DIR / "test.csv")
sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

TARGET = 'Drafted'
ID_COL = 'Id'

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Target column: {TARGET}")
print(f"ID column: {ID_COL}")

# Class imbalance
class_dist = train[TARGET].value_counts()
print(f"\nClass distribution:")
print(class_dist)
print(f"Positive rate: {train[TARGET].mean():.4f}")

# Feature types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [ID_COL, TARGET]]
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# Missing values
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print(f"\nMissing values in train:")
print(missing_train)

missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
print(f"\nMissing values in test:")
print(missing_test)

# Summary statistics
print(f"\nNumerical feature statistics:")
print(train[numeric_cols].describe().round(3).to_string())

# Feature informativeness (point-biserial correlation with target)
from scipy.stats import pointbiserialr
print(f"\nFeature correlation with target (point-biserial):")
correlations = {}
for col in numeric_cols:
    mask = train[col].notna()
    if mask.sum() > 10:
        corr, pval = pointbiserialr(train.loc[mask, TARGET], train.loc[mask, col])
        correlations[col] = corr
corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
print(corr_series.round(4).to_string())

# Save EDA report
eda_report = f"""# NFL Draft Prediction - EDA Report

## Dataset Overview
- **Train samples**: {train.shape[0]}
- **Test samples**: {test.shape[0]}
- **Features**: {train.shape[1] - 2} (excluding Id and target)
- **Target**: `{TARGET}` (binary: 0 = not drafted, 1 = drafted)
- **ID column**: `{ID_COL}`
- **Metric**: ROC-AUC

## Class Distribution
| Class | Count | Proportion |
|-------|-------|------------|
| Not Drafted (0) | {int(class_dist[0.0])} | {class_dist[0.0]/len(train):.4f} |
| Drafted (1) | {int(class_dist[1.0])} | {class_dist[1.0]/len(train):.4f} |

**Positive rate**: {train[TARGET].mean():.4f}

## Feature Types
### Numerical Features ({len(numeric_cols)})
{', '.join(numeric_cols)}

### Categorical Features ({len(cat_cols)})
{', '.join(cat_cols)}

## Missing Values (Train)
| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
"""
for col, count in missing_train.items():
    eda_report += f"| {col} | {count} | {count/len(train)*100:.1f}% |\n"

eda_report += f"""
## Missing Values (Test)
| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
"""
for col, count in missing_test.items():
    eda_report += f"| {col} | {count} | {count/len(test)*100:.1f}% |\n"

eda_report += f"""
## Feature Correlations with Target
| Feature | Correlation |
|---------|-------------|
"""
for col, corr in corr_series.items():
    eda_report += f"| {col} | {corr:.4f} |\n"

eda_report += f"""
## Key Insights
1. **Balanced dataset**: Positive rate ~{train[TARGET].mean():.1%}, reasonably balanced.
2. **Missing data pattern**: Age, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump, Agility_3cone, Shuttle have varying missingness - the ABSENCE of data may itself be informative (players who skip combine drills).
3. **Physical measurements**: Height, Weight, and combine metrics are key features.
4. **Categorical hierarchy**: Position → Position_Type → Player_Type provides multi-level grouping.
5. **School**: High cardinality feature - needs encoding strategies (count/frequency/target encoding).
"""

with open(OUTPUT_DIR / "eda_report.md", 'w', encoding='utf-8') as f:
    f.write(eda_report)
print(f"\nEDA report saved to {OUTPUT_DIR / 'eda_report.md'}")


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def engineer_features(df, is_train=True, train_df=None, target_encodings=None):
    """
    Advanced feature engineering for NFL draft prediction.
    Returns the engineered dataframe and any fitted encodings.
    """
    df = df.copy()
    
    # ---- Physical Composite Features ----
    # BMI = Weight / Height^2
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Speed Score = (Weight * 200) / (Sprint_40yd^4)
    df['SpeedScore'] = np.where(
        df['Sprint_40yd'].notna(),
        (df['Weight'] * 200) / (df['Sprint_40yd'] ** 4),
        np.nan
    )
    
    # Explosiveness Score = Vertical_Jump * Broad_Jump / 1000
    df['ExplosivenessScore'] = df['Vertical_Jump'] * df['Broad_Jump'] / 1000
    
    # Agility Ratio = Agility_3cone / Shuttle
    df['AgilityRatio'] = np.where(
        (df['Agility_3cone'].notna()) & (df['Shuttle'].notna()) & (df['Shuttle'] != 0),
        df['Agility_3cone'] / df['Shuttle'],
        np.nan
    )
    
    # Power Index = Bench_Press_Reps * Weight / 1000
    df['PowerIndex'] = df['Bench_Press_Reps'] * df['Weight'] / 1000
    
    # Jump Ratio = Vertical_Jump / Broad_Jump * 100
    df['JumpRatio'] = np.where(
        (df['Vertical_Jump'].notna()) & (df['Broad_Jump'].notna()) & (df['Broad_Jump'] != 0),
        df['Vertical_Jump'] / df['Broad_Jump'] * 100,
        np.nan
    )
    
    # Height x Weight interaction
    df['HeightWeight'] = df['Height'] * df['Weight']
    
    # Weight-adjusted sprint
    df['WeightAdjustedSprint'] = np.where(
        df['Sprint_40yd'].notna(),
        df['Sprint_40yd'] / df['Weight'] * 100,
        np.nan
    )
    
    # Strength-to-Weight ratio
    df['StrengthWeightRatio'] = np.where(
        (df['Bench_Press_Reps'].notna()) & (df['Weight'] != 0),
        df['Bench_Press_Reps'] / df['Weight'] * 100,
        np.nan
    )
    
    # Athletic composite: combination of all physical metrics (z-scored)
    # Speed + explosiveness combined
    df['SprintBroadCombo'] = np.where(
        (df['Sprint_40yd'].notna()) & (df['Broad_Jump'].notna()),
        df['Broad_Jump'] / df['Sprint_40yd'],
        np.nan
    )
    
    # Vertical per unit height
    df['VerticalPerHeight'] = np.where(
        df['Vertical_Jump'].notna(),
        df['Vertical_Jump'] / df['Height'],
        np.nan
    )
    
    # Total agility = 3cone + shuttle (lower is better, so inverse)
    df['TotalAgility'] = np.where(
        (df['Agility_3cone'].notna()) & (df['Shuttle'].notna()),
        1.0 / (df['Agility_3cone'] + df['Shuttle']),
        np.nan
    )
    
    # Relative athleticism = broad_jump - weight-penalized sprint
    df['RelativeAthleticism'] = np.where(
        (df['Sprint_40yd'].notna()) & (df['Broad_Jump'].notna()),
        df['Broad_Jump'] - df['Sprint_40yd'] * 60,
        np.nan
    )
    
    # Height squared
    df['HeightSq'] = df['Height'] ** 2
    
    # Weight squared
    df['WeightSq'] = df['Weight'] ** 2
    
    # Sprint squared
    df['SprintSq'] = df['Sprint_40yd'] ** 2
    
    # Bench * Vertical (power-explosiveness)
    df['BenchVertical'] = df['Bench_Press_Reps'] * df['Vertical_Jump']
    
    # Interactions
    df['School_Position'] = df['School'].astype(str) + "_" + df['Position_Type'].astype(str)
    df['School_Position_Type'] = df['School'].astype(str) + "_" + df['Player_Type'].astype(str)
    df['Year_Position'] = df['Year'].astype(str) + "_" + df['Position_Type'].astype(str)
    
    # ---- Missing Indicators ----
    missing_indicator_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                              'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in missing_indicator_cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
    
    # Count of missing combine metrics
    combine_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                    'Broad_Jump', 'Agility_3cone', 'Shuttle']
    df['n_missing_combine'] = df[combine_cols].isnull().sum(axis=1)
    df['n_available_combine'] = 6 - df['n_missing_combine']
    
    # ---- Categorical Encoding ----
    if target_encodings is None: target_encodings = {}
    for col in ['Player_Type', 'Position_Type', 'Position']:
        if is_train:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            target_encodings[col] = le
        else:
            le = target_encodings[col]
            classes = list(le.classes_)
            df[f'{col}_encoded'] = df[col].astype(str).map(lambda x: classes.index(x) if x in classes else -1)
    
    # ---- Count Encoding for School ----
    if is_train:
        school_counts = df['School'].value_counts().to_dict()
    else:
        school_counts = train_df['School'].value_counts().to_dict()
    df['School_count'] = df['School'].map(school_counts).fillna(1)
    
    # ---- Frequency Encoding for School ----
    if is_train:
        school_freq = (df['School'].value_counts() / len(df)).to_dict()
    else:
        school_freq = (train_df['School'].value_counts() / len(train_df)).to_dict()
    df['School_freq'] = df['School'].map(school_freq).fillna(0)
    
    # Position count encoding
    if is_train:
        pos_counts = df['Position'].value_counts().to_dict()
    else:
        pos_counts = train_df['Position'].value_counts().to_dict()
    df['Position_count'] = df['Position'].map(pos_counts).fillna(1)
    
    # ---- Position-level aggregations ----
    # For each position type, compute mean/std of key physical features
    ref_df = df if is_train else train_df
    for agg_col in ['Position_Type', 'Player_Type']:
        for feat in ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'BMI']:
            grp = ref_df.groupby(agg_col)[feat].agg(['mean', 'std']).reset_index()
            grp.columns = [agg_col, f'{feat}_{agg_col}_mean', f'{feat}_{agg_col}_std']
            df = df.merge(grp, on=agg_col, how='left')
            # Deviation from group mean
            df[f'{feat}_{agg_col}_diff'] = df[feat] - df[f'{feat}_{agg_col}_mean']
            # Z-Score
            df[f'{feat}_{agg_col}_zscore'] = np.where(
                df[f'{feat}_{agg_col}_std'] > 0,
                df[f'{feat}_{agg_col}_diff'] / df[f'{feat}_{agg_col}_std'],
                0
            )
    
    # ---- Year-based features ----
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'] / 11)
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'] / 11)
    
    # ---- K-Means Clustering ----
    cluster_features = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    if target_encodings.get('kmeans_scaler') is None:
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        
        X_cluster = df[cluster_features].fillna(df[cluster_features].median())
        X_scaled = scaler.fit_transform(X_cluster)
        df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
        
        distances = kmeans.transform(X_scaled)
        for i in range(8):
            df[f'Dist_to_cluster_{i}'] = distances[:, i]
            
        target_encodings['kmeans_scaler'] = scaler
        target_encodings['kmeans_model'] = kmeans
        target_encodings['cluster_medians'] = df[cluster_features].median()
    else:
        scaler = target_encodings['kmeans_scaler']
        kmeans = target_encodings['kmeans_model']
        medians = target_encodings['cluster_medians']
        
        X_cluster = df[cluster_features].fillna(medians)
        X_scaled = scaler.transform(X_cluster)
        df['Cluster_ID'] = kmeans.predict(X_scaled)
        
        distances = kmeans.transform(X_scaled)
        for i in range(8):
            df[f'Dist_to_cluster_{i}'] = distances[:, i]
    
    return df, target_encodings


def add_target_encoding_cv(train_df, test_df, target_col, cat_cols, n_folds=5, seed=42):
    """Smoothed target encoding using cross-validation to prevent leakage."""
    global_mean = train_df[target_col].mean()
    smooth_factor = 10
    
    for col in cat_cols:
        train_df[f'{col}_te'] = np.nan
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(train_df, train_df[target_col]):
            # Compute encoding on train fold
            fold_train = train_df.iloc[train_idx]
            stats = fold_train.groupby(col)[target_col].agg(['mean', 'count'])
            smoothed = (stats['count'] * stats['mean'] + smooth_factor * global_mean) / (stats['count'] + smooth_factor)
            
            # Apply to validation fold
            train_df.loc[train_df.index[val_idx], f'{col}_te'] = train_df.iloc[val_idx][col].map(smoothed)
        
        # Fill any remaining NaN with global mean
        train_df[f'{col}_te'] = train_df[f'{col}_te'].fillna(global_mean)
        
        # For test: use full training data
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smooth_factor * global_mean) / (stats['count'] + smooth_factor)
        test_df[f'{col}_te'] = test_df[col].map(smoothed).fillna(global_mean)
    
    return train_df, test_df


# Apply feature engineering
print("Engineering features for train...")
train_fe, enc = engineer_features(train, is_train=True)
print("Engineering features for test...")
test_fe, _ = engineer_features(test, is_train=False, train_df=train_fe, target_encodings=enc)

# Apply target encoding
te_cols = ['School', 'Position', 'Position_Type', 'Player_Type']
print("Applying smoothed target encoding with CV...")
train_fe, test_fe = add_target_encoding_cv(train_fe, test_fe, TARGET, te_cols, n_folds=N_FOLDS, seed=SEED)

# Drop raw categorical columns and ID
drop_cols = ['School', 'Player_Type', 'Position_Type', 'Position', ID_COL, 'School_Position', 'School_Position_Type', 'Year_Position']
feature_cols = [c for c in train_fe.columns if c not in drop_cols + [TARGET]]
print(f"\nTotal features after engineering: {len(feature_cols)}")
print(f"Feature list: {feature_cols[:20]}... (showing first 20)")

X = train_fe[feature_cols].values
y = train_fe[TARGET].values
X_test = test_fe[feature_cols].values
test_ids = test_fe[ID_COL].values


print(f"X shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test.shape}")




# ============================================================
# STEP 3 & 4: CROSS-VALIDATION + MODEL TRAINING (Default params first)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3 & 4: CROSS-VALIDATION + MODEL TRAINING (Default Params)")
print("=" * 70)

from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage
oof_preds = {}
test_preds = {}
fold_scores = {}
feature_importance_dict = {}

# ---- LightGBM Default ----
print("\n--- Training LightGBM (default params) ---")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'random_state': SEED,
    'n_jobs': -1,
}

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
lgb_fold_scores = []
lgb_importances = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    
    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    lgb_fold_scores.append(fold_auc)
    lgb_importances += model.feature_importances_ / N_FOLDS
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f} (best_iter={model.best_iteration_})")

lgb_mean_auc = roc_auc_score(y, oof_lgb)
print(f"  LightGBM OOF AUC: {lgb_mean_auc:.6f}")
oof_preds['lgb_default'] = oof_lgb
test_preds['lgb_default'] = test_lgb
fold_scores['lgb_default'] = lgb_fold_scores


# ---- XGBoost Default ----
print("\n--- Training XGBoost (default params) ---")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
}

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))
xgb_fold_scores = []
xgb_importances = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=100)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    
    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    xgb_fold_scores.append(fold_auc)
    xgb_importances += model.feature_importances_ / N_FOLDS
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f} (best_iter={model.best_iteration})")

xgb_mean_auc = roc_auc_score(y, oof_xgb)
print(f"  XGBoost OOF AUC: {xgb_mean_auc:.6f}")
oof_preds['xgb_default'] = oof_xgb
test_preds['xgb_default'] = test_xgb
fold_scores['xgb_default'] = xgb_fold_scores


# ---- CatBoost Default ----
print("\n--- Training CatBoost (default params) ---")
cb_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'random_seed': SEED,
    'verbose': 0,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'use_best_model': True,
    'od_type': 'Iter',
    'od_wait': 100,
    'allow_writing_files': False,
}

oof_cb = np.zeros(len(X))
test_cb = np.zeros(len(X_test))
cb_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = cb.CatBoostClassifier(**cb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=0,
    )
    
    oof_cb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cb += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_cb[val_idx])
    cb_fold_scores.append(fold_auc)
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f} (best_iter={model.get_best_iteration()})")

cb_mean_auc = roc_auc_score(y, oof_cb)
print(f"  CatBoost OOF AUC: {cb_mean_auc:.6f}")
oof_preds['cb_default'] = oof_cb
test_preds['cb_default'] = test_cb
fold_scores['cb_default'] = cb_fold_scores


# ============================================================
# STEP 5: HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
print("=" * 70)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

best_params_all = {}

# ---- Optuna LightGBM ----
print("\n--- Optimizing LightGBM ---")
def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10.0, log=True),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.3, 1.0),
        'random_state': SEED,
        'n_jobs': -1,
    }
    
    oof = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y, oof)

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_params_all['lgb'] = study_lgb.best_params
print(f"  Best LightGBM AUC: {study_lgb.best_value:.6f}")
print(f"  Best params: {study_lgb.best_params}")


# ---- Optuna XGBoost ----
print("\n--- Optimizing XGBoost ---")
def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 1e-4, 5.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
    }
    
    oof = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y, oof)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_params_all['xgb'] = study_xgb.best_params
print(f"  Best XGBoost AUC: {study_xgb.best_value:.6f}")
print(f"  Best params: {study_xgb.best_params}")


# ---- Optuna CatBoost ----
print("\n--- Optimizing CatBoost ---")
import gc
def cb_objective(trial):
    params = {
        'iterations': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_seed': SEED,
        'verbose': 0,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'use_best_model': True,
        'od_type': 'Iter',
        'od_wait': 50,
        'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False,
    }
    
    oof = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        del model
        gc.collect()
    return roc_auc_score(y, oof)

study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(cb_objective, n_trials=30, show_progress_bar=False)
best_params_all['catboost'] = study_cb.best_params
print(f"  Best CatBoost AUC: {study_cb.best_value:.6f}")
print(f"  Best params: {study_cb.best_params}")

# Save best parameters
with open(OUTPUT_DIR / 'best_parameters.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_all, f, indent=2)
print(f"\nBest parameters saved to {OUTPUT_DIR / 'best_parameters.json'}")


# ============================================================
# RE-TRAIN WITH OPTIMIZED PARAMS
# ============================================================
print("\n" + "=" * 70)
print("RE-TRAINING WITH OPTIMIZED PARAMETERS")
print("=" * 70)

# ---- Optimized LightGBM ----
print("\n--- Training LightGBM (optimized) ---")
lgb_opt_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_estimators': 5000,
    'random_state': SEED,
    'n_jobs': -1,
    **best_params_all['lgb']
}

oof_lgb_opt = np.zeros(len(X))
test_lgb_opt = np.zeros(len(X_test))
lgb_opt_fold_scores = []
lgb_opt_importances = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_opt_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    
    oof_lgb_opt[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb_opt += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_lgb_opt[val_idx])
    lgb_opt_fold_scores.append(fold_auc)
    lgb_opt_importances += model.feature_importances_ / N_FOLDS
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

lgb_opt_auc = roc_auc_score(y, oof_lgb_opt)
print(f"  LightGBM Optimized OOF AUC: {lgb_opt_auc:.6f}")
oof_preds['lgb_opt'] = oof_lgb_opt
test_preds['lgb_opt'] = test_lgb_opt
fold_scores['lgb_opt'] = lgb_opt_fold_scores

# ---- Optimized XGBoost ----
print("\n--- Training XGBoost (optimized) ---")
xgb_opt_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 5000,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
    **best_params_all['xgb']
}

oof_xgb_opt = np.zeros(len(X))
test_xgb_opt = np.zeros(len(X_test))
xgb_opt_fold_scores = []
xgb_opt_importances = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(**xgb_opt_params, early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_xgb_opt[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb_opt += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_xgb_opt[val_idx])
    xgb_opt_fold_scores.append(fold_auc)
    xgb_opt_importances += model.feature_importances_ / N_FOLDS
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

xgb_opt_auc = roc_auc_score(y, oof_xgb_opt)
print(f"  XGBoost Optimized OOF AUC: {xgb_opt_auc:.6f}")
oof_preds['xgb_opt'] = oof_xgb_opt
test_preds['xgb_opt'] = test_xgb_opt
fold_scores['xgb_opt'] = xgb_opt_fold_scores

# ---- Optimized CatBoost ----
print("\n--- Training CatBoost (optimized) ---")
cb_opt_params = {
    'iterations': 2000,
    'random_seed': SEED,
    'verbose': 0,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'use_best_model': True,
    'od_type': 'Iter',
    'od_wait': 100,
    'bootstrap_type': 'Bernoulli',
    'allow_writing_files': False,
    **best_params_all['catboost']
}

oof_cb_opt = np.zeros(len(X))
test_cb_opt = np.zeros(len(X_test))
cb_opt_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = cb.CatBoostClassifier(**cb_opt_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    
    oof_cb_opt[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cb_opt += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, oof_cb_opt[val_idx])
    cb_opt_fold_scores.append(fold_auc)
    
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

cb_opt_auc = roc_auc_score(y, oof_cb_opt)
print(f"  CatBoost Optimized OOF AUC: {cb_opt_auc:.6f}")
oof_preds['cb_opt'] = oof_cb_opt
test_preds['cb_opt'] = test_cb_opt
fold_scores['cb_opt'] = cb_opt_fold_scores


# ============================================================
# STEP 6: ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: ENSEMBLE STRATEGIES")
print("=" * 70)

from scipy.stats import rankdata
from scipy.optimize import minimize

# Use only optimized models for ensemble
ensemble_models = ['lgb_opt', 'xgb_opt', 'cb_opt']
oof_matrix = np.column_stack([oof_preds[m] for m in ensemble_models])
test_matrix = np.column_stack([test_preds[m] for m in ensemble_models])

# 1) Simple Average
oof_simple_avg = oof_matrix.mean(axis=1)
test_simple_avg = test_matrix.mean(axis=1)
auc_simple = roc_auc_score(y, oof_simple_avg)
print(f"Simple Average OOF AUC: {auc_simple:.6f}")

# 2) Rank Average
oof_ranks = np.column_stack([rankdata(oof_preds[m]) for m in ensemble_models])
test_ranks = np.column_stack([rankdata(test_preds[m]) for m in ensemble_models])
oof_rank_avg = oof_ranks.mean(axis=1)
test_rank_avg = test_ranks.mean(axis=1)
auc_rank = roc_auc_score(y, oof_rank_avg)
print(f"Rank Average OOF AUC: {auc_rank:.6f}")

# 3) Optimized Weighted Average (using scipy minimize)
def neg_auc_weighted(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = oof_matrix @ w
    return -roc_auc_score(y, blend)

# Try multiple starting points
best_result = None
best_neg_auc = 0
for _ in range(50):
    w0 = np.random.dirichlet(np.ones(len(ensemble_models)))
    result = minimize(neg_auc_weighted, w0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    if best_result is None or result.fun < best_neg_auc:
        best_neg_auc = result.fun
        best_result = result

optimal_weights = np.array(best_result.x)
optimal_weights = optimal_weights / optimal_weights.sum()
oof_weighted = oof_matrix @ optimal_weights
test_weighted = test_matrix @ optimal_weights
auc_weighted = roc_auc_score(y, oof_weighted)
print(f"Weighted Average OOF AUC: {auc_weighted:.6f}")
print(f"  Optimal weights: {dict(zip(ensemble_models, optimal_weights.round(4)))}")

# 4) Optimized Weighted Rank Average
def neg_auc_rank_weighted(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = oof_ranks @ w
    return -roc_auc_score(y, blend)

best_result_rank = None
best_neg_auc_rank = 0
for _ in range(50):
    w0 = np.random.dirichlet(np.ones(len(ensemble_models)))
    result = minimize(neg_auc_rank_weighted, w0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    if best_result_rank is None or result.fun < best_neg_auc_rank:
        best_neg_auc_rank = result.fun
        best_result_rank = result

optimal_weights_rank = np.array(best_result_rank.x)
optimal_weights_rank = optimal_weights_rank / optimal_weights_rank.sum()
oof_weighted_rank = oof_ranks @ optimal_weights_rank
test_weighted_rank = test_ranks @ optimal_weights_rank
auc_weighted_rank = roc_auc_score(y, oof_weighted_rank)
print(f"Weighted Rank Average OOF AUC: {auc_weighted_rank:.6f}")
print(f"  Optimal weights: {dict(zip(ensemble_models, optimal_weights_rank.round(4)))}")

# 5) Stacking Classifier
from sklearn.linear_model import LogisticRegression
print("\n--- Training Stacking Classifier ---")
oof_stack = np.zeros(len(y))
test_stack = np.zeros(len(X_test))
for train_idx, val_idx in skf.split(X, y):
    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    meta_model.fit(oof_matrix[train_idx], y[train_idx])
    oof_stack[val_idx] = meta_model.predict_proba(oof_matrix[val_idx])[:, 1]
    test_stack += meta_model.predict_proba(test_matrix)[:, 1] / N_FOLDS
auc_stack = roc_auc_score(y, oof_stack)
print(f"Stacking OOF AUC: {auc_stack:.6f}")

# Choose best ensemble
ensemble_scores = {
    'simple_avg': (auc_simple, test_simple_avg),
    'rank_avg': (auc_rank, test_rank_avg),
    'weighted_avg': (auc_weighted, test_weighted),
    'weighted_rank_avg': (auc_weighted_rank, test_weighted_rank),
    'stacking': (auc_stack, test_stack),
}

# Also consider individual optimized models
for m in ensemble_models:
    ensemble_scores[m] = (roc_auc_score(y, oof_preds[m]), test_preds[m])

best_ensemble_name = max(ensemble_scores, key=lambda k: ensemble_scores[k][0])
best_ensemble_auc = ensemble_scores[best_ensemble_name][0]
best_test_preds = ensemble_scores[best_ensemble_name][1]

# Normalize to [0, 1] if needed (e.g., for rank averages)
if best_test_preds.max() > 1.0 or best_test_preds.min() < 0.0:
    best_test_preds = (best_test_preds - best_test_preds.min()) / (best_test_preds.max() - best_test_preds.min())

print(f"\nBest ensemble: {best_ensemble_name} with OOF AUC: {best_ensemble_auc:.6f}")


# ============================================================
# STEP 7: FINAL SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATING FINAL SUBMISSION")
print("=" * 70)

submission = pd.DataFrame({
    'Id': test_ids.astype(int),
    'Drafted': best_test_preds
})

# Verify alignment with sample submission
assert list(submission['Id']) == list(sample_sub['Id']), "ID order mismatch!"
assert len(submission) == len(sample_sub), "Length mismatch!"
assert submission['Drafted'].between(0, 1).all(), "Predictions out of range!"

submission.to_csv(OUTPUT_DIR / f'submission_{SEED}.csv', index=False)
# Also save to competition root for easy access
submission.to_csv(BASE_DIR / f'submission_{SEED}.csv', index=False)
print(f"Submission saved! Shape: {submission.shape}")
print(f"Predictions range: [{submission['Drafted'].min():.6f}, {submission['Drafted'].max():.6f}]")
print(submission.head(10))


# ============================================================
# STEP 8: OUTPUT DELIVERABLES
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: GENERATING DELIVERABLES")
print("=" * 70)

# Feature importance
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'lgb_importance': lgb_opt_importances,
    'xgb_importance': xgb_opt_importances,
})
fi_df['avg_importance'] = (fi_df['lgb_importance'] + fi_df['xgb_importance']) / 2
fi_df = fi_df.sort_values('avg_importance', ascending=False)
fi_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print(f"Feature importance saved. Top 15 features:")
print(fi_df.head(15).to_string(index=False))

# CV Results
cv_results = []
for model_name in fold_scores:
    scores = fold_scores[model_name]
    cv_results.append({
        'model': model_name,
        'fold_1': scores[0],
        'fold_2': scores[1],
        'fold_3': scores[2],
        'fold_4': scores[3],
        'fold_5': scores[4],
        'mean_auc': np.mean(scores),
        'std_auc': np.std(scores),
        'oof_auc': roc_auc_score(y, oof_preds[model_name]),
    })

# Add ensemble results
for ens_name, (ens_auc, _) in ensemble_scores.items():
    if ens_name not in fold_scores:
        cv_results.append({
            'model': f'ensemble_{ens_name}',
            'fold_1': np.nan, 'fold_2': np.nan, 'fold_3': np.nan,
            'fold_4': np.nan, 'fold_5': np.nan,
            'mean_auc': np.nan, 'std_auc': np.nan,
            'oof_auc': ens_auc,
        })

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
print(f"\nCV results saved.")
print(cv_df.to_string(index=False))

# Experiment Log
experiment_log = f"""# NFL Draft Prediction - Experiment Log

## Pipeline Summary
- **Date**: 2026-06-12
- **Seed**: {SEED}
- **Folds**: {N_FOLDS}
- **Optuna trials per model**: {OPTUNA_TRIALS}

## Feature Engineering
- Total features: {len(feature_cols)}
- Physical composites: BMI, SpeedScore, ExplosivenessScore, AgilityRatio, PowerIndex, JumpRatio
- Interactions: HeightWeight, WeightAdjustedSprint, StrengthWeightRatio, SprintBroadCombo, etc.
- Missing indicators: 7 binary flags + count of missing combine metrics
- Encodings: Label, Count, Frequency, Smoothed Target Encoding (CV)
- Group-level aggregations: Position_Type and Player_Type level stats

## Model Results

### Default Parameters
| Model | OOF AUC |
|-------|---------|
| LightGBM | {roc_auc_score(y, oof_preds['lgb_default']):.6f} |
| XGBoost | {roc_auc_score(y, oof_preds['xgb_default']):.6f} |
| CatBoost | {roc_auc_score(y, oof_preds['cb_default']):.6f} |

### Optimized Parameters
| Model | OOF AUC |
|-------|---------|
| LightGBM | {lgb_opt_auc:.6f} |
| XGBoost | {xgb_opt_auc:.6f} |
| CatBoost | {cb_opt_auc:.6f} |

### Ensembles
| Strategy | OOF AUC |
|----------|---------|
| Simple Average | {auc_simple:.6f} |
| Rank Average | {auc_rank:.6f} |
| Weighted Average | {auc_weighted:.6f} |
| Weighted Rank Average | {auc_weighted_rank:.6f} |

## Final Submission
- **Best strategy**: {best_ensemble_name}
- **OOF AUC**: {best_ensemble_auc:.6f}
- **Prediction range**: [{submission['Drafted'].min():.6f}, {submission['Drafted'].max():.6f}]

## Best Hyperparameters
```json
{json.dumps(best_params_all, indent=2)}
```

## Recommendations for Further Improvement
1. **More feature engineering**: Sport-specific power metrics, Z-score per position
2. **Stacking**: Use OOF predictions as features for a second-level model
3. **Neural networks**: TabNet or 1D-CNN for tabular data
4. **Pseudo-labeling**: Use confident test predictions to augment training
5. **Feature selection**: Boruta or recursive feature elimination
6. **More Optuna trials**: 100-200 trials per model
7. **Additional models**: Extra Trees, Random Forest, Ridge regression
"""

with open(OUTPUT_DIR / 'experiment_log.md', 'w', encoding='utf-8') as f:
    f.write(experiment_log)
print(f"\nExperiment log saved.")


# ============================================================
# STEP 9: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: FINAL SUMMARY")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   NFL DRAFT PREDICTION                       ║
║                   PIPELINE COMPLETE                          ║
╠══════════════════════════════════════════════════════════════╣
║  Best Model/Ensemble: {best_ensemble_name:<37s} ║
║  Cross-validation ROC-AUC: {best_ensemble_auc:<32.6f} ║
║                                                              ║
║  Individual Model OOF AUCs:                                  ║
║    LightGBM (optimized):  {lgb_opt_auc:<33.6f} ║
║    XGBoost (optimized):   {xgb_opt_auc:<33.6f} ║
║    CatBoost (optimized):  {cb_opt_auc:<33.6f} ║
║                                                              ║
║  Ensemble Strategies:                                        ║
║    Simple Average:        {auc_simple:<33.6f} ║
║    Rank Average:          {auc_rank:<33.6f} ║
║    Weighted Average:      {auc_weighted:<33.6f} ║
║    Weighted Rank Average: {auc_weighted_rank:<33.6f} ║
║                                                              ║
║  Files Generated:                                            ║
║    ✓ submission.csv                                          ║
║    ✓ eda_report.md                                           ║
║    ✓ feature_importance.csv                                  ║
║    ✓ cv_results.csv                                          ║
║    ✓ best_parameters.json                                    ║
║    ✓ experiment_log.md                                       ║
╚══════════════════════════════════════════════════════════════╝
""")

print("All deliverables generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Submission file: {BASE_DIR / 'submission.csv'}")
