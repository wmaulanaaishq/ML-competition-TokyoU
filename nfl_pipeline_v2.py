"""
NFL Draft Prediction — Pipeline V2
===================================
GCI Data Science Competition | Matsuo Lab, University of Tokyo

Key improvements over V1:
  1. IterativeImputer (MICE) for smart missing-data handling
  2. Focused feature engineering (fewer but stronger features)  
  3. Model diversity: ExtraTrees + Ridge alongside GBDT trio
  4. Deeper Optuna search (50 trials)
  5. Multi-strategy ensemble with automatic best selection
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import gc
import sys
from pathlib import Path
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIG
# ============================================================
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_FOLDS = 5
OPTUNA_TRIALS = 50
BASE_DIR = Path('.')
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

print("=" * 70)
print(f"  NFL DRAFT PREDICTION — PIPELINE V2  (SEED={SEED})")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
train = pd.read_csv(INPUT_DIR / 'train.csv')
test = pd.read_csv(INPUT_DIR / 'test.csv')
sample_sub = pd.read_csv(INPUT_DIR / 'sample_submission.csv')

TARGET = 'Drafted'
ID_COL = 'Id'

print(f"\nTrain: {train.shape}, Test: {test.shape}")
print(f"Target rate: {train[TARGET].mean():.4f}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

combine_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                'Broad_Jump', 'Agility_3cone', 'Shuttle']
num_raw = ['Age', 'Height', 'Weight'] + combine_cols

def build_features(df, ref_df=None, imputer=None, te_maps=None):
    """Build features. ref_df is the training dataframe for test-time."""
    d = df.copy()
    is_train = ref_df is None
    ref = d if is_train else ref_df
    artifacts = {}

    # ---- Missing indicators (before imputation) ----
    for c in combine_cols + ['Age']:
        d[f'{c}_miss'] = d[c].isnull().astype(int)
    d['n_miss_combine'] = d[combine_cols].isnull().sum(axis=1)

    # ---- Iterative Imputer (MICE) ----
    if imputer is None:
        imputer = IterativeImputer(
            max_iter=20, random_state=SEED, sample_posterior=False,
            initial_strategy='median', n_nearest_features=None
        )
        d[num_raw] = imputer.fit_transform(d[num_raw])
    else:
        d[num_raw] = imputer.transform(d[num_raw])
    artifacts['imputer'] = imputer

    # ---- Core physical features ----
    d['BMI'] = d['Weight'] / (d['Height'] ** 2)
    d['SpeedScore'] = (d['Weight'] * 200) / (d['Sprint_40yd'] ** 4)
    d['Explosiveness'] = d['Vertical_Jump'] * d['Broad_Jump'] / 1000
    d['PowerIndex'] = d['Bench_Press_Reps'] * d['Weight'] / 1000
    d['AgilityRatio'] = d['Agility_3cone'] / d['Shuttle'].replace(0, np.nan)
    d['HeightWeight'] = d['Height'] * d['Weight']
    d['WeightAdjSprint'] = d['Sprint_40yd'] / d['Weight'] * 100
    d['StrengthWeight'] = d['Bench_Press_Reps'] / d['Weight'] * 100
    d['SprintBroad'] = d['Broad_Jump'] / d['Sprint_40yd']
    d['VertPerHeight'] = d['Vertical_Jump'] / d['Height']
    d['TotalAgility'] = 1.0 / (d['Agility_3cone'] + d['Shuttle'])
    d['BenchVertical'] = d['Bench_Press_Reps'] * d['Vertical_Jump']
    d['RelAthletic'] = d['Broad_Jump'] - d['Sprint_40yd'] * 60

    # ---- Rank within Position_Type (percentile) ----
    for feat in ['Sprint_40yd', 'Weight', 'Bench_Press_Reps', 'Vertical_Jump',
                 'Broad_Jump', 'SpeedScore', 'BMI']:
        grp = ref.groupby('Position_Type')[feat]
        grp_mean = grp.transform('mean') if is_train else d['Position_Type'].map(ref.groupby('Position_Type')[feat].mean())
        grp_std = grp.transform('std') if is_train else d['Position_Type'].map(ref.groupby('Position_Type')[feat].std())
        grp_std = grp_std.replace(0, 1)
        d[f'{feat}_pos_zscore'] = (d[feat] - grp_mean) / grp_std

    # ---- Year-Position interaction percentiles ----
    for feat in ['Sprint_40yd', 'Weight', 'Vertical_Jump']:
        key = d['Year'].astype(str) + '_' + d['Position_Type'].astype(str)
        ref_key = ref['Year'].astype(str) + '_' + ref['Position_Type'].astype(str)
        grp_mean = ref.groupby(ref_key)[feat].mean()
        grp_std = ref.groupby(ref_key)[feat].std().replace(0, 1)
        d[f'{feat}_yrpos_zscore'] = (d[feat] - key.map(grp_mean).fillna(d[feat].mean())) / key.map(grp_std).fillna(1)

    # ---- Label Encoding ----
    cat_cols = ['Player_Type', 'Position_Type', 'Position']
    if te_maps is None:
        te_maps = {}
    for c in cat_cols:
        if c not in te_maps:
            le = LabelEncoder()
            d[f'{c}_enc'] = le.fit_transform(d[c].astype(str))
            te_maps[c] = le
        else:
            le = te_maps[c]
            cls = list(le.classes_)
            d[f'{c}_enc'] = d[c].astype(str).map(lambda x, cls=cls: cls.index(x) if x in cls else -1)

    # ---- Count / Frequency encoding ----
    for c in ['School', 'Position']:
        counts = ref[c].value_counts().to_dict()
        d[f'{c}_count'] = d[c].map(counts).fillna(1)
        d[f'{c}_freq'] = d[f'{c}_count'] / len(ref)

    artifacts['te_maps'] = te_maps
    return d, artifacts


print("Engineering features for train...")
train_fe, art = build_features(train)
print("Engineering features for test...")
test_fe, _ = build_features(test, ref_df=train_fe, imputer=art['imputer'], te_maps=art['te_maps'])

# ---- OOF Target Encoding ----
print("Applying OOF target encoding...")
global_mean = train_fe[TARGET].mean()
smooth = 10

te_cat_cols = ['School', 'Position', 'Position_Type', 'Player_Type']
skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for c in te_cat_cols:
    train_fe[f'{c}_te'] = np.nan
    for tr_idx, va_idx in skf_te.split(train_fe, train_fe[TARGET]):
        fold_train = train_fe.iloc[tr_idx]
        stats = fold_train.groupby(c)[TARGET].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
        train_fe.loc[train_fe.index[va_idx], f'{c}_te'] = train_fe.iloc[va_idx][c].map(smoothed)
    train_fe[f'{c}_te'] = train_fe[f'{c}_te'].fillna(global_mean)
    
    stats_full = train_fe.groupby(c)[TARGET].agg(['mean', 'count'])
    smoothed_full = (stats_full['count'] * stats_full['mean'] + smooth * global_mean) / (stats_full['count'] + smooth)
    test_fe[f'{c}_te'] = test_fe[c].map(smoothed_full).fillna(global_mean)

# ---- Prepare final arrays ----
drop_cols = [ID_COL, TARGET, 'School', 'Position', 'Position_Type', 'Player_Type', 'Year']
feature_cols = [c for c in train_fe.columns if c not in drop_cols and train_fe[c].dtype in ['float64', 'int64', 'float32', 'int32']]

X = train_fe[feature_cols].values.astype(np.float32)
y = train_fe[TARGET].values
X_test = test_fe[feature_cols].values.astype(np.float32)
test_ids = test_fe[ID_COL].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

# ============================================================
# STEP 3: CROSS-VALIDATION FRAMEWORK
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: TRAINING WITH DEFAULT PARAMETERS")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = {}
test_preds = {}
fold_scores_all = {}

def train_model(name, ModelClass, params, use_early_stop=True, fit_params=None):
    """Generic CV training loop."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = ModelClass(**params)
        fp = fit_params or {}
        
        if name.startswith('lgb'):
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        elif name.startswith('xgb'):
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif name.startswith('cb'):
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        else:
            model.fit(X_tr, y_tr)

        if hasattr(model, 'predict_proba'):
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        else:
            oof[va_idx] = model.decision_function(X_va)
            test_pred += model.decision_function(X_test) / N_FOLDS

        fold_auc = roc_auc_score(y_va, oof[va_idx])
        scores.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")
        
        del model; gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"  {name} OOF AUC: {oof_auc:.6f} (mean={np.mean(scores):.6f}, std={np.std(scores):.6f})")
    oof_preds[name] = oof
    test_preds[name] = test_pred
    fold_scores_all[name] = scores
    return oof_auc

# ---- Default models ----
print("\n--- LightGBM (default) ---")
train_model('lgb_def', lgb.LGBMClassifier, dict(
    objective='binary', metric='auc', n_estimators=3000, learning_rate=0.05,
    max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20, random_state=SEED,
    n_jobs=-1, verbosity=-1
))

print("\n--- XGBoost (default) ---")
train_model('xgb_def', xgb.XGBClassifier, dict(
    objective='binary:logistic', eval_metric='auc', n_estimators=3000,
    learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=20, random_state=SEED,
    n_jobs=-1, verbosity=0, early_stopping_rounds=100
))

print("\n--- CatBoost (default) ---")
train_model('cb_def', cb.CatBoostClassifier, dict(
    iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
    random_seed=SEED, verbose=0, eval_metric='AUC', loss_function='Logloss',
    use_best_model=True, od_type='Iter', od_wait=100, allow_writing_files=False
))

print("\n--- ExtraTrees ---")
train_model('et_def', ExtraTreesClassifier, dict(
    n_estimators=1000, max_depth=12, min_samples_leaf=5, max_features='sqrt',
    random_state=SEED, n_jobs=-1
))

print("\n--- Ridge Classifier ---")
# Scale data for linear model
scaler_ridge = StandardScaler()
X_scaled = scaler_ridge.fit_transform(np.nan_to_num(X))
X_test_scaled = scaler_ridge.transform(np.nan_to_num(X_test))

oof_ridge = np.zeros(len(X))
test_ridge = np.zeros(len(X_test))
ridge_scores = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = RidgeClassifier(alpha=1.0, random_state=SEED)
    model.fit(X_scaled[tr_idx], y[tr_idx])
    oof_ridge[va_idx] = model.decision_function(X_scaled[va_idx])
    test_ridge += model.decision_function(X_test_scaled) / N_FOLDS
    fold_auc = roc_auc_score(y[va_idx], oof_ridge[va_idx])
    ridge_scores.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")
oof_preds['ridge'] = oof_ridge
test_preds['ridge'] = test_ridge
fold_scores_all['ridge'] = ridge_scores
print(f"  Ridge OOF AUC: {roc_auc_score(y, oof_ridge):.6f}")

# ============================================================
# STEP 4: OPTUNA HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: HYPERPARAMETER OPTIMIZATION (Optuna)")
print("=" * 70)

best_params = {}

# ---- LightGBM ----
print("\n--- Optimizing LightGBM ---")
def lgb_objective(trial):
    p = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'n_estimators': 3000, 'random_state': SEED, 'n_jobs': -1,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 8, 96),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10, log=True),
    }
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**p)
        m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        del m; gc.collect()
    return roc_auc_score(y, oof)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS)
best_params['lgb'] = study.best_params
print(f"  Best LGB AUC: {study.best_value:.6f}")

# ---- XGBoost ----
print("\n--- Optimizing XGBoost ---")
def xgb_objective(trial):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
        'n_estimators': 3000, 'random_state': SEED, 'n_jobs': -1,
        'early_stopping_rounds': 100,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 1e-4, 5, log=True),
    }
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        m = xgb.XGBClassifier(**p)
        m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        del m; gc.collect()
    return roc_auc_score(y, oof)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS)
best_params['xgb'] = study.best_params
print(f"  Best XGB AUC: {study.best_value:.6f}")

# ---- CatBoost ----
print("\n--- Optimizing CatBoost ---")
def cb_objective(trial):
    p = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_seed': SEED, 'verbose': 0, 'eval_metric': 'AUC',
        'loss_function': 'Logloss', 'use_best_model': True,
        'od_type': 'Iter', 'od_wait': 50,
        'bootstrap_type': 'Bernoulli', 'allow_writing_files': False,
    }
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        m = cb.CatBoostClassifier(**p)
        m.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        del m; gc.collect()
    return roc_auc_score(y, oof)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(cb_objective, n_trials=OPTUNA_TRIALS)
best_params['cb'] = study.best_params
print(f"  Best CB AUC: {study.best_value:.6f}")

# ---- ExtraTrees ----
print("\n--- Optimizing ExtraTrees ---")
def et_objective(trial):
    p = {
        'n_estimators': 1500,
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'max_features': trial.suggest_float('max_features', 0.2, 1.0),
        'random_state': SEED, 'n_jobs': -1,
    }
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        m = ExtraTreesClassifier(**p)
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        del m; gc.collect()
    return roc_auc_score(y, oof)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(et_objective, n_trials=30)
best_params['et'] = study.best_params
print(f"  Best ET AUC: {study.best_value:.6f}")

# Save best params
with open(OUTPUT_DIR / 'best_parameters_v2.json', 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, default=str)

# ============================================================
# STEP 5: RETRAIN WITH OPTIMIZED PARAMS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: RE-TRAINING WITH OPTIMIZED PARAMETERS")
print("=" * 70)

bp = best_params['lgb']
print("\n--- LightGBM (optimized) ---")
auc_lgb = train_model('lgb_opt', lgb.LGBMClassifier, dict(
    objective='binary', metric='auc', verbosity=-1, n_estimators=3000,
    random_state=SEED, n_jobs=-1,
    learning_rate=bp['lr'], max_depth=bp['max_depth'], num_leaves=bp['num_leaves'],
    subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
    reg_alpha=bp['reg_alpha'], reg_lambda=bp['reg_lambda'],
    min_child_samples=bp['min_child_samples'], min_child_weight=bp['min_child_weight'],
))

bp = best_params['xgb']
print("\n--- XGBoost (optimized) ---")
auc_xgb = train_model('xgb_opt', xgb.XGBClassifier, dict(
    objective='binary:logistic', eval_metric='auc', verbosity=0,
    n_estimators=3000, random_state=SEED, n_jobs=-1, early_stopping_rounds=100,
    learning_rate=bp['lr'], max_depth=bp['max_depth'],
    subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
    colsample_bylevel=bp['colsample_bylevel'],
    reg_alpha=bp['reg_alpha'], reg_lambda=bp['reg_lambda'],
    min_child_weight=bp['min_child_weight'], gamma=bp['gamma'],
))

bp = best_params['cb']
print("\n--- CatBoost (optimized) ---")
auc_cb = train_model('cb_opt', cb.CatBoostClassifier, dict(
    iterations=2000, random_seed=SEED, verbose=0, eval_metric='AUC',
    loss_function='Logloss', use_best_model=True, od_type='Iter', od_wait=50,
    bootstrap_type='Bernoulli', allow_writing_files=False,
    learning_rate=bp['lr'], depth=bp['depth'], l2_leaf_reg=bp['l2_leaf_reg'],
    subsample=bp['subsample'], colsample_bylevel=bp['colsample_bylevel'],
    min_data_in_leaf=bp['min_data_in_leaf'],
))

bp = best_params['et']
print("\n--- ExtraTrees (optimized) ---")
auc_et = train_model('et_opt', ExtraTreesClassifier, dict(
    n_estimators=1500, random_state=SEED, n_jobs=-1,
    max_depth=bp['max_depth'], min_samples_leaf=bp['min_samples_leaf'],
    min_samples_split=bp['min_samples_split'], max_features=bp['max_features'],
))

# ============================================================
# STEP 6: ENSEMBLE STRATEGIES
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: ENSEMBLE STRATEGIES")
print("=" * 70)

ensemble_models = ['lgb_opt', 'xgb_opt', 'cb_opt', 'et_opt', 'ridge']
oof_matrix = np.column_stack([oof_preds[m] for m in ensemble_models])
test_matrix = np.column_stack([test_preds[m] for m in ensemble_models])

# 1) Simple Average
oof_simple = oof_matrix.mean(axis=1)
auc_simple = roc_auc_score(y, oof_simple)
print(f"Simple Average OOF AUC: {auc_simple:.6f}")

# 2) Rank Average
oof_ranks = np.column_stack([rankdata(oof_preds[m]) for m in ensemble_models])
test_ranks = np.column_stack([rankdata(test_preds[m]) for m in ensemble_models])
oof_rank_avg = oof_ranks.mean(axis=1)
auc_rank = roc_auc_score(y, oof_rank_avg)
print(f"Rank Average OOF AUC: {auc_rank:.6f}")

# 3) Weighted Average (optimized)
def neg_auc_w(weights):
    w = np.array(weights); w = w / w.sum()
    return -roc_auc_score(y, oof_matrix @ w)

best_res = None
for _ in range(100):
    w0 = np.random.dirichlet(np.ones(len(ensemble_models)))
    res = minimize(neg_auc_w, w0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    if best_res is None or res.fun < best_res.fun:
        best_res = res

opt_w = np.array(best_res.x); opt_w = opt_w / opt_w.sum()
oof_weighted = oof_matrix @ opt_w
test_weighted = test_matrix @ opt_w
auc_weighted = roc_auc_score(y, oof_weighted)
print(f"Weighted Average OOF AUC: {auc_weighted:.6f}")
print(f"  Weights: {dict(zip(ensemble_models, opt_w.round(4)))}")

# 4) Weighted Rank Average
def neg_auc_rw(weights):
    w = np.array(weights); w = w / w.sum()
    return -roc_auc_score(y, oof_ranks @ w)

best_res_r = None
for _ in range(100):
    w0 = np.random.dirichlet(np.ones(len(ensemble_models)))
    res = minimize(neg_auc_rw, w0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    if best_res_r is None or res.fun < best_res_r.fun:
        best_res_r = res

opt_wr = np.array(best_res_r.x); opt_wr = opt_wr / opt_wr.sum()
oof_wrank = oof_ranks @ opt_wr
test_wrank = test_ranks @ opt_wr
auc_wrank = roc_auc_score(y, oof_wrank)
print(f"Weighted Rank Average OOF AUC: {auc_wrank:.6f}")
print(f"  Weights: {dict(zip(ensemble_models, opt_wr.round(4)))}")

# 5) Stacking
print("\n--- Stacking Classifier ---")
oof_stack = np.zeros(len(y))
test_stack = np.zeros(len(X_test))
for tr_idx, va_idx in skf.split(X, y):
    meta = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    meta.fit(oof_matrix[tr_idx], y[tr_idx])
    oof_stack[va_idx] = meta.predict_proba(oof_matrix[va_idx])[:, 1]
    test_stack += meta.predict_proba(test_matrix)[:, 1] / N_FOLDS
auc_stack = roc_auc_score(y, oof_stack)
print(f"Stacking OOF AUC: {auc_stack:.6f}")

# ---- Best ensemble ----
results = {
    'simple_avg': (auc_simple, oof_simple, oof_matrix.mean(axis=1), test_matrix.mean(axis=1)),
    'rank_avg': (auc_rank, oof_rank_avg, oof_rank_avg, test_ranks.mean(axis=1)),
    'weighted_avg': (auc_weighted, oof_weighted, oof_weighted, test_weighted),
    'weighted_rank': (auc_wrank, oof_wrank, oof_wrank, test_wrank),
    'stacking': (auc_stack, oof_stack, oof_stack, test_stack),
}

# Also check individual models
for m in ensemble_models:
    auc_m = roc_auc_score(y, oof_preds[m])
    results[m] = (auc_m, oof_preds[m], oof_preds[m], test_preds[m])

best_name = max(results, key=lambda k: results[k][0])
best_auc = results[best_name][0]
best_test_pred = results[best_name][3]

print(f"\nBest: {best_name} with OOF AUC: {best_auc:.6f}")

# ============================================================
# STEP 7: GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATING SUBMISSION")
print("=" * 70)

# Normalize predictions to [0, 1]
pred_final = best_test_pred.copy()
pred_final = (pred_final - pred_final.min()) / (pred_final.max() - pred_final.min())

submission = pd.DataFrame({ID_COL: test_ids, TARGET: pred_final})
submission[ID_COL] = submission[ID_COL].astype(int)

assert len(submission) == len(sample_sub), "Submission length mismatch!"
assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"

sub_path = OUTPUT_DIR / f'submission_v2_{SEED}.csv'
submission.to_csv(sub_path, index=False)
print(f"Submission saved to: {sub_path}")
print(f"Shape: {submission.shape}, Range: [{submission[TARGET].min():.6f}, {submission[TARGET].max():.6f}]")
print(submission.head(10))

# ============================================================
# STEP 8: SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: SAVING RESULTS")
print("=" * 70)

# CV results
rows = []
for name, scores in fold_scores_all.items():
    row = {'model': name}
    for i, s in enumerate(scores):
        row[f'fold_{i+1}'] = round(s, 6)
    row['mean_auc'] = round(np.mean(scores), 6)
    row['std_auc'] = round(np.std(scores), 6)
    row['oof_auc'] = round(roc_auc_score(y, oof_preds[name]), 6)
    rows.append(row)

cv_df = pd.DataFrame(rows)
cv_df.to_csv(OUTPUT_DIR / 'cv_results_v2.csv', index=False, encoding='utf-8')
print("CV results saved.")
print(cv_df.to_string(index=False))

# Experiment log
log = f"""# Experiment Log — Pipeline V2 (Seed={SEED})

## Best Ensemble: {best_name}
## Best OOF AUC: {best_auc:.6f}

## Ensemble Comparison
| Strategy | OOF AUC |
|----------|---------|
"""
for name, (auc, _, _, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    log += f"| {name} | {auc:.6f} |\n"

with open(OUTPUT_DIR / 'experiment_log_v2.md', 'w', encoding='utf-8') as f:
    f.write(log)

print(f"\nPIPELINE V2 COMPLETE! Best OOF AUC = {best_auc:.6f}")
