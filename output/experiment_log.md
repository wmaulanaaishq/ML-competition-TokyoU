# NFL Draft Prediction - Experiment Log

## Pipeline Summary
- **Date**: 2026-06-12
- **Seed**: 777
- **Folds**: 5
- **Optuna trials per model**: 50

## Feature Engineering
- Total features: 97
- Physical composites: BMI, SpeedScore, ExplosivenessScore, AgilityRatio, PowerIndex, JumpRatio
- Interactions: HeightWeight, WeightAdjustedSprint, StrengthWeightRatio, SprintBroadCombo, etc.
- Missing indicators: 7 binary flags + count of missing combine metrics
- Encodings: Label, Count, Frequency, Smoothed Target Encoding (CV)
- Group-level aggregations: Position_Type and Player_Type level stats

## Model Results

### Default Parameters
| Model | OOF AUC |
|-------|---------|
| LightGBM | 0.808222 |
| XGBoost | 0.828884 |
| CatBoost | 0.831322 |

### Optimized Parameters
| Model | OOF AUC |
|-------|---------|
| LightGBM | 0.837067 |
| XGBoost | 0.837427 |
| CatBoost | 0.837241 |

### Ensembles
| Strategy | OOF AUC |
|----------|---------|
| Simple Average | 0.843194 |
| Rank Average | 0.843628 |
| Weighted Average | 0.843279 |
| Weighted Rank Average | 0.843685 |

## Final Submission
- **Best strategy**: weighted_rank_avg
- **OOF AUC**: 0.843685
- **Prediction range**: [0.000000, 1.000000]

## Best Hyperparameters
```json
{
  "lgb": {
    "learning_rate": 0.1039860659846293,
    "max_depth": 8,
    "num_leaves": 115,
    "subsample": 0.7598768071338529,
    "colsample_bytree": 0.414559391679623,
    "reg_alpha": 0.3054648849277249,
    "reg_lambda": 1.6728243690614313,
    "min_child_samples": 48,
    "min_child_weight": 0.00011713740184471722,
    "feature_fraction_bynode": 0.48394726320578474
  },
  "xgb": {
    "learning_rate": 0.13279548634187446,
    "max_depth": 6,
    "subsample": 0.5343455260770003,
    "colsample_bytree": 0.7085946022801533,
    "colsample_bylevel": 0.8138639312807303,
    "reg_alpha": 0.11135391451052044,
    "reg_lambda": 0.0017877992368468705,
    "min_child_weight": 38,
    "gamma": 0.5985249932214983
  },
  "catboost": {
    "learning_rate": 0.07145040556931584,
    "depth": 5,
    "l2_leaf_reg": 2.029229193505671,
    "subsample": 0.5958155155107768,
    "colsample_bylevel": 0.6905067053355629,
    "min_data_in_leaf": 56
  }
}
```

## Recommendations for Further Improvement
1. **More feature engineering**: Sport-specific power metrics, Z-score per position
2. **Stacking**: Use OOF predictions as features for a second-level model
3. **Neural networks**: TabNet or 1D-CNN for tabular data
4. **Pseudo-labeling**: Use confident test predictions to augment training
5. **Feature selection**: Boruta or recursive feature elimination
6. **More Optuna trials**: 100-200 trials per model
7. **Additional models**: Extra Trees, Random Forest, Ridge regression
