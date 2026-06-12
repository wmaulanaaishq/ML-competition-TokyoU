# NFL Draft Prediction - EDA Report

## Dataset Overview
- **Train samples**: 2781
- **Test samples**: 696
- **Features**: 14 (excluding Id and target)
- **Target**: `Drafted` (binary: 0 = not drafted, 1 = drafted)
- **ID column**: `Id`
- **Metric**: ROC-AUC

## Class Distribution
| Class | Count | Proportion |
|-------|-------|------------|
| Not Drafted (0) | 978 | 0.3517 |
| Drafted (1) | 1803 | 0.6483 |

**Positive rate**: 0.6483

## Feature Types
### Numerical Features (10)
Year, Age, Height, Weight, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump, Agility_3cone, Shuttle

### Categorical Features (4)
School, Player_Type, Position_Type, Position

## Missing Values (Train)
| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| Agility_3cone | 970 | 34.9% |
| Shuttle | 912 | 32.8% |
| Bench_Press_Reps | 721 | 25.9% |
| Broad_Jump | 581 | 20.9% |
| Vertical_Jump | 554 | 19.9% |
| Age | 435 | 15.6% |
| Sprint_40yd | 145 | 5.2% |

## Missing Values (Test)
| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| Agility_3cone | 247 | 35.5% |
| Shuttle | 228 | 32.8% |
| Bench_Press_Reps | 184 | 26.4% |
| Broad_Jump | 147 | 21.1% |
| Vertical_Jump | 143 | 20.5% |
| Age | 115 | 16.5% |
| Sprint_40yd | 29 | 4.2% |

## Feature Correlations with Target
| Feature | Correlation |
|---------|-------------|
| Age | -0.1331 |
| Bench_Press_Reps | 0.1322 |
| Broad_Jump | 0.1127 |
| Sprint_40yd | -0.1124 |
| Vertical_Jump | 0.1094 |
| Shuttle | -0.0879 |
| Agility_3cone | -0.0734 |
| Weight | 0.0710 |
| Height | 0.0469 |
| Year | -0.0246 |

## Key Insights
1. **Balanced dataset**: Positive rate ~64.8%, reasonably balanced.
2. **Missing data pattern**: Age, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump, Agility_3cone, Shuttle have varying missingness - the ABSENCE of data may itself be informative (players who skip combine drills).
3. **Physical measurements**: Height, Weight, and combine metrics are key features.
4. **Categorical hierarchy**: Position → Position_Type → Player_Type provides multi-level grouping.
5. **School**: High cardinality feature - needs encoding strategies (count/frequency/target encoding).
