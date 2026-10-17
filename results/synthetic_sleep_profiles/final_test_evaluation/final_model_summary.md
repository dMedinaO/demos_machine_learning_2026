# Final selected model — synthetic_sleep_profiles

## Frozen selection

- **Task:** multiclass_classification
- **Selected family:** Extra Trees
- **Estimator:** ExtraTreesClassifier
- **Primary selection metric:** Macro F1
- **Development CV value:** 0.481772
- **Final TEST primary value:** 0.410640
- **TEST minus development CV:** -0.071132
- **Development data:** TRAIN + VALIDATION
- **Final evaluation data:** protected TEST only
- **Refit performed in TEST notebook:** no

## Hyperparameters

- `n_estimators`: `150`
- `max_depth`: `None`
- `min_samples_leaf`: `2`
- `max_features`: `sqrt`
- `class_weight`: `balanced`

## Selected predictors

- `age`
- `bmi`
- `sleep_duration_hours`
- `sleep_efficiency_pct`
- `sleep_latency_min`
- `night_awakenings`
- `screen_time_evening_min`
- `stress_score_0_10`
- `social_jetlag_hours`
- `bedtime_variability_min`
- `sleep_efficiency_deficit_pct`
- `sleep_fragmentation_per_hour`
- `circadian_disruption_index`
- `behavioral_risk_count`
- `shift_worker`
- `anxiety_history`

## Interpretation rule

The model, hyperparameters, preprocessing, and feature selector were frozen before TEST evaluation. TEST results quantify held-out generalization and must not be used to reopen model selection or hyperparameter optimization.
