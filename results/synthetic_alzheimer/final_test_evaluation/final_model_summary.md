# Final selected model — synthetic_alzheimer

## Frozen selection

- **Task:** binary_classification
- **Selected family:** SVM
- **Estimator:** SVC
- **Primary selection metric:** MCC
- **Development CV value:** 0.441254
- **Final TEST primary value:** 0.438128
- **TEST minus development CV:** -0.003126
- **Development data:** TRAIN + VALIDATION
- **Final evaluation data:** protected TEST only
- **Refit performed in TEST notebook:** no

## Hyperparameters

- `kernel`: `linear`
- `C`: `0.04235455276099052`
- `class_weight`: `None`

## Selected predictors

- `age`
- `bmi`
- `waist_hip_ratio`
- `systolic_bp_mmhg`
- `diastolic_bp_mmhg`
- `pulse_pressure_mmhg`
- `mean_arterial_pressure_mmhg`
- `cardiometabolic_risk_count`
- `neurobehavioral_risk_count`
- `physical_inactivity`
- `hypertension`
- `diabetes_type2`
- `dyslipidemia`
- `cardiovascular_disease`
- `stroke_history`
- `depression_history`
- `family_history_alzheimer`
- `chronic_kidney_disease`
- `hearing_loss`
- `allergy_medication`
- `allergy_food`

## Interpretation rule

The model, hyperparameters, preprocessing, and feature selector were frozen before TEST evaluation. TEST results quantify held-out generalization and must not be used to reopen model selection or hyperparameter optimization.
