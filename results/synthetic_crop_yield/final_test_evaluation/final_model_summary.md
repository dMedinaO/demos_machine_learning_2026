# Final selected model — synthetic_crop_yield

## Frozen selection

- **Task:** regression
- **Selected family:** SVR
- **Estimator:** SVR
- **Primary selection metric:** MAE (t/ha)
- **Development CV value:** 0.794175
- **Final TEST primary value:** 0.891841
- **TEST minus development CV:** +0.097666
- **Development data:** TRAIN + VALIDATION
- **Final evaluation data:** protected TEST only
- **Refit performed in TEST notebook:** no

## Hyperparameters

- `kernel`: `rbf`
- `C`: `10.594635089049195`
- `epsilon`: `0.013559579357622604`
- `gamma`: `0.01155978355611923`

## Selected predictors

- `season_rainfall_mm`
- `soil_organic_matter_pct`
- `nitrogen_kg_ha`
- `irrigation_mm`
- `solar_radiation_mj_m2_day`
- `field_slope_pct`
- `pest_pressure_score_0_10`
- `soil_moisture_pct`
- `total_water_mm`
- `npk_total_kg_ha`
- `n_to_p_ratio`
- `soil_moisture_pct_squared`
- `water_x_drought`
- `pest_x_fungicide`
- `organic_matter_x_moisture`
- `cultivar_improved`
- `drought_event`

## Interpretation rule

The model, hyperparameters, preprocessing, and feature selector were frozen before TEST evaluation. TEST results quantify held-out generalization and must not be used to reopen model selection or hyperparameter optimization.
