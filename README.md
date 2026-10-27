# Binary Tropical Cyclones over the Western North Pacific

Data and core model code accompanying the manuscript "Prolonged Coexistence of Binary Tropical Cyclones over the Western North Pacific in a Warming Climate".

## Files

- `Data_BTC.csv`: Six-hourly BTC track, intensity, and event information for 1945–2023. Environmental fields are excluded.
- `BTC_FTIN_nohup.py`: BiLSTM model definition, training, and prediction functions.
- `BTC_Data_Description.txt`: Definitions and units of all fields in the supplied CSV.
- `requirements.txt`: Python dependencies for model training and prediction.
- `README.md`: Repository contents, model interface, and usage.

## Model

The model contains a bidirectional LSTM with 48 units, dropout of 0.30, a dense layer with 64 units and ReLU activation, and 24 output values. Training uses Adam with a learning rate of 0.001, mean squared error loss, early stopping, and learning-rate reduction. The default batch size is 32 and the maximum training length is 200 epochs.

The module accepts arrays prepared externally. CSV reading, environmental matching, feature calculation, window construction, missing-value handling, standardization, train/validation/test splitting, and cross-validation or sensitivity-experiment loops are outside its scope. Environmental predictors may be included as extra input features when prepared externally.

## Usage

```bash
python -m pip install -r requirements.txt
```

Import the functions after preparing the following arrays:

- `X_train`, `X_validation`, and `X_test`: shape `(samples, 4, n_features)`, representing four consecutive six-hour input times.
- `y_train` and `y_validation`: shape `(samples, 24)`, ordered as lead 1 member 1, lead 1 member 2, through lead 12 member 1 and lead 12 member 2. The twelve leads cover 6–72 hours.

```python
from BTC_FTIN_nohup import build_model, train_model, predict_intensity

model = build_model(n_features=X_train.shape[-1], seed=11)
history = train_model(
    model, X_train, y_train, X_validation, y_validation,
    model_path="BTC_FTIN.keras",
)
prediction = predict_intensity(model, X_test)  # (samples, 12, 2)
```

This example assumes the named arrays already exist. Running the module as a standalone script does not load the CSV or start training.

Optional training arguments accept externally prepared sample weights. For symmetric inference, pass externally prepared member-swapped inputs using `swapped_x_test`. Predictions have the same scale and units as the training targets. If targets were standardized, apply the inverse transformation outside this module. Fit any preprocessing using training data only.
