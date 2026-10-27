#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core BTC BiLSTM model: definition, training, and prediction.

Call these functions with arrays prepared outside this module.
Inputs: (samples, 4, n_features).
Training targets: (samples, 24), ordered by lead time then member (1, 2).
Predictions: (samples, 12, 2), in the same scale as the training targets.
"""

LOOKBACK = 4
FORECAST_STEPS = 12


def build_model(n_features, seed=11):
    """Build the original BiLSTM(48) -> Dropout -> Dense(64) -> Dense(24)."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    l2 = tf.keras.regularizers.l2

    inputs = layers.Input(shape=(LOOKBACK, n_features), name='input')
    hidden = layers.Bidirectional(
        layers.LSTM(
            48,
            return_sequences=False,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-5),
        ),
        name='bilstm',
    )(inputs)
    hidden = layers.Dropout(0.30, name='dropout')(hidden)
    hidden = layers.Dense(
        64, activation='relu', kernel_regularizer=l2(1e-4), name='dense'
    )(hidden)
    outputs = layers.Dense(FORECAST_STEPS * 2, name='forecast')(hidden)
    model = tf.keras.Model(inputs, outputs, name='BTC_FTIN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae'],
    )
    return model


def train_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    *,
    sample_weight=None,
    validation_sample_weight=None,
    epochs=200,
    batch_size=32,
    model_path=None,
    verbose=1,
):
    """Fit prepared arrays and optionally save the trained model to .keras.

    Supply already flattened targets of shape (samples, 24). Any sample
    weights or member-swap augmentation must be prepared by the caller.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=20, min_delta=1e-5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-6, verbose=0,
        ),
    ]
    validation_data = (x_validation, y_validation)
    if validation_sample_weight is not None:
        validation_data += (validation_sample_weight,)
    history = model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    if model_path is not None:
        model.save(model_path)
    return history


def predict_intensity(model, x_test, *, swapped_x_test=None, batch_size=32):
    """Predict 12 leads for two members, without scaling or unit conversion.

    For symmetric inference, optionally pass the already prepared
    member-swapped inputs, processed using the same training preprocessor.
    The swapped outputs are restored to the original member order before
    averaging. Without swapped inputs, return the direct model prediction.
    """
    prediction = model.predict(
        x_test, batch_size=batch_size, verbose=0
    ).reshape(-1, FORECAST_STEPS, 2)
    if swapped_x_test is not None:
        swapped = model.predict(
            swapped_x_test, batch_size=batch_size, verbose=0
        ).reshape(-1, FORECAST_STEPS, 2)
        if swapped.shape != prediction.shape:
            raise ValueError('Direct and swapped inputs must have matching sample counts.')
        prediction = (prediction + swapped[:, :, ::-1]) / 2
    return prediction
