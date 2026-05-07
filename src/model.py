"""Modèle CNN simple pour la classification fresh / rotten."""

from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_model(image_size, learning_rate=0.001, dropout_rate=0.3):
    """Construit et compile le baseline_model."""
    image_height, image_width = image_size
    input_shape = (image_height, image_width, 3)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_model",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model
